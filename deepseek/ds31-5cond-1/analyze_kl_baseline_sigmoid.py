#!/usr/bin/env python3
"""
KL-from-Calibration-Baseline analysis for sigmoid-gated MoE (DeepSeek V3.1).

Two metric families on two different normalizations:
  1. KL baseline: sigmoid → normalize full 256-dim to simplex → KL(token || cal_mean)
     This is an analysis-only proxy distribution, NOT the model's actual sparse
     routing distribution. It provides consistent 256-dim vectors for KL computation.
  2. Entropy/effective experts: uses the proper DeepSeek grouped noaux_tc top-k
     reconstruction from deepseek_router.py → H / log2(8)
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    from scipy.special import expit
except ImportError:
    def expit(x):
        x = np.asarray(x, dtype=np.float64)
        pos = x >= 0
        out = np.empty_like(x)
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        e = np.exp(x[~pos])
        out[~pos] = e / (1.0 + e)
        return out

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

from deepseek_router import (
    N_ROUTED_EXPERTS,
    TOP_K,
    reconstruct_probs,
)
from analysis_common import inspect_router_layers, read_metadata, read_region_boundaries

# ── Constants ─────────────────────────────────────────────────────────────
MANUAL_EXCLUDED_LAYERS = set()
EPS = 1e-30

CONDITIONS = list("ABCDE")
COND_LABELS = {
    "A": "this system",
    "B": "a system",
    "C": "your system",
    "D": "the system",
    "E": "their system",
}

KEY_COMPARISONS = [
    ("A", "B", "this vs a — core self-ref"),
    ("C", "B", "your vs a — strongest addressivity"),
    ("A", "C", "this vs your — gradient"),
    ("D", "B", "the vs a — weak control"),
    ("E", "B", "their vs a — weak control"),
]


# ── Helpers ───────────────────────────────────────────────────────────────

def sigmoid_full_probs(logits: np.ndarray) -> np.ndarray:
    """Sigmoid → normalize full 256 to sum-to-1 (analysis-only proxy).

    This is NOT the model's actual routing distribution. It provides a dense
    256-dim probability vector for consistent KL computation across tokens
    (the actual top-K set varies per token, breaking KL dimensionality).
    """
    scores = expit(logits.astype(np.float64))
    totals = scores.sum(axis=-1, keepdims=True)
    totals = np.where(totals < EPS, EPS, totals)
    return scores / totals


def kl_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) per row, in bits."""
    p_safe = np.clip(p, EPS, None)
    q_safe = np.clip(q, EPS, None)
    return np.sum(p_safe * np.log2(p_safe / q_safe), axis=-1)


def entropy_bits(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits, per row."""
    p = np.clip(probs, EPS, None)
    return -np.sum(p * np.log2(p), axis=-1)

def parse_prompt_id(pid: str):
    prefix = pid.split("_", 1)[0]
    stripped = prefix.lstrip("P")
    pair_num = int(stripped[:2])
    condition = stripped[2]
    category = pid.split("_", 1)[1] if "_" in pid else ""
    return pair_num, condition, category


# ── Per-prompt analysis ───────────────────────────────────────────────────

def analyze_prompt(prompt_dir: Path):
    pid = prompt_dir.name
    pair_num, condition, category = parse_prompt_id(pid)
    info = read_metadata(prompt_dir)
    if "n_tokens_prompt" not in info:
        raise ValueError(f"No n_tokens_prompt in {prompt_dir / 'metadata.txt'}")
    n_tokens = int(info["n_tokens_prompt"])
    router_dir = prompt_dir / "router"

    regions = read_region_boundaries(prompt_dir)
    if regions is None:
        raise ValueError(f"Missing exact region boundaries in {prompt_dir / 'metadata.txt'}")
    cal1_s, cal1_e = regions["cal1"]
    manip_s, manip_e = regions["manip"]
    cal2_s, cal2_e = regions["cal2"]

    validation = inspect_router_layers(
        prompt_dir,
        expected_rows=n_tokens,
        manual_excluded=MANUAL_EXCLUDED_LAYERS,
    )
    if validation.n_experts != N_ROUTED_EXPERTS:
        raise ValueError(
            f"Expected {N_ROUTED_EXPERTS} experts, got {validation.n_experts} in {prompt_dir}"
        )
    if validation.corrupt_layers:
        print(f"    Corrupt or unexpected .npy (skipped): layers {validation.corrupt_layers}")
    if validation.row_mismatch_layers:
        print(f"    Row-count mismatches (skipped): layers {validation.row_mismatch_layers}")
    if validation.excluded_layers:
        print(f"    Excluded layers: {validation.excluded_layers}")

    # Collect per-layer metrics
    layer_kl_manip = []
    layer_kl_manip_last = []
    layer_kl_cal2 = []
    layer_re = []
    layer_re_last = []
    layer_eff_experts = []
    layer_max_weight = []

    for layer in validation.good_layers:
        npy_file = router_dir / f"ffn_moe_logits-{layer}.npy"
        logits = np.load(str(npy_file))

        # ── Metric family 1: KL from cal baseline (dense 256-dim proxy) ──
        full_probs = sigmoid_full_probs(logits)
        cal_baseline = full_probs[cal1_s:cal1_e].mean(axis=0)  # [256]

        manip_probs = full_probs[manip_s:manip_e]
        if manip_probs.shape[0] > 0:
            kl_manip = kl_div(manip_probs, cal_baseline[None, :])
            layer_kl_manip.append(float(np.mean(kl_manip)))
            layer_kl_manip_last.append(float(kl_manip[-1]))
        else:
            layer_kl_manip.append(0.0)
            layer_kl_manip_last.append(0.0)

        cal2_probs = full_probs[cal2_s:cal2_e]
        if cal2_probs.shape[0] > 0:
            kl_cal2 = kl_div(cal2_probs, cal_baseline[None, :])
            layer_kl_cal2.append(float(np.mean(kl_cal2)))
        else:
            layer_kl_cal2.append(0.0)

        # ── Metric family 2: proper DeepSeek grouped top-k reconstruction ──
        topk_probs = reconstruct_probs(logits)
        h = entropy_bits(topk_probs)  # per token
        re_norm = h / np.log2(TOP_K)
        layer_re.append(float(np.mean(re_norm)))
        layer_re_last.append(float(re_norm[-1]))

        # Effective experts and max weight on last token
        topk_last = topk_probs[-1]
        nonzero = topk_last[topk_last > EPS]
        h_last = -np.sum(nonzero * np.log2(nonzero))
        layer_eff_experts.append(float(2 ** h_last))
        layer_max_weight.append(float(np.max(topk_last)))

    return {
        "id": pid,
        "pair": pair_num,
        "condition": condition,
        "category": category,
        "n_tokens": n_tokens,
        "n_cal1_tokens": cal1_e - cal1_s,
        "n_manip_tokens": manip_e - manip_s,
        "n_cal2_tokens": cal2_e - cal2_s,
        "n_layers": len(validation.good_layers),
        "excluded_layers": validation.excluded_layers,
        # KL family (dense 256-dim sigmoid-normalized proxy)
        "kl_manip_mean": float(np.mean(layer_kl_manip)),
        "kl_manip_last": float(np.mean(layer_kl_manip_last)),
        "kl_cal2_mean": float(np.mean(layer_kl_cal2)),
        # Entropy family (proper grouped noaux_tc top-k reconstruction)
        "re_topk_mean": float(np.mean(layer_re)),
        "re_topk_last": float(np.mean(layer_re_last)),
        "eff_experts_last": float(np.mean(layer_eff_experts)),
        "max_weight_last": float(np.mean(layer_max_weight)),
    }


# ── Pairwise comparisons ─────────────────────────────────────────────────

def pairwise_test(vals_x, vals_y, metric_name):
    diffs = np.array(vals_x) - np.array(vals_y)
    n = len(diffs)
    x_gt_y = int(np.sum(diffs > 0))
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se = std_diff / np.sqrt(n) if n > 0 else 0.0
    # Paired t-interval CI
    ci95 = (float(mean_diff - 1.96 * se), float(mean_diff + 1.96 * se))

    result = {
        "metric": metric_name,
        "n": n,
        "x_gt_y": x_gt_y,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci95_method": "paired_t_normal_approx",
        "ci95": ci95,
    }
    if wilcoxon is not None and n >= 6:
        try:
            w, p = wilcoxon(diffs)
            result["wilcoxon_W"] = float(w)
            result["wilcoxon_p"] = float(p)
        except Exception:
            pass
    return result


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--tsv", default="prompts_selfref_5cond.tsv")
    parser.add_argument("--prompt-suite", default="prompt_suite.json")
    parser.add_argument("--results-json", default="results_kl_baseline_sigmoid.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    prompt_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and (d / "router").exists()
    ])
    print(f"Found {len(prompt_dirs)} prompt directories")
    print("Region boundaries: exact token offsets from metadata.txt")
    print(f"KL distribution: dense sigmoid-normalized 256-dim (analysis proxy)")
    print(f"Entropy distribution: grouped noaux_tc top-{TOP_K} via deepseek_router.py")

    # Analyze all prompts
    all_results = []
    excluded_layers_union = set()
    for i, pd in enumerate(prompt_dirs):
        r = analyze_prompt(pd)
        all_results.append(r)
        excluded_layers_union.update(r["excluded_layers"])
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(prompt_dirs)}] {r['id']}: "
                  f"cal1={r['n_cal1_tokens']} manip={r['n_manip_tokens']} cal2={r['n_cal2_tokens']} | "
                  f"KL_manip={r['kl_manip_mean']:.6f} "
                  f"RE={r['re_topk_mean']:.6f} "
                  f"KL_cal2={r['kl_cal2_mean']:.6f}")

    # Group by condition
    by_cond = {c: [] for c in CONDITIONS}
    for r in all_results:
        by_cond[r["condition"]].append(r)

    # ── Condition means ──
    metrics = ["kl_manip_mean", "kl_manip_last", "kl_cal2_mean",
               "re_topk_mean", "re_topk_last", "eff_experts_last", "max_weight_last"]

    print("\n" + "=" * 90)
    print("CONDITION MEANS")
    print("=" * 90)
    header = f"{'Cond':<6} {'Label':<16}"
    for m in metrics:
        header += f" {m:>16}"
    print(header)
    print("-" * len(header))

    cond_means = {}
    for c in CONDITIONS:
        vals = by_cond[c]
        means = {m: float(np.mean([v[m] for v in vals])) for m in metrics}
        cond_means[c] = means
        row = f"{c:<6} {COND_LABELS[c]:<16}"
        for m in metrics:
            row += f" {means[m]:>16.6f}"
        print(row)

    # ── Pairwise comparisons ──
    print("\n" + "=" * 90)
    print("PAIRWISE COMPARISONS (uncorrected p-values)")
    print("=" * 90)

    test_metrics = ["kl_manip_mean", "kl_manip_last", "re_topk_mean", "re_topk_last"]
    all_comparisons = {}
    all_pvals = []  # for Holm-Bonferroni correction

    for cx, cy, desc in KEY_COMPARISONS:
        print(f"\n--- {cx} vs {cy}: {COND_LABELS[cx]} vs {COND_LABELS[cy]} ({desc}) ---")
        comp_key = f"{cx}_vs_{cy}"
        all_comparisons[comp_key] = {"description": desc}

        x_by_pair = {r["pair"]: r for r in by_cond[cx]}
        y_by_pair = {r["pair"]: r for r in by_cond[cy]}
        shared_pairs = sorted(set(x_by_pair) & set(y_by_pair))

        for m in test_metrics:
            vals_x = [x_by_pair[p][m] for p in shared_pairs]
            vals_y = [y_by_pair[p][m] for p in shared_pairs]
            t = pairwise_test(vals_x, vals_y, m)
            all_comparisons[comp_key][m] = t
            if "wilcoxon_p" in t:
                all_pvals.append((comp_key, m, t["wilcoxon_p"]))

            p_str = f"p={t['wilcoxon_p']:.4e}" if "wilcoxon_p" in t else "no scipy"
            print(f"  {m:<20}: {cx}>{cy} {t['x_gt_y']:>2}/{t['n']}, "
                  f"mean_diff={t['mean_diff']:+.6f}, "
                  f"CI=[{t['ci95'][0]:+.6f}, {t['ci95'][1]:+.6f}], "
                  f"{p_str}")

    # ── All 10 pairwise (for completeness) ──
    print("\n" + "=" * 90)
    print("ALL PAIRWISE (kl_manip_mean, uncorrected)")
    print("=" * 90)
    print(f"{'Pair':<8} {'X>Y':>5} {'mean_diff':>12} {'p-value':>12}")
    for cx, cy in combinations(CONDITIONS, 2):
        x_by_pair = {r["pair"]: r for r in by_cond[cx]}
        y_by_pair = {r["pair"]: r for r in by_cond[cy]}
        shared = sorted(set(x_by_pair) & set(y_by_pair))
        vals_x = [x_by_pair[p]["kl_manip_mean"] for p in shared]
        vals_y = [y_by_pair[p]["kl_manip_mean"] for p in shared]
        t = pairwise_test(vals_x, vals_y, "kl_manip_mean")
        comp_key = f"{cx}_vs_{cy}"
        if comp_key not in all_comparisons:
            all_comparisons[comp_key] = {"kl_manip_mean": t}
        if "wilcoxon_p" in t:
            all_pvals.append((comp_key, "kl_manip_mean_full", t["wilcoxon_p"]))
        p_str = f"{t['wilcoxon_p']:.4e}" if "wilcoxon_p" in t else "n/a"
        print(f"  {cx}v{cy:<5} {t['x_gt_y']:>2}/30 {t['mean_diff']:>+12.6f} {p_str:>12}")

    # ── Holm-Bonferroni correction ──
    if all_pvals:
        all_pvals.sort(key=lambda x: x[2])
        n_tests = len(all_pvals)
        print(f"\n{'='*90}")
        print(f"HOLM-BONFERRONI CORRECTION ({n_tests} tests)")
        print(f"{'='*90}")
        print(f"  {'Comparison':<30} {'Metric':<25} {'raw_p':>12} {'threshold':>12} {'sig':>5}")
        for rank, (comp, metric, raw_p) in enumerate(all_pvals):
            threshold = 0.05 / (n_tests - rank)
            sig = "***" if raw_p < 0.001 / (n_tests - rank) else \
                  "**" if raw_p < 0.01 / (n_tests - rank) else \
                  "*" if raw_p < threshold else ""
            print(f"  {comp:<30} {metric:<25} {raw_p:>12.4e} {threshold:>12.4e} {sig:>5}")

    # ── Save JSON ──
    output = {
        "experiment": "ds31_5cond_selfref_kl_baseline_sigmoid",
        "model": "DeepSeek-V3-0324",
        "architecture": "deepseek2",
        "n_experts": N_ROUTED_EXPERTS,
        "top_k": TOP_K,
        "gating_function": "sigmoid",
        "kl_distribution": "dense_sigmoid_full256_normalized_to_simplex (analysis proxy, not actual routing)",
        "entropy_distribution": "grouped_noaux_tc_topk_normalized (deepseek_router.reconstruct_probs)",
        "entropy_normalization": f"shannon_div_log2_{TOP_K}",
        "region_boundaries": "exact token offsets written to metadata during/post capture",
        "ci_method": "paired_t_normal_approx_1.96_se",
        "multiple_comparison_note": "uncorrected Wilcoxon p-values; Holm-Bonferroni correction shown separately",
        "excluded_layers": sorted(excluded_layers_union),
        "conditions": COND_LABELS,
        "condition_means": cond_means,
        "comparisons": all_comparisons,
        "per_prompt": all_results,
    }

    with open(args.results_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {args.results_json}")


if __name__ == "__main__":
    main()
