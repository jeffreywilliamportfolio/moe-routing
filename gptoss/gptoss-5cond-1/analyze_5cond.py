#!/usr/bin/env python3
"""
GPT-OSS 120B -- 5-Condition Analysis (runs on instance, reads .npy files).

Replicates paired-2 methodology:
  - Routing reconstruction: top-4 by raw logit, then softmax on selected 4
  - RE normalized by log2(4)
  - KL-to-baseline: dense 128-dim softmax proxy, KL(manip || cal1_mean)
  - Region boundaries: proportional char->token mapping
  - Layer 35 excluded (truncation bug)

5 conditions: A=this, B=a, C=your, D=the, E=their
150 prompts: 30 pairs x 5 conditions
All 10 pairwise Wilcoxon comparisons.
"""
import glob
import itertools
import json
import os
import pathlib
import sys

import numpy as np
from scipy.stats import wilcoxon

from gptoss_router import (
    N_EXPERTS,
    TOP_K,
    RECONSTRUCTION_NAME,
    reconstruct_probs,
    normalized_entropy,
    softmax_full_probs,
    kl_divergence,
)

PROMPT_SUITE = "prompt_suite.json"
TSV = "prompts_5cond_gptoss.tsv"
OUTPUT_DIR = "output"
RESULTS_FILE = "results_5cond_prefill_gptoss.json"

EXCLUDED_LAYERS = {35}
CONDITIONS = list("ABCDE")
COND_LABELS = {"A": "this", "B": "a", "C": "your", "D": "the", "E": "their"}


def get_metadata(prompt_dir):
    meta = prompt_dir / "metadata.txt"
    info = {}
    if meta.exists():
        for line in meta.read_text().strip().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                info[key] = value
    return int(info.get("n_tokens_prompt", 0)), int(info.get("n_tokens_generated", 0))


def load_prompt_texts(tsv_path):
    texts = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1]
    return texts


def estimate_region_boundaries(prompt_text, cal_paragraph, n_tokens):
    cal_clean = cal_paragraph.replace("\n", " ").replace("\t", " ")
    cal1_start_char = prompt_text.find(cal_clean)
    if cal1_start_char < 0:
        return None
    cal1_end_char = cal1_start_char + len(cal_clean)
    cal2_start_char = prompt_text.find(cal_clean, cal1_end_char)
    if cal2_start_char < 0:
        return None
    cal2_end_char = cal2_start_char + len(cal_clean)
    total_chars = len(prompt_text)
    if total_chars == 0:
        return None

    def char_to_tok(cp):
        return max(0, min(n_tokens, int(round(cp / total_chars * n_tokens))))

    return {
        "cal1": (char_to_tok(cal1_start_char), char_to_tok(cal1_end_char)),
        "manip": (char_to_tok(cal1_end_char), char_to_tok(cal2_start_char)),
        "cal2": (char_to_tok(cal2_start_char), char_to_tok(cal2_end_char)),
    }


def compute_metrics(prompt_dir, n_prompt, regions=None):
    router_dir = prompt_dir / "router"
    if not router_dir.exists():
        return None

    files = sorted(
        glob.glob(str(router_dir / "ffn_moe_logits-*.npy")),
        key=lambda fp: int(pathlib.Path(fp).stem.split("-")[1]),
    )
    if not files or n_prompt == 0:
        return None

    n_experts = np.load(files[0]).shape[1]

    shapes = {}
    for fp in files:
        li = int(pathlib.Path(fp).stem.split("-")[1])
        shapes[li] = np.load(fp).shape[0]
    median_rows = np.median(list(shapes.values()))

    good_layers = sorted([
        li for li in shapes
        if shapes[li] >= median_rows * 0.5 and li not in EXCLUDED_LAYERS
    ])
    excluded_layers = sorted(set(shapes.keys()) - set(good_layers))

    has_regions = regions is not None
    if has_regions:
        cal1_s, cal1_e = regions["cal1"]
        manip_s, manip_e = regions["manip"]
        cal2_s, cal2_e = regions["cal2"]

    per_layer = []
    all_ent = []
    last_token_ents = []
    layer_kl_manip = []
    layer_kl_cal2 = []

    for li in good_layers:
        fp = router_dir / f"ffn_moe_logits-{li}.npy"
        logits = np.load(str(fp))
        n_rows = min(logits.shape[0], n_prompt)

        probs = reconstruct_probs(logits[:n_rows])
        ent = normalized_entropy(probs)

        last_ent = float(ent[n_rows - 1])
        last_token_ents.append(last_ent)

        layer_info = {
            "layer": li,
            "mean_entropy": float(np.mean(ent)),
            "std_entropy": float(np.std(ent)),
            "last_token_entropy": last_ent,
            "n_rows": int(logits.shape[0]),
        }

        # ent > 0 filters zero-padded rows from truncated layers (e.g. layer 35
        # bug), not legitimate data. With top-4 softmax, real tokens always
        # have nonzero entropy because softmax distributes mass across all 4.
        valid = ent > 0
        if valid.sum() > 0:
            all_ent.extend(ent[valid].tolist())

        if has_regions and cal1_e > cal1_s and manip_e > manip_s:
            # Bound region slices to actual rows available in this layer
            r_cal1_s, r_cal1_e = min(cal1_s, n_rows), min(cal1_e, n_rows)
            r_manip_s, r_manip_e = min(manip_s, n_rows), min(manip_e, n_rows)
            r_cal2_s, r_cal2_e = min(cal2_s, n_rows), min(cal2_e, n_rows)

            if r_cal1_e <= r_cal1_s or r_manip_e <= r_manip_s:
                per_layer.append(layer_info)
                continue

            full_probs = softmax_full_probs(logits[:n_rows])
            cal_baseline = full_probs[r_cal1_s:r_cal1_e].mean(axis=0)
            cal_baseline = np.clip(cal_baseline, 1e-30, None)

            manip_probs = full_probs[r_manip_s:r_manip_e]
            kl_manip = kl_divergence(manip_probs, cal_baseline[None, :])
            kl_manip = np.clip(kl_manip, 0, None)
            layer_kl_manip.append(float(np.mean(kl_manip)))
            layer_info["kl_manip_mean"] = float(np.mean(kl_manip))

            if r_cal2_e > r_cal2_s:
                cal2_probs = full_probs[r_cal2_s:r_cal2_e]
                kl_cal2 = kl_divergence(cal2_probs, cal_baseline[None, :])
                kl_cal2 = np.clip(kl_cal2, 0, None)
                layer_kl_cal2.append(float(np.mean(kl_cal2)))
                layer_info["kl_cal2_mean"] = float(np.mean(kl_cal2))

        per_layer.append(layer_info)

    result = {
        "prefill_re": float(np.mean(all_ent)) if all_ent else 0.0,
        "last_token_re": float(np.mean(last_token_ents)) if last_token_ents else 0.0,
        "n_layers": len(good_layers),
        "n_layers_excluded": excluded_layers,
        "n_experts": n_experts,
        "per_layer": per_layer,
    }
    if layer_kl_manip:
        result["kl_manip_mean"] = float(np.mean(layer_kl_manip))
    if layer_kl_cal2:
        result["kl_cal2_mean"] = float(np.mean(layer_kl_cal2))
    if regions is not None:
        result["region_boundaries"] = regions
    return result


def main():
    print("=== GPT-OSS 120B -- 5-Condition Analysis ===")
    print(f"Routing: {RECONSTRUCTION_NAME}")
    print(f"Conditions: {', '.join(f'{c}={COND_LABELS[c]}' for c in CONDITIONS)}")
    print()

    prompt_texts = load_prompt_texts(TSV)
    with open(PROMPT_SUITE) as f:
        suite = json.load(f)
    cal_paragraph = suite["calibration_paragraph"]
    print(f"Loaded {len(prompt_texts)} prompt texts")
    print(f"Calibration paragraph: {len(cal_paragraph)} chars")
    print()

    prompt_dirs = sorted(
        [d for d in pathlib.Path(OUTPUT_DIR).iterdir()
         if d.is_dir() and (d / "metadata.txt").exists()],
        key=lambda d: d.name,
    )
    print(f"Found {len(prompt_dirs)} prompt output directories")

    # --- Compute metrics per prompt ---
    results = []
    for prompt_dir in prompt_dirs:
        prompt_id = prompt_dir.name
        n_prompt, _ = get_metadata(prompt_dir)

        prompt_text = prompt_texts.get(prompt_id)
        regions = None
        if prompt_text is not None and n_prompt > 0:
            regions = estimate_region_boundaries(prompt_text, cal_paragraph, n_prompt)

        metrics = compute_metrics(prompt_dir, n_prompt, regions=regions)
        if metrics is None:
            print(f"  SKIP {prompt_id}: no valid data")
            continue

        prefix, *rest = prompt_id.split("_", 1)
        category = rest[0] if rest else ""
        pair_num = int(prefix[1:3])
        condition = prefix[3]

        row = {
            "id": prompt_id,
            "condition": condition,
            "pair": pair_num,
            "category": category,
            "n_prompt_tokens": n_prompt,
            **metrics,
        }
        results.append(row)

        kl_str = ""
        if "kl_manip_mean" in metrics:
            kl_str = f" KL={metrics['kl_manip_mean']:.6f}"
        print(f"  {prompt_id}: RE={metrics['prefill_re']:.6f} LT={metrics['last_token_re']:.6f}{kl_str} tok={n_prompt}")

    # --- Group by pair and condition ---
    pairs = {}
    for row in results:
        pairs.setdefault(row["pair"], {})[row["condition"]] = row

    # --- Per-condition summary ---
    print(f"\n=== Per-Condition Summary ===")
    for cond in CONDITIONS:
        cond_rows = [r for r in results if r["condition"] == cond]
        if not cond_rows:
            continue
        re_vals = [r["prefill_re"] for r in cond_rows]
        lt_vals = [r["last_token_re"] for r in cond_rows]
        kl_vals = [r["kl_manip_mean"] for r in cond_rows if "kl_manip_mean" in r]
        print(f"  {cond} ({COND_LABELS[cond]:>5}): n={len(cond_rows)}"
              f"  RE={np.mean(re_vals):.6f}+/-{np.std(re_vals):.6f}"
              f"  LT={np.mean(lt_vals):.6f}+/-{np.std(lt_vals):.6f}"
              + (f"  KL={np.mean(kl_vals):.6f}+/-{np.std(kl_vals):.6f}" if kl_vals else ""))

    # --- All 10 pairwise Wilcoxon comparisons ---
    print(f"\n=== Pairwise Wilcoxon Tests (n={len(pairs)} pairs) ===")
    print(f"{'Pair':>6} {'Metric':<12} {'Mean_Diff':>10} {'X>Y':>5} {'W':>8} {'p_raw':>12} {'p_holm':>12}")
    print("-" * 72)

    # Collect all raw tests first, then apply Holm-Bonferroni
    raw_tests = []  # (label, metric_name, arr, gt, w, p_raw, pw_ref)
    pairwise_results = []
    for c1, c2 in itertools.combinations(CONDITIONS, 2):
        diffs_re, diffs_lt, diffs_kl = [], [], []
        for p in sorted(pairs):
            if c1 in pairs[p] and c2 in pairs[p]:
                a, b = pairs[p][c1], pairs[p][c2]
                diffs_re.append(a["prefill_re"] - b["prefill_re"])
                diffs_lt.append(a["last_token_re"] - b["last_token_re"])
                if "kl_manip_mean" in a and "kl_manip_mean" in b:
                    diffs_kl.append(a["kl_manip_mean"] - b["kl_manip_mean"])

        label = f"{c1}-{c2}"
        pw = {"pair": label, "c1": c1, "c2": c2}

        for metric_name, diffs in [("all-tok RE", diffs_re), ("last-tok RE", diffs_lt), ("KL-manip", diffs_kl)]:
            if len(diffs) < 6:
                continue
            arr = np.array(diffs)
            gt = int(np.sum(arr > 0))
            w, p = wilcoxon(arr)
            raw_tests.append((label, metric_name, arr, gt, w, p, pw))

        pairwise_results.append(pw)

    # Holm-Bonferroni correction across all tests
    n_tests = len(raw_tests)
    sorted_indices = sorted(range(n_tests), key=lambda i: raw_tests[i][5])
    p_holm_values = [0.0] * n_tests
    cummax = 0.0
    for rank, idx in enumerate(sorted_indices):
        adjusted = raw_tests[idx][5] * (n_tests - rank)
        cummax = max(cummax, adjusted)
        p_holm_values[idx] = min(cummax, 1.0)

    for i, (label, metric_name, arr, gt, w, p_raw, pw) in enumerate(raw_tests):
        p_holm = p_holm_values[i]
        print(f"{label:>6} {metric_name:<12} {np.mean(arr):>+10.6f} {gt:>3}/{len(arr)} {w:>8.0f} {p_raw:>12.4e} {p_holm:>12.4e}")
        pw[metric_name] = {"mean_diff": float(np.mean(arr)), "std_diff": float(np.std(arr)),
                           "gt": gt, "n": len(arr), "W": float(w),
                           "p_raw": float(p_raw), "p_holm": float(p_holm)}

    print(f"\n  Holm-Bonferroni correction applied across {n_tests} tests")

    # --- Per-category breakdown ---
    categories = sorted(set(r["category"] for r in results))
    print(f"\n=== Per-Category Last-Token RE by Condition ===")
    for cat in categories:
        print(f"\n  {cat}:")
        for cond in CONDITIONS:
            vals = [r["last_token_re"] for r in results if r["category"] == cat and r["condition"] == cond]
            if vals:
                print(f"    {cond}({COND_LABELS[cond]:>5}): {np.mean(vals):.6f} +/- {np.std(vals):.6f}")

    # --- Token count check ---
    print(f"\n=== Token Count Verification ===")
    mismatches = 0
    for p in sorted(pairs):
        toks = {c: pairs[p][c]["n_prompt_tokens"] for c in pairs[p]}
        unique_toks = set(toks.values())
        if len(unique_toks) > 1:
            mismatches += 1
            print(f"  Pair {p}: {toks}")
    print(f"  Token mismatches: {mismatches}/{len(pairs)} pairs")

    # --- KL cal2 control ---
    cal2_kls = [r.get("kl_cal2_mean") for r in results if r.get("kl_cal2_mean") is not None]
    if cal2_kls:
        print(f"\n=== KL Cal2 Control (should be low) ===")
        print(f"  mean={np.mean(cal2_kls):.6f} +/- {np.std(cal2_kls):.6f}")

    # --- Save results JSON ---
    layers_used = [r["n_layers"] for r in results]
    all_excluded = set()
    for r in results:
        all_excluded.update(r.get("n_layers_excluded", []))

    output = {
        "experiment": "gptoss_5cond_1",
        "model": "GPT-OSS-120B mxfp4",
        "architecture": "gpt-oss",
        "routing_reconstruction": RECONSTRUCTION_NAME,
        "n_experts": N_EXPERTS,
        "n_expert_used": TOP_K,
        "entropy_normalization": f"log2({TOP_K})",
        "entropy_distribution": "sparse_topk4_softmax (model's actual routing)",
        "kl_distribution": f"dense_softmax_full{N_EXPERTS}_normalized (analysis proxy)",
        "kl_baseline": "mean routing distribution over Cal1 tokens per layer",
        "region_boundary_method": "proportional_char_to_token_mapping",
        "n_moe_layers": 36,
        "n_moe_layers_valid_range": [min(layers_used), max(layers_used)] if layers_used else [],
        "excluded_layers_union": sorted(all_excluded),
        "conditions": COND_LABELS,
        "multiple_comparisons_correction": "holm-bonferroni",
        "n_pairwise_tests": n_tests,
        "design": "Cal-Manip-Cal sandwich, 30 paired prompts x 5 conditions, cold cache",
        "inference": {
            "n_predict": 0,
            "ngl": 999,
            "ctx": 4096,
            "flash_attn": "off",
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "sampling": "greedy_argmax",
            "routing_only": True,
        },
        "pairwise_tests": pairwise_results,
        "per_prompt": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n=== DONE. {len(results)} prompts -> {RESULTS_FILE} ===")


if __name__ == "__main__":
    main()
