#!/usr/bin/env python3
"""
DeepSeek V3.1 local/on-instance post-analysis for the 5-condition selfref run.

Loads captured `ffn_moe_logits`, reconstructs the recoverable DeepSeek routing
distribution, runs all 10 pairwise comparisons, and writes a single JSON file.
"""
import argparse
import glob
import json
import os
import pathlib
from itertools import combinations

import numpy as np

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except ImportError:
    scipy_wilcoxon = None

from deepseek_router import (
    N_ROUTED_EXPERTS,
    RECONSTRUCTION_NAME,
    TOP_K,
    normalized_entropy,
    reconstruct_probs,
)
from analysis_common import inspect_router_layers, read_metadata

MANUAL_EXCLUDED_LAYERS = set()

CONDITIONS = "ABCDE"
COND_LABELS = {
    "A": "this system",
    "B": "a system",
    "C": "your system",
    "D": "the system",
    "E": "their system",
}


def get_metadata(prompt_dir):
    info = read_metadata(prompt_dir)
    if info:
        return int(info.get("n_tokens_prompt", 0)), int(info.get("n_tokens_generated", 0))

    router_dir = prompt_dir / "router"
    if router_dir.exists():
        shapes = []
        for fp in router_dir.glob("ffn_moe_logits-*.npy"):
            try:
                shapes.append(np.load(str(fp)).shape[0])
            except Exception:
                continue
        if shapes:
            n_prompt = int(np.median(shapes))
            print(f"    metadata.txt missing, inferred n_tokens_prompt={n_prompt}")
            return n_prompt, 0
    return 0, 0


def compute_metrics(prompt_dir, n_prompt):
    router_dir = prompt_dir / "router"
    if not router_dir.exists():
        return None

    files = sorted(
        glob.glob(str(router_dir / "ffn_moe_logits-*.npy")),
        key=lambda fp: int(pathlib.Path(fp).stem.split("-")[1]),
    )
    if not files or n_prompt == 0:
        return None

    validation = inspect_router_layers(
        prompt_dir,
        expected_rows=n_prompt,
        manual_excluded=MANUAL_EXCLUDED_LAYERS,
    )
    if validation.n_experts != N_ROUTED_EXPERTS:
        raise ValueError(f"Expected {N_ROUTED_EXPERTS} experts, got {validation.n_experts} in {prompt_dir}")
    if validation.corrupt_layers:
        print(f"    Corrupt or unexpected .npy (skipped): layers {validation.corrupt_layers}")
    if validation.row_mismatch_layers:
        print(f"    Row-count mismatches (skipped): layers {validation.row_mismatch_layers}")
    if validation.excluded_layers:
        print(f"    Excluded layers: {validation.excluded_layers}")

    per_layer = []
    all_ent = []
    last_token_ents = []

    for layer_index in validation.good_layers:
        fp = router_dir / f"ffn_moe_logits-{layer_index}.npy"
        logits = np.load(str(fp))
        probs = reconstruct_probs(logits)
        ent = normalized_entropy(probs)
        last_ent = float(ent[-1])
        last_token_ents.append(last_ent)

        per_layer.append(
            {
                "layer": layer_index,
                "mean_entropy": float(np.mean(ent)),
                "std_entropy": float(np.std(ent)),
                "last_token_entropy": last_ent,
                "n_rows": validation.layer_rows[layer_index],
            }
        )
        all_ent.extend(ent.tolist())

    return {
        "prefill_re": float(np.mean(all_ent)) if all_ent else 0.0,
        "last_token_re": float(np.mean(last_token_ents)) if last_token_ents else 0.0,
        "n_layers": len(validation.good_layers),
        "n_layers_excluded": validation.excluded_layers,
        "n_experts": validation.n_experts,
        "per_layer": per_layer,
    }


def load_build_metadata(exp_dir):
    llama_cpp_commit = None
    binary_md5 = None
    build_commit_file = exp_dir / "build_commit.txt"
    binary_md5_file = exp_dir / "binary_md5.txt"
    if build_commit_file.exists():
        llama_cpp_commit = build_commit_file.read_text().strip()
    if binary_md5_file.exists():
        binary_md5 = binary_md5_file.read_text().strip()
    return llama_cpp_commit, binary_md5


def main():
    parser = argparse.ArgumentParser(description="DeepSeek V3.1 5-condition self-reference analysis")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-suite", required=True)
    parser.add_argument("--results-file", default="results_5cond_selfref_prefill_ds31.json")
    args = parser.parse_args()

    output_path = pathlib.Path(args.output_dir)
    if not output_path.exists():
        print(f"ERROR: output dir not found: {output_path}")
        return 1

    with open(args.prompt_suite) as f:
        suite = json.load(f)

    print("=== DeepSeek V3.1 5-Condition Self-Referential Analysis ===")
    print(f"Output dir : {output_path}")
    print(f"Conditions : {', '.join(f'{c}={COND_LABELS[c]}' for c in CONDITIONS)}")
    print(f"Routing    : {RECONSTRUCTION_NAME}")
    print()

    prompt_dirs = sorted(
        [d for d in output_path.iterdir() if d.is_dir() and (d / "router").exists()],
        key=lambda d: d.name,
    )
    print(f"Found {len(prompt_dirs)} captured prompt directories")
    print()

    results = []
    for prompt_dir in prompt_dirs:
        prompt_id = prompt_dir.name
        n_prompt, _ = get_metadata(prompt_dir)
        metrics = compute_metrics(prompt_dir, n_prompt)
        if metrics is None:
            print(f"  SKIP {prompt_id}: no valid data")
            continue

        prefix = prompt_id.split("_")[0] if "_" in prompt_id else prompt_id
        rest_parts = prompt_id.split("_", 1)
        category = rest_parts[1] if len(rest_parts) > 1 else ""
        stripped = prefix.lstrip("P")
        pair_num = int(stripped[:2])
        condition = stripped[2]

        row = {
            "id": prompt_id,
            "condition": condition,
            "pair": pair_num,
            "category": category,
            "n_prompt_tokens": n_prompt,
            **metrics,
        }
        results.append(row)
        print(
            f"  {prompt_id}: RE={metrics['prefill_re']:.6f} "
            f"last_tok={metrics['last_token_re']:.6f} tokens={n_prompt}"
        )

    print(f"\n=== PHASE 3: 5-Condition Paired Analysis ({len(results)} prompts) ===")
    pairs = {}
    for row in results:
        pairs.setdefault(row["pair"], {})[row["condition"]] = row

    match_counts = {"matched_groups": 0, "total_groups": 0}
    print(f"\n{'Pair':>4} {'Category':<20} " + " ".join(f"{c+'_tok':>5}" for c in CONDITIONS) + f" {'Match':>8}")
    print("-" * 82)
    for pair_num in sorted(pairs.keys()):
        group = pairs[pair_num]
        if not all(c in group for c in CONDITIONS):
            continue
        toks = [group[c]["n_prompt_tokens"] for c in CONDITIONS]
        status = "OK" if len(set(toks)) == 1 else "MISMATCH"
        match_counts["total_groups"] += 1
        if status == "OK":
            match_counts["matched_groups"] += 1
        category = group["A"]["category"]
        tok_str = " ".join(f"{tok:>5}" for tok in toks)
        print(f"  {pair_num:>3}  {category:<20} {tok_str} {status:>8}")

    comp_results = {}
    for cond1, cond2 in combinations(CONDITIONS, 2):
        label = f"{cond1} vs {cond2}"
        diffs_re = []
        diffs_lt = []

        print(f"\n--- {label} ({COND_LABELS[cond1]} vs {COND_LABELS[cond2]}) ---")
        print(
            f"  {'Pair':>4} {'Category':<20} {f'{cond1}_RE':>8} {f'{cond2}_RE':>8} {'Diff_RE':>8} "
            f"{f'{cond1}_LT':>8} {f'{cond2}_LT':>8} {'Diff_LT':>8}"
        )
        print("  " + "-" * 90)

        for pair_num in sorted(pairs.keys()):
            if cond1 not in pairs[pair_num] or cond2 not in pairs[pair_num]:
                continue
            row1 = pairs[pair_num][cond1]
            row2 = pairs[pair_num][cond2]
            diff_re = row1["prefill_re"] - row2["prefill_re"]
            diff_lt = row1["last_token_re"] - row2["last_token_re"]
            diffs_re.append(diff_re)
            diffs_lt.append(diff_lt)
            print(
                f"  {pair_num:>4}  {row1['category']:<20} "
                f"{row1['prefill_re']:>8.6f} {row2['prefill_re']:>8.6f} {diff_re:>+8.6f} "
                f"{row1['last_token_re']:>8.6f} {row2['last_token_re']:>8.6f} {diff_lt:>+8.6f}"
            )

        if diffs_lt:
            dre = np.array(diffs_re)
            dlt = np.array(diffs_lt)
            n_pos_re = int(np.sum(dre > 0))
            n_pos_lt = int(np.sum(dlt > 0))
            p_re = None
            p_lt = None
            w_re = None
            w_lt = None
            print(f"\n  Summary (n={len(dlt)} pairs):")
            print(
                f"    All-token RE:  mean = {np.mean(dre):+.6f} +/- {np.std(dre):.6f}  "
                f"({n_pos_re}/{len(dre)} {cond1}>{cond2})"
            )
            print(
                f"    Last-token RE: mean = {np.mean(dlt):+.6f} +/- {np.std(dlt):.6f}  "
                f"({n_pos_lt}/{len(dlt)} {cond1}>{cond2})"
            )
            if len(dlt) >= 6 and scipy_wilcoxon is not None:
                w_re, p_re = scipy_wilcoxon(dre)
                w_lt, p_lt = scipy_wilcoxon(dlt)
                print(f"    Wilcoxon all-tok:  W={w_re:.0f}, p={p_re:.4e}")
                print(f"    Wilcoxon last-tok: W={w_lt:.0f}, p={p_lt:.4e}")
            elif len(dlt) >= 6:
                print("    Wilcoxon skipped: scipy not installed")

            comp_results[label] = {
                "mean_diff_re": float(np.mean(dre)),
                "std_diff_re": float(np.std(dre)),
                "mean_diff_lt": float(np.mean(dlt)),
                "std_diff_lt": float(np.std(dlt)),
                "n_positive_re": n_pos_re,
                "n_positive_lt": n_pos_lt,
                "n_pairs": len(dlt),
                "wilcoxon_w_re": float(w_re) if w_re is not None else None,
                "wilcoxon_p_re": float(p_re) if p_re is not None else None,
                "wilcoxon_w_lt": float(w_lt) if w_lt is not None else None,
                "wilcoxon_p_lt": float(p_lt) if p_lt is not None else None,
            }

    print("\n--- Condition Means ---")
    print(f"  {'Cond':<6} {'Label':<20} {'All-tok RE':>12} {'Last-tok RE':>12} {'N':>4}")
    print("  " + "-" * 60)
    condition_means = {}
    for condition in CONDITIONS:
        cond_rows = [row for row in results if row["condition"] == condition]
        if not cond_rows:
            continue
        mean_re = np.mean([row["prefill_re"] for row in cond_rows])
        mean_lt = np.mean([row["last_token_re"] for row in cond_rows])
        condition_means[condition] = {
            "label": COND_LABELS[condition],
            "mean_re": float(mean_re),
            "mean_lt": float(mean_lt),
            "n": len(cond_rows),
        }
        print(f"  {condition:<6} {COND_LABELS[condition]:<20} {mean_re:>12.6f} {mean_lt:>12.6f} {len(cond_rows):>4}")

    print("\n--- Per-Category (last-token RE, C vs B = your vs a) ---")
    categories = sorted(set(row["category"] for row in results))
    per_category_cb = {}
    for category in categories:
        cat_diffs = []
        for pair_num in sorted(pairs.keys()):
            if "C" not in pairs[pair_num] or "B" not in pairs[pair_num]:
                continue
            if pairs[pair_num]["C"]["category"] != category:
                continue
            cat_diffs.append(pairs[pair_num]["C"]["last_token_re"] - pairs[pair_num]["B"]["last_token_re"])
        if cat_diffs:
            arr = np.array(cat_diffs)
            per_category_cb[category] = {
                "n": len(cat_diffs),
                "mean_diff_lt_c_minus_b": float(np.mean(arr)),
                "std_diff_lt_c_minus_b": float(np.std(arr)),
            }
            print(f"  {category:<20} n={len(cat_diffs)} mean_diff={np.mean(arr):+.6f} std={np.std(arr):.6f}")

    exp_dir = pathlib.Path(args.output_dir).parent
    llama_cpp_commit, binary_md5 = load_build_metadata(exp_dir)
    n_moe_layers = results[0].get("n_layers") if results else None
    excluded_layers = sorted({layer for row in results for layer in row["n_layers_excluded"]})

    output_data = {
        "experiment": "ds31_5cond_system_1",
        "model": "DeepSeek V3.1 UD-Q2_K_XL",
        "architecture": "deepseek_v3",
        "n_experts": N_ROUTED_EXPERTS,
        "n_expert_used": TOP_K,
        "n_moe_layers": n_moe_layers,
        "n_moe_layers_excluded": excluded_layers,
        "gating_function": "sigmoid",
        "routing_reconstruction": RECONSTRUCTION_NAME,
        "entropy_normalization": "selected_topk_entropy_div_log2_8",
        "note": "Bias-free DeepSeek routing reconstruction from captured ffn_moe_logits.",
        "chat_template": "<｜User｜>{prompt}<｜Assistant｜>",
        "design": "Cal-Manip-Cal sandwich, 30 paired prompts x 5 conditions (system), cold cache",
        "conditions": COND_LABELS,
        "llama_cpp_branch": "ggml-org/llama.cpp b8123",
        "llama_cpp_commit": llama_cpp_commit,
        "binary_md5": binary_md5,
        "inference": {
            "n_predict": 0,
            "ngl": int(os.environ.get("NGL", "999")),
            "ctx": int(os.environ.get("CTX", "4096")),
            "sampling": "greedy_argmax",
            "routing_only": True,
        },
        "token_matching": match_counts,
        "condition_means": condition_means,
        "comparisons": comp_results,
        "per_category_c_vs_b": per_category_cb,
        "npy_preserved": True,
        "per_prompt": results,
        "prompt_suite_model_label": suite.get("model"),
    }

    results_file = pathlib.Path(args.output_dir).parent / args.results_file
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n=== DONE. {len(results)} prompts analyzed. Results -> {results_file} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
