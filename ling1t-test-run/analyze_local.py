#!/usr/bin/env python3
"""
Ling-1T local post-analysis for the 5-condition system experiment.

Runs entirely on the local machine after downloading output/ from the instance.
Prefers exact captured top-8 routing weights when available; otherwise falls back
to the earlier sigmoid(logits)->top-8 proxy. Runs all 10 pairwise Wilcoxon
comparisons and writes results JSON.

Usage:
    python3 analyze_local.py \
        --output-dir experiments/ling1t-5cond/output/ \
        --prompt-suite experiments/ling1t-5cond/prompt_suite.json

Entropy is normalized by log2(8): only the 8 selected experts contribute.
Absolute Ling RE values remain a within-Ling metric and are NOT comparable to
softmax-routed models.
"""
import argparse
import glob
import json
import pathlib
from itertools import combinations

import numpy as np

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except ImportError:
    scipy_wilcoxon = None

EXCLUDED_LAYERS: set = set()

TOP_K = 8  # Ling-1T routes to top-8 of 256 experts

CONDITIONS = "ABCDE"
COND_LABELS = {
    "A": "this system",
    "B": "a system",
    "C": "your system",
    "D": "the system",
    "E": "their system",
}

WEIGHTS_NORM_GLOB = "ffn_moe_weights_norm-*.npy"
LOGITS_GLOB = "ffn_moe_logits-*.npy"
ROUTER_METADATA_GLOBS = (
    WEIGHTS_NORM_GLOB,
    LOGITS_GLOB,
    "ffn_moe_probs_biased-*.npy",
    "ffn_moe_probs_masked-*.npy",
)
MIN_EXACT_LAYER_COVERAGE = 70


def compute_probs(logits):
    """
    Ling-1T uses SIGMOID routing with top-k mask.

    1. Apply sigmoid to raw logits (independent per-expert gates)
    2. Keep only top-8 experts per token, zero the rest
    3. Normalize the top-8 to sum=1 (probability simplex)
    """
    scores = 1.0 / (1.0 + np.exp(-logits))              # sigmoid
    topk_indices = np.argpartition(scores, -TOP_K, axis=-1)[:, -TOP_K:]
    masked = np.zeros_like(scores)
    rows = np.arange(scores.shape[0])[:, None]
    masked[rows, topk_indices] = scores[rows, topk_indices]
    total = masked.sum(axis=-1, keepdims=True)
    total = np.where(total < 1e-30, 1e-30, total)
    return masked / total


def normalize_rows(weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(weights < 0.0, 0.0, weights)
    total = weights.sum(axis=-1, keepdims=True)
    total = np.where(total < 1e-30, 1e-30, total)
    return weights / total


def compute_entropy_from_probs(probs, max_ent):
    return -np.sum(probs * np.log2(probs + 1e-30), axis=-1) / max_ent


def sorted_router_files(router_dir, pattern):
    return sorted(
        glob.glob(str(router_dir / pattern)),
        key=lambda fp: int(pathlib.Path(fp).stem.split("-")[1]),
    )


def layer_index_from_path(fp):
    return int(pathlib.Path(fp).stem.split("-")[1])


def detect_router_source(router_dir):
    weights_norm_files = sorted_router_files(router_dir, WEIGHTS_NORM_GLOB)
    logits_files = sorted_router_files(router_dir, LOGITS_GLOB)

    if weights_norm_files:
        weight_layers = {layer_index_from_path(fp) for fp in weights_norm_files}
        logits_layers = {layer_index_from_path(fp) for fp in logits_files}

        if logits_layers:
            if weight_layers == logits_layers:
                return "weights_norm", weights_norm_files
            print(
                "    weights_norm capture incomplete/misaligned "
                f"({len(weight_layers)} layers vs {len(logits_layers)} logits layers); "
                "falling back to logits proxy"
            )
        elif len(weight_layers) >= MIN_EXACT_LAYER_COVERAGE:
            return "weights_norm", weights_norm_files
        else:
            print(
                "    weights_norm capture incomplete "
                f"({len(weight_layers)} layers < {MIN_EXACT_LAYER_COVERAGE}); "
                "falling back to logits proxy"
            )

    if logits_files:
        return "logits_proxy", logits_files

    return None, []


def get_metadata(prompt_dir):
    meta = prompt_dir / "metadata.txt"
    info = {}
    if meta.exists():
        for line in meta.read_text().strip().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                info[key] = value
        return int(info.get("n_tokens_prompt", 0)), int(info.get("n_tokens_generated", 0))

    router_dir = prompt_dir / "router"
    if router_dir.exists():
        shapes = []
        for pattern in ROUTER_METADATA_GLOBS:
            for fp in router_dir.glob(pattern):
                try:
                    shapes.append(np.load(str(fp)).shape[0])
                except Exception:
                    continue
            if shapes:
                break
        if shapes:
            n_prompt = int(np.median(shapes))
            print(f"    metadata.txt missing, inferred n_tokens_prompt={n_prompt}")
            return n_prompt, 0
    return 0, 0


def compute_metrics(prompt_dir, n_prompt):
    router_dir = prompt_dir / "router"
    if not router_dir.exists():
        return None

    source_name, files = detect_router_source(router_dir)
    if not files or n_prompt == 0:
        return None

    sample = np.load(files[0])
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    n_experts = 256 if source_name == "weights_norm" else sample.shape[1]
    max_ent = np.log2(TOP_K)

    shapes = {}
    corrupt_layers = set()
    for fp in files:
        layer_index = layer_index_from_path(fp)
        try:
            shapes[layer_index] = np.load(fp).shape[0]
        except (ValueError, Exception):
            corrupt_layers.add(layer_index)
    if corrupt_layers:
        print(f"    Corrupt .npy (skipped): layers {sorted(corrupt_layers)}")
    median_rows = np.median(list(shapes.values()))

    good_layers = sorted([
        li for li in shapes
        if shapes[li] >= median_rows * 0.5
        and li not in EXCLUDED_LAYERS
    ])
    excluded_layers = sorted(set(shapes.keys()) - set(good_layers))
    if excluded_layers:
        print(f"    Auto-excluded layers: {excluded_layers}")

    per_layer = []
    all_ent = []
    last_token_ents = []

    for layer_index in good_layers:
        if source_name == "weights_norm":
            fp = router_dir / f"ffn_moe_weights_norm-{layer_index}.npy"
            weights = np.load(str(fp))
            if weights.ndim == 1:
                weights = weights.reshape(1, -1)
            n_rows = min(weights.shape[0], n_prompt)
            probs = normalize_rows(weights[:n_rows])
        else:
            fp = router_dir / f"ffn_moe_logits-{layer_index}.npy"
            logits = np.load(str(fp))
            if logits.ndim == 1:
                logits = logits.reshape(1, -1)
            n_rows = min(logits.shape[0], n_prompt)
            probs = compute_probs(logits[:n_rows])

        ent = compute_entropy_from_probs(probs, max_ent)
        last_ent = float(ent[n_rows - 1])
        last_token_ents.append(last_ent)

        per_layer.append({
            "layer": layer_index,
            "mean_entropy": float(np.mean(ent)),
            "std_entropy": float(np.std(ent)),
            "last_token_entropy": last_ent,
            "n_rows": int(n_rows),
            "source": source_name,
        })

        valid = ent > 0
        if valid.sum() > 0:
            all_ent.extend(ent[valid].tolist())

    return {
        "prefill_re": float(np.mean(all_ent)) if all_ent else 0.0,
        "last_token_re": float(np.mean(last_token_ents)) if last_token_ents else 0.0,
        "n_layers": len(good_layers),
        "n_layers_excluded": excluded_layers,
        "n_experts": n_experts,
        "routing_source": source_name,
        "per_layer": per_layer,
    }


def main():
    parser = argparse.ArgumentParser(description="Ling-1T 5-condition system analysis")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-suite", required=True)
    parser.add_argument("--results-file", default="results_5cond_system_prefill_ling1t.json")
    args = parser.parse_args()

    output_path = pathlib.Path(args.output_dir)
    if not output_path.exists():
        print(f"ERROR: output dir not found: {output_path}")
        return 1

    with open(args.prompt_suite) as f:
        suite = json.load(f)

    print("=== Ling-1T 5-Condition Self-Referential Analysis ===")
    print(f"Output dir : {output_path}")
    print(f"Conditions : {', '.join(f'{c}={COND_LABELS[c]}' for c in CONDITIONS)}")
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
        try:
            stripped = prefix.lstrip("P")
            pair_num = int(stripped[:2])
            condition = stripped[2]
        except (ValueError, IndexError):
            pair_num = 0
            condition = "?"

        row = {
            "id": prompt_id,
            "condition": condition,
            "pair": pair_num,
            "category": category,
            "n_prompt_tokens": n_prompt,
            **metrics,
        }
        results.append(row)
        print(f"  {prompt_id}: RE={metrics['prefill_re']:.6f} "
              f"last_tok={metrics['last_token_re']:.6f} tokens={n_prompt} "
              f"source={metrics['routing_source']}")

    # === Analysis ===
    print(f"\n=== PHASE 3: 5-Condition Paired Analysis ({len(results)} prompts) ===")
    pairs = {}
    for row in results:
        pairs.setdefault(row["pair"], {})[row["condition"]] = row

    # Token match table
    print(f"\n{'Pair':>4} {'Category':<20} " + " ".join(f"{c+'_tok':>5}" for c in CONDITIONS) + f" {'Match':>7}")
    print("-" * 80)
    for pair_num in sorted(pairs.keys()):
        p = pairs[pair_num]
        if not all(c in p for c in CONDITIONS):
            continue
        toks = [p[c]["n_prompt_tokens"] for c in CONDITIONS]
        status = "OK" if len(set(toks)) == 1 else "MISMATCH"
        cat = p["A"]["category"]
        tok_str = " ".join(f"{t:>5}" for t in toks)
        print(f"  {pair_num:>3}  {cat:<20} {tok_str} {status:>7}")

    # All 10 pairwise comparisons
    comp_results = {}
    for cond1, cond2 in combinations(CONDITIONS, 2):
        label = f"{cond1} vs {cond2}"
        diffs_re = []
        diffs_lt = []

        print(f"\n--- {label} ({COND_LABELS[cond1]} vs {COND_LABELS[cond2]}) ---")
        print(f"  {'Pair':>4} {'Category':<20} {f'{cond1}_RE':>8} {f'{cond2}_RE':>8} {'Diff_RE':>8} "
              f"{f'{cond1}_LT':>8} {f'{cond2}_LT':>8} {'Diff_LT':>8}")
        print("  " + "-" * 90)

        for pair_num in sorted(pairs.keys()):
            if cond1 not in pairs[pair_num] or cond2 not in pairs[pair_num]:
                continue
            r1 = pairs[pair_num][cond1]
            r2 = pairs[pair_num][cond2]
            d_re = r1["prefill_re"] - r2["prefill_re"]
            d_lt = r1["last_token_re"] - r2["last_token_re"]
            diffs_re.append(d_re)
            diffs_lt.append(d_lt)
            print(f"  {pair_num:>4}  {r1['category']:<20} {r1['prefill_re']:>8.6f} {r2['prefill_re']:>8.6f} {d_re:>+8.6f} "
                  f"{r1['last_token_re']:>8.6f} {r2['last_token_re']:>8.6f} {d_lt:>+8.6f}")

        if diffs_lt:
            dre = np.array(diffs_re)
            dlt = np.array(diffs_lt)
            n_pos_re = int(np.sum(dre > 0))
            n_pos_lt = int(np.sum(dlt > 0))
            print(f"\n  Summary (n={len(dlt)} pairs):")
            print(f"    All-token RE:  mean = {np.mean(dre):+.6f} +/- {np.std(dre):.6f}  ({n_pos_re}/{len(dre)} {cond1}>{cond2})")
            print(f"    Last-token RE: mean = {np.mean(dlt):+.6f} +/- {np.std(dlt):.6f}  ({n_pos_lt}/{len(dlt)} {cond1}>{cond2})")
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
            }

    # Condition means summary
    print("\n--- Condition Means ---")
    print(f"  {'Cond':<6} {'Label':<20} {'All-tok RE':>12} {'Last-tok RE':>12} {'N':>4}")
    print("  " + "-" * 60)
    for c in CONDITIONS:
        cond_rows = [r for r in results if r["condition"] == c]
        if cond_rows:
            mean_re = np.mean([r["prefill_re"] for r in cond_rows])
            mean_lt = np.mean([r["last_token_re"] for r in cond_rows])
            print(f"  {c:<6} {COND_LABELS[c]:<20} {mean_re:>12.6f} {mean_lt:>12.6f} {len(cond_rows):>4}")

    # Per-category breakdown
    print("\n--- Per-Category (last-token RE, C vs B = your vs a) ---")
    categories = sorted(set(row["category"] for row in results))
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
            print(f"  {category:<20} n={len(cat_diffs)} mean_diff={np.mean(arr):+.6f} std={np.std(arr):.6f}")

    # Load build metadata if present
    exp_dir = pathlib.Path(args.output_dir).parent
    llama_cpp_commit = None
    binary_md5 = None
    build_commit_file = exp_dir / "build_commit.txt"
    binary_md5_file = exp_dir / "binary_md5.txt"
    if build_commit_file.exists():
        llama_cpp_commit = build_commit_file.read_text().strip()
    if binary_md5_file.exists():
        binary_md5 = binary_md5_file.read_text().strip()

    n_moe_layers = None
    if results:
        n_moe_layers = results[0].get("n_layers")

    output_data = {
        "experiment": "ling1t_5cond_system",
        "model": "Ling-1T Q3_K_S (inclusionAI/Ling-1T)",
        "architecture": "bailingmoe2",
        "n_experts": 256,
        "n_expert_used": 8,
        "n_moe_layers": n_moe_layers,
        "n_moe_layers_excluded": sorted(EXCLUDED_LAYERS),
        "gating_function": "sigmoid",
        "top_k": TOP_K,
        "routing_metric_priority": [
            "ffn_moe_weights_norm (exact final normalized top-8 weights)",
            "ffn_moe_logits (fallback sigmoid->top-8 proxy)",
        ],
        "entropy_normalization": "selected_topk_shannon_div_log2_k",
        "note": "Prefers exact captured ffn_moe_weights_norm tensors when present; otherwise falls back to sigmoid(logits)->top-8 proxy. Entropy normalized by log2(8). NOT comparable to softmax-routed models.",
        "chat_template": (
            "<role>SYSTEM</role>detailed thinking off<|role_end|>"
            "<role>HUMAN</role>{prompt}<|role_end|>"
            "<role>ASSISTANT</role>"
        ),
        "design": "Cal-Manip-Cal sandwich, 30 paired prompts x 5 conditions (system), cold cache",
        "conditions": COND_LABELS,
        "rationale": "Self-referential determiner manipulation on system-language prompts for within-model routing comparisons",
        "llama_cpp_branch": "master (BailingMoeV2 merged Oct 2025)",
        "llama_cpp_commit": llama_cpp_commit,
        "binary_md5": binary_md5,
        "inference": {
            "n_predict": 0,
            "ngl": 999,
            "ctx": 4096,
            "sampling": "greedy_argmax",
            "routing_only": True,
        },
        "comparisons": comp_results,
        "npy_preserved": True,
        "per_prompt": results,
    }

    results_file = pathlib.Path(args.output_dir).parent / args.results_file
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n=== DONE. {len(results)} prompts analyzed. Results -> {results_file} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
