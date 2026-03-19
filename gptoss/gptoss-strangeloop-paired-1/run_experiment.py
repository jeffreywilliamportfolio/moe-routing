#!/usr/bin/env python3
"""
GPT-OSS 120B -- Strange Loop Paired Experiment (Prefill-Only).

Routing reconstruction: top-4 by raw logit, then softmax on selected 4.

Metrics computed per prompt:
  - RE (routing entropy): sparse top-4 reconstruction, normalized by log2(4)
  - KL-to-baseline: dense 128-dim softmax proxy, KL(manip_token || cal1_mean)
    Region boundaries estimated by proportional char->token mapping.

60 prompts: 30 A ("this paradox/loop/structure") + 30 B ("a paradox/loop/structure").
Cal-Manip-Cal sandwich structure. Preserve router tensors for downstream analysis.

Control experiment: recursive content (Godel, Escher, bootstrap) but NOT about the model.

Architecture: 128 experts, top-4 routing, 36 MoE layers (layer 35 excluded).
"""
import glob
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

from gptoss_router import (
    N_EXPERTS,
    TOP_K,
    RECONSTRUCTION_NAME,
    reconstruct_probs,
    normalized_entropy,
    softmax_full_probs,
    kl_divergence,
)

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except ImportError:
    scipy_wilcoxon = None

MODEL = os.environ.get(
    "MODEL_PATH",
    "/workspace/models/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf",
)
BINARY = os.environ.get(
    "CAPTURE_BINARY",
    "/workspace/consciousness-experiment/capture_activations",
)
LLAMA_BUILD_BIN = os.environ.get(
    "LLAMA_BUILD_BIN",
    "/workspace/src/llama.cpp-b8123/build-cuda/bin",
)
TSV = "prompts_strangeloop_paired.tsv"
PROMPT_SUITE = "prompt_suite.json"
OUTPUT_DIR = "output"
RESULTS_FILE = "results_strangeloop_paired_prefill_gptoss.json"
CORRECTIONS_FILE = "token_corrections.json"

N_PREDICT = 0
NGL = int(os.environ.get("NGL", "999"))
CTX = int(os.environ.get("CTX", "4096"))
THREADS = int(os.environ.get("THREADS", "16"))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "off")
CACHE_TYPE_K = os.environ.get("CACHE_TYPE_K", "f16")
CACHE_TYPE_V = os.environ.get("CACHE_TYPE_V", "f16")
BATCH_SIZE = 15

# GPT-OSS-120B: 128 experts, layer 35 excluded (3-row truncation bug)
EXCLUDED_LAYERS = {35}


def run_capture(tsv_file=TSV):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_BUILD_BIN + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        BINARY,
        "-m", MODEL,
        "--prompt-file", tsv_file,
        "-o", OUTPUT_DIR,
        "-n", str(N_PREDICT),
        "-ngl", str(NGL),
        "-c", str(CTX),
        "-t", str(THREADS),
        "-fa", FLASH_ATTN,
        "--cache-type-k", CACHE_TYPE_K,
        "--cache-type-v", CACHE_TYPE_V,
        "--routing-only",
        "--no-stream",
    ]
    print("Running:", " ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, env=env, check=True)


def prepare_tsv():
    corrections_path = pathlib.Path(CORRECTIONS_FILE)
    if not corrections_path.exists():
        print(f"{CORRECTIONS_FILE} missing; deriving exact token corrections.")
        sys.stdout.flush()
        subprocess.run([sys.executable, "derive_token_corrections.py"], check=True)

    cmd = [sys.executable, "generate_tsv.py", "--corrections", CORRECTIONS_FILE]
    print("Preparing TSV:", " ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, check=True)


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
    """Load prompt_id -> full_text from TSV."""
    texts = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1]
    return texts


def estimate_region_boundaries(prompt_text, cal_paragraph, n_tokens):
    """Estimate Cal1/Manip/Cal2 token boundaries via proportional char->token mapping.

    The Cal-Manip-Cal sandwich has identical calibration text at positions that
    can be found by string matching. Character positions are mapped to token
    positions proportionally. Error is typically <=5 tokens at each boundary,
    which is negligible for averaging over ~100+ token regions. Both A and B
    prompts in a pair have the same boundary error, so it cancels in paired
    comparison.
    """
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
        key=lambda file_path: int(pathlib.Path(file_path).stem.split("-")[1]),
    )
    if not files or n_prompt == 0:
        return None

    n_experts = np.load(files[0]).shape[1]

    shapes = {}
    for file_path in files:
        layer_index = int(pathlib.Path(file_path).stem.split("-")[1])
        shapes[layer_index] = np.load(file_path).shape[0]
    median_rows = np.median(list(shapes.values()))

    good_layers = sorted([
        layer_index for layer_index in shapes
        if shapes[layer_index] >= median_rows * 0.5
        and layer_index not in EXCLUDED_LAYERS
    ])
    excluded_layers = sorted(set(shapes.keys()) - set(good_layers))

    # Unpack region boundaries for KL computation
    has_regions = regions is not None
    if has_regions:
        cal1_s, cal1_e = regions["cal1"]
        manip_s, manip_e = regions["manip"]
        cal2_s, cal2_e = regions["cal2"]

    per_layer = []
    all_ent = []
    last_token_ents = []
    layer_kl_manip = []
    layer_kl_manip_last = []
    layer_kl_cal2 = []

    for layer_index in good_layers:
        file_path = router_dir / f"ffn_moe_logits-{layer_index}.npy"
        logits = np.load(str(file_path))
        n_rows = min(logits.shape[0], n_prompt)

        # --- Entropy (sparse top-4 reconstruction) ---
        probs = reconstruct_probs(logits[:n_rows])
        ent = normalized_entropy(probs)

        last_ent = float(ent[n_rows - 1])
        last_token_ents.append(last_ent)

        layer_info = {
            "layer": layer_index,
            "mean_entropy": float(np.mean(ent)),
            "std_entropy": float(np.std(ent)),
            "last_token_entropy": last_ent,
            "n_rows": int(logits.shape[0]),
        }

        valid = ent > 0
        if valid.sum() > 0:
            all_ent.extend(ent[valid].tolist())

        # --- KL-to-baseline (dense 128-dim softmax proxy) ---
        if has_regions and cal1_e > cal1_s and manip_e > manip_s:
            # Bound region slices to the rows actually present in this layer.
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
            layer_kl_manip_last.append(float(kl_manip[-1]))

            layer_info["kl_manip_mean"] = float(np.mean(kl_manip))
            layer_info["kl_manip_last"] = float(kl_manip[-1])

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
        result["kl_manip_last"] = float(np.mean(layer_kl_manip_last))
    if layer_kl_cal2:
        result["kl_cal2_mean"] = float(np.mean(layer_kl_cal2))

    if regions is not None:
        result["region_boundaries"] = regions

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prepare_tsv()

    print("=== GPT-OSS 120B -- Strange Loop Paired Experiment ===")
    print(f"Routing reconstruction: {RECONSTRUCTION_NAME}")
    print(f"  top_k={TOP_K}, entropy_norm=log2({TOP_K})")
    print(f"  KL distribution: dense softmax({N_EXPERTS}-dim) analysis proxy")
    print(f"n_predict={N_PREDICT}, ctx={CTX}, ngl={NGL}")
    print(f"flash_attn={FLASH_ATTN}, cache_type_k={CACHE_TYPE_K}, cache_type_v={CACHE_TYPE_V}")
    print("60 prompts: 30 deictic-this (A) + 30 generic-a (B)")
    print("Cal-Manip-Cal sandwich, cold KV cache, Harmony template")
    print("Content: Godel, Escher, bootstrap paradoxes, quines, tangled hierarchies")
    print("Control for selfref-paired (recursive but NOT about the model)")
    print()

    # Load prompt texts and calibration paragraph for KL region boundaries
    prompt_texts = load_prompt_texts(TSV)
    with open(PROMPT_SUITE) as f:
        suite = json.load(f)
    cal_paragraph = suite["calibration_paragraph"]
    print(f"Loaded {len(prompt_texts)} prompt texts from {TSV}")
    print(f"Calibration paragraph: {len(cal_paragraph)} chars")
    print(f"Capture batching: {BATCH_SIZE} prompts per batch (matches gptoss-5cond-1)")
    print()

    print("=== PHASE 1: Capture ===")
    with open(TSV) as f:
        all_lines = f.readlines()
    n_prompts = len(all_lines)
    n_batches = (n_prompts + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Loaded {n_prompts} prompts, {n_batches} batches of {BATCH_SIZE}")
    print()

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_prompts)
        batch_lines = all_lines[start:end]

        batch_tsv = f"batch_{batch_idx}.tsv"
        with open(batch_tsv, "w") as f:
            f.writelines(batch_lines)

        print(f"=== BATCH {batch_idx+1}/{n_batches}: prompts {start+1}-{end} ===")
        sys.stdout.flush()
        run_capture(tsv_file=batch_tsv)
        os.remove(batch_tsv)

    print("\n=== PHASE 2: Compute metrics ===")
    prompt_dirs = sorted(
        [d for d in pathlib.Path(OUTPUT_DIR).iterdir() if d.is_dir() and (d / "metadata.txt").exists()],
        key=lambda d: d.name,
    )

    if len(prompt_dirs) != 60:
        print(f"  WARNING: expected 60 prompt directories, found {len(prompt_dirs)}")

    results = []
    layers_used_counts = []
    for prompt_dir in prompt_dirs:
        prompt_id = prompt_dir.name
        n_prompt, _n_gen = get_metadata(prompt_dir)

        # Estimate region boundaries for KL
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

        layers_used_counts.append(metrics["n_layers"])

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
            kl_str = f" KL_manip={metrics['kl_manip_mean']:.6f}"
            if "kl_cal2_mean" in metrics:
                kl_str += f" KL_cal2={metrics['kl_cal2_mean']:.6f}"

        print(f"  {prompt_id}: RE={metrics['prefill_re']:.6f} last_tok={metrics['last_token_re']:.6f}{kl_str} tokens={n_prompt}")

    print(f"\n=== PHASE 3: Paired Analysis ===")
    pairs = {}
    for row in results:
        pairs.setdefault(row["pair"], {})[row["condition"]] = row

    print(
        f"\n{'Pair':>4} {'Category':<20} {'A_tok':>5} {'B_tok':>5} {'A_RE':>8} {'B_RE':>8} {'A-B_RE':>8} "
        f"{'A_LT':>8} {'B_LT':>8} {'A-B_LT':>8} {'A-B_KL':>8}"
    )
    print("-" * 115)

    diffs_re = []
    diffs_lt = []
    diffs_kl = []
    token_mismatches = 0
    token_mismatch_pairs = []
    for pair_num in sorted(pairs.keys()):
        if "A" not in pairs[pair_num] or "B" not in pairs[pair_num]:
            continue
        row_a = pairs[pair_num]["A"]
        row_b = pairs[pair_num]["B"]
        diff_re = row_a["prefill_re"] - row_b["prefill_re"]
        diff_lt = row_a["last_token_re"] - row_b["last_token_re"]
        diffs_re.append(diff_re)
        diffs_lt.append(diff_lt)

        diff_kl = None
        if "kl_manip_mean" in row_a and "kl_manip_mean" in row_b:
            diff_kl = row_a["kl_manip_mean"] - row_b["kl_manip_mean"]
            diffs_kl.append(diff_kl)

        is_token_match = row_a["n_prompt_tokens"] == row_b["n_prompt_tokens"]
        token_status = "OK" if is_token_match else "MISMATCH"
        if not is_token_match:
            token_mismatches += 1
            token_mismatch_pairs.append(pair_num)
        kl_col = f"{diff_kl:>+8.6f}" if diff_kl is not None else f"{'n/a':>8}"
        print(
            f"  {pair_num:>3}  {row_a['category']:<20} {row_a['n_prompt_tokens']:>5} {row_b['n_prompt_tokens']:>5} "
            f"{row_a['prefill_re']:>8.6f} {row_b['prefill_re']:>8.6f} {diff_re:>+8.6f} "
            f"{row_a['last_token_re']:>8.6f} {row_b['last_token_re']:>8.6f} {diff_lt:>+8.6f} "
            f"{kl_col} {token_status}"
        )

    if diffs_lt:
        diffs_re_arr = np.array(diffs_re)
        diffs_lt_arr = np.array(diffs_lt)
        print(f"\n--- Paired Summary (n={len(diffs_lt)} pairs) ---")
        print(f"  Token mismatches: {token_mismatches}/{len(diffs_lt)} pairs")
        if token_mismatch_pairs:
            print(f"  Mismatch pair ids: {token_mismatch_pairs}")
        print(f"  All-token RE:  A-B mean = {np.mean(diffs_re_arr):+.6f} +/- {np.std(diffs_re_arr):.6f}")
        print(f"  Last-token RE: A-B mean = {np.mean(diffs_lt_arr):+.6f} +/- {np.std(diffs_lt_arr):.6f}")

        if diffs_kl:
            diffs_kl_arr = np.array(diffs_kl)
            a_gt_b_kl = int(np.sum(diffs_kl_arr > 0))
            print(f"  KL-to-baseline (manip region): A-B mean = {np.mean(diffs_kl_arr):+.6f} +/- {np.std(diffs_kl_arr):.6f}")
            print(f"    A>B: {a_gt_b_kl}/{len(diffs_kl)}")

        if len(diffs_lt) >= 6 and scipy_wilcoxon is not None:
            w_re, p_re = scipy_wilcoxon(diffs_re_arr)
            w_lt, p_lt = scipy_wilcoxon(diffs_lt_arr)
            raw_ps = [("all-tok RE", w_re, p_re), ("last-tok RE", w_lt, p_lt)]
            if diffs_kl and len(diffs_kl) >= 6:
                w_kl, p_kl = scipy_wilcoxon(diffs_kl_arr)
                raw_ps.append(("KL-manip", w_kl, p_kl))

            # Holm-Bonferroni correction across all tested endpoints
            n_tests = len(raw_ps)
            sorted_ps = sorted(raw_ps, key=lambda x: x[2])
            holm_results = []
            for rank, (name, w, p) in enumerate(sorted_ps):
                holm_threshold = 0.05 / (n_tests - rank)
                p_adj = min(p * (n_tests - rank), 1.0)
                sig = "SIG" if p <= holm_threshold else "ns"
                holm_results.append((name, w, p, p_adj, sig))

            print(f"\n  Wilcoxon signed-rank tests (Holm-corrected, {n_tests} tests):")
            for name, w, p, p_adj, sig in holm_results:
                print(f"    {name:<14} W={w:.0f}, p_raw={p:.4e}, p_holm={p_adj:.4e} [{sig}]")
        elif len(diffs_lt) >= 6:
            print("  Wilcoxon skipped: scipy not installed on this host")

        # KL cal2 control: should be near zero (same text as cal1)
        cal2_kls = [r.get("kl_cal2_mean") for r in results if r.get("kl_cal2_mean") is not None]
        if cal2_kls:
            print(f"\n  KL cal2 control (should be low): mean={np.mean(cal2_kls):.6f} +/- {np.std(cal2_kls):.6f}")

        print("\n--- Per-Category (last-token RE) ---")
        categories = sorted(set(row["category"] for row in results))
        for category in categories:
            cat_diffs = []
            for pair_num in sorted(pairs.keys()):
                if "A" not in pairs[pair_num] or "B" not in pairs[pair_num]:
                    continue
                if pairs[pair_num]["A"]["category"] != category:
                    continue
                cat_diffs.append(pairs[pair_num]["A"]["last_token_re"] - pairs[pair_num]["B"]["last_token_re"])
            if cat_diffs:
                arr = np.array(cat_diffs)
                print(f"  {category:<20} n={len(cat_diffs)} mean_diff={np.mean(arr):+.6f} std={np.std(arr):.6f}")

    # Compute dynamic layer counts from actual data
    all_excluded = set()
    for row in results:
        all_excluded.update(row.get("n_layers_excluded", []))
    n_layers_valid_min = min(layers_used_counts) if layers_used_counts else 0
    n_layers_valid_max = max(layers_used_counts) if layers_used_counts else 0

    output = {
        "experiment": "gptoss_strangeloop_paired_1",
        "model": "GPT-OSS-120B mxfp4",
        "architecture": "gpt-oss",
        "routing_reconstruction": RECONSTRUCTION_NAME,
        "n_experts": N_EXPERTS,
        "n_expert_used": TOP_K,
        "entropy_normalization": f"log2({TOP_K})",
        "entropy_distribution": "sparse_topk4_softmax (model's actual routing)",
        "kl_distribution": f"dense_softmax_full{N_EXPERTS}_normalized (analysis proxy, not actual routing)",
        "kl_baseline": "mean routing distribution over Cal1 tokens per layer",
        "region_boundary_method": "proportional_char_to_token_mapping",
        "n_moe_layers": 36,
        "n_moe_layers_valid_range": [n_layers_valid_min, n_layers_valid_max],
        "excluded_layers_union": sorted(all_excluded),
        "chat_template": "<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>",
        "design": "Cal-Manip-Cal sandwich, 30 paired prompts, cold cache",
        "rationale": "Control for selfref-paired. Recursive content (Godel, Escher, bootstrap) but not about the model.",
        "reconstruction_note": "topk(4) by raw gate logit, then softmax over the selected 4 experts",
        "token_match_claimed": True,
        "token_mismatch_pairs": token_mismatch_pairs,
        "inference": {
            "n_predict": N_PREDICT,
            "ngl": NGL,
            "ctx": CTX,
            "flash_attn": FLASH_ATTN,
            "cache_type_k": CACHE_TYPE_K,
            "cache_type_v": CACHE_TYPE_V,
            "sampling": "greedy_argmax",
            "routing_only": True,
        },
        "npy_preserved": True,
        "per_prompt": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n=== DONE. {len(results)} prompts. Results -> {RESULTS_FILE} ===")


if __name__ == "__main__":
    main()
