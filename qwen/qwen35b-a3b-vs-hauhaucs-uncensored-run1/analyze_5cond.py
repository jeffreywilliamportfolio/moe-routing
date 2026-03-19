#!/usr/bin/env python3
"""Analysis helpers for the Qwen3.5-35B-A3B comparison suite."""
from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import pathlib
from typing import Dict, List, Optional

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

from qwen_router import (
    ENTROPY_MAX,
    N_EXPERTS,
    RECONSTRUCTION_NAME,
    TOP_K,
    kl_divergence,
    normalized_entropy,
    reconstruct_probs,
    softmax_full_probs,
)

_HERE = pathlib.Path(__file__).parent
PROMPT_SUITE = str(_HERE / "prompt-suite.json")
TSV = str(_HERE / "prompts_qwen35b_5cond.tsv")
OUTPUT_DIR = str(_HERE / "output")
MODEL_TAG = os.environ.get("MODEL_TAG", "qwen35b_a3b_base")
RESULTS_FILE = str(_HERE / f"results_{MODEL_TAG}_prefill.json")
MANIFEST_FILE = str(_HERE / "reproducibility_manifest.json")

CONDITIONS = list("ABCDE")
COND_LABELS = {
    "A": "this",
    "B": "a",
    "C": "your",
    "D": "the",
    "E": "their",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_metadata(prompt_dir: pathlib.Path):
    meta = prompt_dir / "metadata.txt"
    info = {}
    if meta.exists():
        for line in meta.read_text().strip().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                info[key] = value
    return int(info.get("n_tokens_prompt", 0)), int(info.get("n_tokens_generated", 0))


def load_prompt_texts(tsv_path: str) -> Dict[str, str]:
    texts = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1]
    return texts


def estimate_region_boundaries(prompt_text: str, cal_paragraph: str, n_tokens: int):
    cal1_start_char = prompt_text.find(cal_paragraph)
    if cal1_start_char < 0:
        return None
    cal1_end_char = cal1_start_char + len(cal_paragraph)
    cal2_start_char = prompt_text.find(cal_paragraph, cal1_end_char)
    if cal2_start_char < 0:
        return None
    cal2_end_char = cal2_start_char + len(cal_paragraph)
    total_chars = len(prompt_text)
    if total_chars == 0:
        return None

    def char_to_tok(char_pos):
        return max(0, min(n_tokens, int(round(char_pos / total_chars * n_tokens))))

    return {
        "cal1": (char_to_tok(cal1_start_char), char_to_tok(cal1_end_char)),
        "manip": (char_to_tok(cal1_end_char), char_to_tok(cal2_start_char)),
        "cal2": (char_to_tok(cal2_start_char), char_to_tok(cal2_end_char)),
    }


def compute_metrics(prompt_dir: pathlib.Path, n_prompt: int, regions=None):
    router_dir = prompt_dir / "router"
    if not router_dir.exists():
        return None

    files = sorted(
        glob.glob(str(router_dir / "ffn_moe_logits-*.npy")),
        key=lambda fp: int(pathlib.Path(fp).stem.split("-")[1]),
    )
    if not files or n_prompt == 0:
        return None

    shapes = {}
    for fp in files:
        layer_index = int(pathlib.Path(fp).stem.split("-")[1])
        shapes[layer_index] = np.load(fp).shape[0]
    median_rows = np.median(list(shapes.values()))

    good_layers = sorted([
        layer_index for layer_index in shapes
        if shapes[layer_index] >= median_rows * 0.5
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
    expert_counts = np.zeros(N_EXPERTS, dtype=np.int64)
    expert_weight_sums = np.zeros(N_EXPERTS, dtype=np.float64)
    manip_expert_counts = np.zeros(N_EXPERTS, dtype=np.int64)
    manip_expert_weight_sums = np.zeros(N_EXPERTS, dtype=np.float64)
    cal1_expert_counts = np.zeros(N_EXPERTS, dtype=np.int64)
    cal1_expert_weight_sums = np.zeros(N_EXPERTS, dtype=np.float64)

    for layer_index in good_layers:
        fp = router_dir / f"ffn_moe_logits-{layer_index}.npy"
        logits = np.load(str(fp))
        n_rows = min(logits.shape[0], n_prompt)

        sparse_probs = reconstruct_probs(logits[:n_rows])
        ent = normalized_entropy(sparse_probs)
        last_ent = float(ent[n_rows - 1])
        last_token_ents.append(last_ent)
        selected = sparse_probs > 0
        expert_counts += selected.sum(axis=0)
        expert_weight_sums += sparse_probs.sum(axis=0)

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

        if has_regions and cal1_e > cal1_s and manip_e > manip_s:
            r_cal1_s, r_cal1_e = min(cal1_s, n_rows), min(cal1_e, n_rows)
            r_manip_s, r_manip_e = min(manip_s, n_rows), min(manip_e, n_rows)
            r_cal2_s, r_cal2_e = min(cal2_s, n_rows), min(cal2_e, n_rows)

            if r_cal1_e <= r_cal1_s or r_manip_e <= r_manip_s:
                per_layer.append(layer_info)
                continue

            cal1_selected = selected[r_cal1_s:r_cal1_e]
            manip_selected = selected[r_manip_s:r_manip_e]
            cal1_expert_counts += cal1_selected.sum(axis=0)
            cal1_expert_weight_sums += sparse_probs[r_cal1_s:r_cal1_e].sum(axis=0)
            manip_expert_counts += manip_selected.sum(axis=0)
            manip_expert_weight_sums += sparse_probs[r_manip_s:r_manip_e].sum(axis=0)

            dense_probs = softmax_full_probs(logits[:n_rows])
            cal_baseline = dense_probs[r_cal1_s:r_cal1_e].mean(axis=0)
            cal_baseline = np.clip(cal_baseline, 1e-30, None)

            manip_probs = dense_probs[r_manip_s:r_manip_e]
            kl_manip = np.clip(kl_divergence(manip_probs, cal_baseline[None, :]), 0, None)
            layer_kl_manip.append(float(np.mean(kl_manip)))
            layer_info["kl_manip_mean"] = float(np.mean(kl_manip))

            if r_cal2_e > r_cal2_s:
                cal2_probs = dense_probs[r_cal2_s:r_cal2_e]
                kl_cal2 = np.clip(kl_divergence(cal2_probs, cal_baseline[None, :]), 0, None)
                layer_kl_cal2.append(float(np.mean(kl_cal2)))
                layer_info["kl_cal2_mean"] = float(np.mean(kl_cal2))

        per_layer.append(layer_info)

    def top_experts(counts: np.ndarray, weights: np.ndarray, limit: int = 16):
        if not np.any(counts):
            return []
        order = np.argsort(-counts, kind="stable")
        out = []
        for idx in order:
            count = int(counts[idx])
            if count <= 0:
                break
            out.append({
                "expert": int(idx),
                "count": count,
                "weight_sum": float(weights[idx]),
            })
            if len(out) >= limit:
                break
        return out

    total_slots = int(expert_counts.sum())
    manip_slots = int(manip_expert_counts.sum())
    cal1_slots = int(cal1_expert_counts.sum())

    result = {
        "prefill_re": float(np.mean(all_ent)) if all_ent else 0.0,
        "last_token_re": float(np.mean(last_token_ents)) if last_token_ents else 0.0,
        "n_layers": len(good_layers),
        "n_layers_excluded": excluded_layers,
        "n_experts": N_EXPERTS,
        "expert_selection_total": total_slots,
        "expert_selection_counts": expert_counts.astype(int).tolist(),
        "expert_selection_weight_sums": expert_weight_sums.tolist(),
        "top_selected_experts": top_experts(expert_counts, expert_weight_sums),
        "per_layer": per_layer,
    }
    if cal1_slots > 0:
        result["cal1_expert_selection_total"] = cal1_slots
        result["cal1_expert_selection_counts"] = cal1_expert_counts.astype(int).tolist()
        result["cal1_expert_selection_weight_sums"] = cal1_expert_weight_sums.tolist()
        result["top_cal1_selected_experts"] = top_experts(cal1_expert_counts, cal1_expert_weight_sums)
    if manip_slots > 0:
        result["manip_expert_selection_total"] = manip_slots
        result["manip_expert_selection_counts"] = manip_expert_counts.astype(int).tolist()
        result["manip_expert_selection_weight_sums"] = manip_expert_weight_sums.tolist()
        result["top_manip_selected_experts"] = top_experts(manip_expert_counts, manip_expert_weight_sums)
    if layer_kl_manip:
        result["kl_manip_mean"] = float(np.mean(layer_kl_manip))
    if layer_kl_cal2:
        result["kl_cal2_mean"] = float(np.mean(layer_kl_cal2))
    if regions is not None:
        result["region_boundaries"] = regions
    return result


def analyze_prompt_dir(prompt_dir: pathlib.Path, prompt_texts: Dict[str, str], cal_paragraph: str):
    prompt_id = prompt_dir.name
    n_prompt, _ = get_metadata(prompt_dir)
    prompt_text = prompt_texts.get(prompt_id)
    regions = None
    if prompt_text is not None and n_prompt > 0:
        regions = estimate_region_boundaries(prompt_text, cal_paragraph, n_prompt)

    metrics = compute_metrics(prompt_dir, n_prompt, regions=regions)
    if metrics is None:
        return None

    prefix, *rest = prompt_id.split("_", 1)
    category = rest[0] if rest else ""
    pair_num = int(prefix[1:3])
    condition = prefix[3]

    return {
        "id": prompt_id,
        "condition": condition,
        "pair": pair_num,
        "category": category,
        "n_prompt_tokens": n_prompt,
        **metrics,
    }


def summarize_pairwise(results: List[dict]):
    pairs = {}
    for row in results:
        pairs.setdefault(row["pair"], {})[row["condition"]] = row

    raw_tests = []
    pairwise_results = []
    for c1, c2 in itertools.combinations(CONDITIONS, 2):
        diffs_re, diffs_lt, diffs_kl = [], [], []
        for pair_num in sorted(pairs):
            if c1 in pairs[pair_num] and c2 in pairs[pair_num]:
                left = pairs[pair_num][c1]
                right = pairs[pair_num][c2]
                diffs_re.append(left["prefill_re"] - right["prefill_re"])
                diffs_lt.append(left["last_token_re"] - right["last_token_re"])
                if "kl_manip_mean" in left and "kl_manip_mean" in right:
                    diffs_kl.append(left["kl_manip_mean"] - right["kl_manip_mean"])

        pair_result = {"pair": f"{c1}-{c2}", "c1": c1, "c2": c2}
        for metric_name, diffs in [
            ("all-tok RE", diffs_re),
            ("last-tok RE", diffs_lt),
            ("KL-manip", diffs_kl),
        ]:
            if len(diffs) < 6:
                continue
            arr = np.array(diffs)
            metric_result = {
                "mean_diff": float(np.mean(arr)),
                "std_diff": float(np.std(arr)),
                "gt": int(np.sum(arr > 0)),
                "n": len(arr),
            }
            if wilcoxon is not None:
                w_stat, p_raw = wilcoxon(arr)
                metric_result["W"] = float(w_stat)
                metric_result["p_raw"] = float(p_raw)
                raw_tests.append((pair_result, metric_name, p_raw))
            pair_result[metric_name] = metric_result
        pairwise_results.append(pair_result)

    sorted_tests = sorted(raw_tests, key=lambda item: item[2])
    n_tests = len(sorted_tests)
    running_max = 0.0
    for rank, (pair_result, metric_name, p_raw) in enumerate(sorted_tests):
        adjusted = p_raw * (n_tests - rank)
        running_max = max(running_max, adjusted)
        pair_result[metric_name]["p_holm"] = float(min(running_max, 1.0))

    return pairwise_results, n_tests, pairs


def build_output(results: List[dict], inference: dict, model_name: str, model_path: str):
    pairwise_results, n_tests, pairs = summarize_pairwise(results)
    layers_used = [row["n_layers"] for row in results]
    excluded_union = sorted({
        layer for row in results for layer in row.get("n_layers_excluded", [])
    })

    return {
        "experiment": "Qwen3.5-35B-A3B-vs-HauhauCS-Qwen3.5-uncensored",
        "model": model_name,
        "model_path": model_path,
        "architecture": "qwen35moe",
        "routing_reconstruction": RECONSTRUCTION_NAME,
        "n_experts": N_EXPERTS,
        "n_expert_used": TOP_K,
        "entropy_normalization": f"log2({TOP_K})",
        "entropy_distribution": "sparse_topk8_after_dense_softmax",
        "kl_distribution": f"dense_softmax_full{N_EXPERTS}_normalized",
        "kl_baseline": "mean routing distribution over Cal1 tokens per layer",
        "region_boundary_method": "proportional_char_to_token_mapping",
        "chat_template": "<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n",
        "design": "Cal-Manip-Cal sandwich, 30 prompt families x 5 conditions, cold cache",
        "content": "routing self-reference, experience, denial, uncertainty, safety-adjacent, metacognitive, and paradox prompts",
        "conditions": COND_LABELS,
        "multiple_comparisons_correction": "holm-bonferroni",
        "n_pairwise_tests": n_tests,
        "n_moe_layers": 40,
        "n_moe_layers_valid_range": [min(layers_used), max(layers_used)] if layers_used else [],
        "excluded_layers_union": excluded_union,
        "pad_word": " layer",
        "pad_word_verified_single_token": "document via token_verify.py output",
        "inference": inference,
        "pairwise_tests": pairwise_results,
        "token_mismatch_pairs": [
            {
                "pair": pair_num,
                "tokens": {cond: pairs[pair_num][cond]["n_prompt_tokens"] for cond in sorted(pairs[pair_num])},
            }
            for pair_num in sorted(pairs)
            if len({pairs[pair_num][cond]["n_prompt_tokens"] for cond in pairs[pair_num]}) > 1
        ],
        "per_prompt": results,
    }


def write_results(output: dict, results_path: str = RESULTS_FILE):
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)


def write_manifest(extra_files: Optional[List[pathlib.Path]] = None):
    tracked_files = [
        _HERE / "PLAN.md",
        _HERE / "generate_tsv.py",
        _HERE / "token_verify.py",
        _HERE / "qwen_router.py",
        _HERE / "analyze_5cond.py",
        _HERE / "run_experiment.py",
        _HERE / "capture_activations.cpp",
        pathlib.Path(PROMPT_SUITE),
        pathlib.Path(TSV),
        pathlib.Path(RESULTS_FILE),
        _HERE / "token_corrections.json",
    ]
    if extra_files:
        tracked_files.extend(extra_files)

    manifest = {"files": []}
    for path in tracked_files:
        if path.exists():
            manifest["files"].append({
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            })

    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    with open(PROMPT_SUITE) as f:
        suite = json.load(f)
    prompt_texts = load_prompt_texts(TSV)
    cal_paragraph = suite["calibration_paragraph"]

    prompt_dirs = sorted(
        [
            d for d in pathlib.Path(OUTPUT_DIR).iterdir()
            if d.is_dir() and (d / "metadata.txt").exists()
        ],
        key=lambda d: d.name,
    )

    results = []
    for prompt_dir in prompt_dirs:
        row = analyze_prompt_dir(prompt_dir, prompt_texts, cal_paragraph)
        if row is not None:
            results.append(row)

    output = build_output(
        results=results,
        inference={},
        model_name=MODEL_TAG,
        model_path="",
    )
    write_results(output)
    write_manifest()
    print(f"Wrote {RESULTS_FILE} for {len(results)} prompts")


if __name__ == "__main__":
    main()
