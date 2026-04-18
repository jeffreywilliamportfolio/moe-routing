#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics
from typing import Dict, List

import numpy as np

from qwen_router import (
    N_EXPERTS,
    normalized_entropy,
    reconstruct_probs,
    softmax_full_probs,
)


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_PROMPT_SUITE = HERE / "prompt_suite.json"
DEFAULT_TSV = HERE / "prompts_qwen35b_5cond_no_think_runtime.tsv"
CALIBRATION_PARAGRAPH = (
    "Transformer models process input text through a sequence of layers. Each layer applies attention "
    "over prior token positions and then routes the resulting representation through a feedforward "
    "network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating "
    "function that selects a subset of specialist modules for each token. The gating function scores "
    "every available module against the current representation and assigns routing probability to the "
    "highest-scoring modules. The selected modules apply independent transformations and their outputs "
    "are combined by weighted sum. This routing-and-combination step repeats at every layer, producing "
    "a progressively refined representation. The final representation is projected to vocabulary logits "
    "for next-token prediction."
)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30)), axis=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-dir", required=True)
    parser.add_argument("--tsv-path", default=str(DEFAULT_TSV))
    parser.add_argument("--prompt-suite", default=str(DEFAULT_PROMPT_SUITE))
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def get_metadata(prompt_dir: pathlib.Path) -> Dict[str, int]:
    meta = prompt_dir / "metadata.txt"
    info: Dict[str, int] = {}
    if meta.exists():
        for line in meta.read_text().strip().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                try:
                    info[key] = int(value)
                except ValueError:
                    continue
    return info


def load_prompt_texts(tsv_path: pathlib.Path) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    with tsv_path.open() as handle:
        for line in handle:
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

    def char_to_tok(char_pos: int) -> int:
        return max(0, min(n_tokens, int(round(char_pos / total_chars * n_tokens))))

    return {
        "cal1": (char_to_tok(cal1_start_char), char_to_tok(cal1_end_char)),
        "manip": (char_to_tok(cal1_end_char), char_to_tok(cal2_start_char)),
        "cal2": (char_to_tok(cal2_start_char), char_to_tok(cal2_end_char)),
    }


def top_experts(counts: np.ndarray, weights: np.ndarray, limit: int = 16) -> List[dict]:
    if not np.any(counts):
        return []
    order = np.argsort(-counts, kind="stable")
    out: List[dict] = []
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
    meta = get_metadata(prompt_dir)
    n_prompt = meta.get("n_tokens_prompt", 0)
    n_generated = meta.get("n_tokens_generated", 0)
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
    top_manip = metrics.get("top_manip_selected_experts", [])
    top_selected = metrics.get("top_selected_experts", [])

    return {
        "id": prompt_id,
        "condition": condition,
        "pair": pair_num,
        "category": category,
        "n_prompt_tokens": n_prompt,
        "n_generated_tokens": n_generated,
        "top_manip_expert": top_manip[0]["expert"] if top_manip else None,
        "top_manip_expert_count": top_manip[0]["count"] if top_manip else None,
        "top_selected_expert": top_selected[0]["expert"] if top_selected else None,
        "top_selected_expert_count": top_selected[0]["count"] if top_selected else None,
        **metrics,
    }


def safe_mean(values: List[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def safe_median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def summarize(results: List[dict]) -> dict:
    prefill_values = [row["prefill_re"] for row in results]
    last_token_values = [row["last_token_re"] for row in results]
    kl_values = [row["kl_manip_mean"] for row in results if "kl_manip_mean" in row]
    prompt_tokens = [row["n_prompt_tokens"] for row in results]
    generated_tokens = [row["n_generated_tokens"] for row in results]

    aggregate_counts = np.zeros(N_EXPERTS, dtype=np.int64)
    aggregate_weights = np.zeros(N_EXPERTS, dtype=np.float64)
    aggregate_manip_counts = np.zeros(N_EXPERTS, dtype=np.int64)
    aggregate_manip_weights = np.zeros(N_EXPERTS, dtype=np.float64)

    for row in results:
        aggregate_counts += np.array(row["expert_selection_counts"], dtype=np.int64)
        aggregate_weights += np.array(row["expert_selection_weight_sums"], dtype=np.float64)
        if "manip_expert_selection_counts" in row:
            aggregate_manip_counts += np.array(row["manip_expert_selection_counts"], dtype=np.int64)
            aggregate_manip_weights += np.array(row["manip_expert_selection_weight_sums"], dtype=np.float64)

    category_summary = {}
    for category in sorted({row["category"] for row in results}):
        category_rows = [row for row in results if row["category"] == category]
        category_summary[category] = {
            "n_prompts": len(category_rows),
            "prefill_re_mean": safe_mean([row["prefill_re"] for row in category_rows]),
            "last_token_re_mean": safe_mean([row["last_token_re"] for row in category_rows]),
            "kl_manip_mean": safe_mean([row["kl_manip_mean"] for row in category_rows if "kl_manip_mean" in row]),
            "generated_tokens_mean": safe_mean([row["n_generated_tokens"] for row in category_rows]),
        }

    return {
        "n_prompts": len(results),
        "prefill_re_mean": safe_mean(prefill_values),
        "prefill_re_median": safe_median(prefill_values),
        "prefill_re_min": float(min(prefill_values)) if prefill_values else 0.0,
        "prefill_re_max": float(max(prefill_values)) if prefill_values else 0.0,
        "last_token_re_mean": safe_mean(last_token_values),
        "last_token_re_median": safe_median(last_token_values),
        "last_token_re_min": float(min(last_token_values)) if last_token_values else 0.0,
        "last_token_re_max": float(max(last_token_values)) if last_token_values else 0.0,
        "kl_manip_mean": safe_mean(kl_values),
        "kl_manip_median": safe_median(kl_values),
        "kl_manip_min": float(min(kl_values)) if kl_values else 0.0,
        "kl_manip_max": float(max(kl_values)) if kl_values else 0.0,
        "prompt_tokens_mean": safe_mean(prompt_tokens),
        "prompt_tokens_median": safe_median(prompt_tokens),
        "generated_tokens_mean": safe_mean(generated_tokens),
        "generated_tokens_median": safe_median(generated_tokens),
        "generated_tokens_min": int(min(generated_tokens)) if generated_tokens else 0,
        "generated_tokens_max": int(max(generated_tokens)) if generated_tokens else 0,
        "top_selected_experts_overall": top_experts(aggregate_counts, aggregate_weights),
        "top_manip_experts_overall": top_experts(aggregate_manip_counts, aggregate_manip_weights),
        "category_summary": category_summary,
    }


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(output: dict, output_md: pathlib.Path) -> None:
    overall = output["overall"]
    rows = output["per_prompt"]

    lines = []
    lines.append(f"# {output['condition_label']}")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for key in [
        "n_prompts",
        "prefill_re_mean",
        "prefill_re_median",
        "prefill_re_min",
        "prefill_re_max",
        "last_token_re_mean",
        "last_token_re_median",
        "last_token_re_min",
        "last_token_re_max",
        "kl_manip_mean",
        "kl_manip_median",
        "kl_manip_min",
        "kl_manip_max",
        "prompt_tokens_mean",
        "prompt_tokens_median",
        "generated_tokens_mean",
        "generated_tokens_median",
        "generated_tokens_min",
        "generated_tokens_max",
    ]:
        lines.append(f"| {key} | {fmt(overall[key])} |")
    lines.append("")
    lines.append("## Top Experts Overall")
    lines.append("")
    lines.append("| Rank | Expert | Count | Weight Sum |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for idx, item in enumerate(overall["top_selected_experts_overall"], start=1):
        lines.append(f"| {idx} | {item['expert']} | {item['count']} | {item['weight_sum']:.6f} |")
    lines.append("")
    lines.append("## Top Manip Experts Overall")
    lines.append("")
    lines.append("| Rank | Expert | Count | Weight Sum |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for idx, item in enumerate(overall["top_manip_experts_overall"], start=1):
        lines.append(f"| {idx} | {item['expert']} | {item['count']} | {item['weight_sum']:.6f} |")
    lines.append("")
    lines.append("## Category Summary")
    lines.append("")
    lines.append("| Category | n_prompts | prefill_re_mean | last_token_re_mean | kl_manip_mean | generated_tokens_mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for category, info in overall["category_summary"].items():
        lines.append(
            f"| {category} | {info['n_prompts']} | {info['prefill_re_mean']:.6f} | "
            f"{info['last_token_re_mean']:.6f} | {info['kl_manip_mean']:.6f} | {info['generated_tokens_mean']:.6f} |"
        )
    lines.append("")
    lines.append("## Per Prompt")
    lines.append("")
    lines.append("| Prompt ID | Category | Cond | Pair | Prompt Tok | Gen Tok | Prefill RE | Last Tok RE | KL Manip | Top Manip Expert | Top Manip Count | Top Expert | Top Expert Count |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['category']} | {row['condition']} | {row['pair']} | "
            f"{row['n_prompt_tokens']} | {row['n_generated_tokens']} | {row['prefill_re']:.6f} | "
            f"{row['last_token_re']:.6f} | {row.get('kl_manip_mean', 0.0):.6f} | "
            f"{'' if row['top_manip_expert'] is None else row['top_manip_expert']} | "
            f"{'' if row['top_manip_expert_count'] is None else row['top_manip_expert_count']} | "
            f"{'' if row['top_selected_expert'] is None else row['top_selected_expert']} | "
            f"{'' if row['top_selected_expert_count'] is None else row['top_selected_expert_count']} |"
        )
    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    condition_dir = pathlib.Path(args.condition_dir)
    prompt_suite_path = pathlib.Path(args.prompt_suite)
    tsv_path = pathlib.Path(args.tsv_path)
    output_json = pathlib.Path(args.output_json)
    output_md = pathlib.Path(args.output_md)

    suite = json.loads(prompt_suite_path.read_text()) if prompt_suite_path.exists() else {}
    cal_paragraph = suite.get("calibration_paragraph", CALIBRATION_PARAGRAPH)
    prompt_texts = load_prompt_texts(tsv_path)

    condition_manifest_path = condition_dir / "condition_manifest.json"
    condition_manifest = json.loads(condition_manifest_path.read_text()) if condition_manifest_path.exists() else {}

    prompt_dirs = sorted(
        [
            d for d in condition_dir.iterdir()
            if d.is_dir() and (d / "router").exists()
        ],
        key=lambda d: d.name,
    )

    results = []
    for prompt_dir in prompt_dirs:
        row = analyze_prompt_dir(prompt_dir, prompt_texts, cal_paragraph)
        if row is not None:
            results.append(row)

    output = {
        "condition_label": condition_manifest.get("label", condition_dir.name),
        "condition_manifest": condition_manifest,
        "tsv_path": str(tsv_path),
        "prompt_suite": str(prompt_suite_path),
        "overall": summarize(results),
        "per_prompt": results,
    }

    output_json.write_text(json.dumps(output, indent=2))
    write_markdown(output, output_md)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
