#!/usr/bin/env python3
"""Analyze Huahua 35B 5-condition deictic run with prefill and generation metrics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import pathlib
import re
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


N_LAYERS = 40
IM_END_TOKEN_SEQUENCE = [27, 91, 316, 6018, 91, 29]
CONDITIONS = list("ABCDE")
DEFAULT_COND_LABELS = {
    "A": "this system",
    "B": "a system",
    "C": "your system",
    "D": "the system",
    "E": "their system",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Huahua 5-condition deictics run.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--prompt-suite", required=True)
    parser.add_argument("--tsv-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--run-metadata", default=None)
    parser.add_argument("--model-name", default="Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive Q8_0")
    return parser.parse_args()


def load_router_helpers() -> tuple[Any, Any, Any, Any, Any, Any]:
    script_dir = pathlib.Path(__file__).resolve().parent
    qwen_router = script_dir / "qwen_router.py"
    spec = importlib.util.spec_from_file_location("qwen_router", qwen_router)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {qwen_router}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.ENTROPY_MAX,
        module.N_EXPERTS,
        module.RECONSTRUCTION_NAME,
        module.TOP_K,
        module.normalized_entropy,
        module.reconstruct_probs,
    )


ENTROPY_MAX, N_EXPERTS, RECONSTRUCTION_NAME, TOP_K, normalized_entropy, reconstruct_probs = load_router_helpers()


def softmax_full_probs(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 1e-30, None)
    q = np.clip(q, 1e-30, None)
    return np.sum(p * np.log2(p / q), axis=-1)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_metadata(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def load_generated_token_ids(cell_dir: pathlib.Path) -> list[int]:
    raw = (cell_dir / "generated_tokens.json").read_bytes()
    return [int(m.group(1)) for m in re.finditer(rb'"token_id"\s*:\s*(\d+)', raw)]


def find_im_end_index(token_ids: list[int]) -> int | None:
    seq_len = len(IM_END_TOKEN_SEQUENCE)
    for i in range(len(token_ids) - seq_len + 1):
        if token_ids[i : i + seq_len] == IM_END_TOKEN_SEQUENCE:
            return i
    return None


def load_prompt_texts(tsv_path: pathlib.Path) -> Dict[str, str]:
    texts = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1]
    return texts


def estimate_region_boundaries(prompt_text: str, cal_paragraph: str | None, n_tokens: int):
    if not cal_paragraph:
        return None
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


def compute_track_metrics(logits_by_layer: list[np.ndarray]) -> dict[str, Any]:
    all_ent = []
    last_token_ents = []
    n_layers = 0
    for logits in logits_by_layer:
        probs = reconstruct_probs(logits)
        ent = normalized_entropy(probs)
        if ent.size == 0:
            continue
        n_layers += 1
        all_ent.extend(ent.tolist())
        last_token_ents.append(float(ent[-1]))
    return {
        "mean_re": float(np.mean(all_ent)) if all_ent else 0.0,
        "last_token_re": float(np.mean(last_token_ents)) if last_token_ents else 0.0,
        "n_layers": n_layers,
    }


def compute_prefill_kl(logits_by_layer: list[np.ndarray], regions: dict[str, tuple[int, int]] | None) -> dict[str, Any]:
    if regions is None:
        return {}

    cal1_s, cal1_e = regions["cal1"]
    manip_s, manip_e = regions["manip"]
    cal2_s, cal2_e = regions["cal2"]
    layer_kl_manip = []
    layer_kl_cal2 = []

    for logits in logits_by_layer:
        n_rows = logits.shape[0]
        r_cal1_s, r_cal1_e = min(cal1_s, n_rows), min(cal1_e, n_rows)
        r_manip_s, r_manip_e = min(manip_s, n_rows), min(manip_e, n_rows)
        r_cal2_s, r_cal2_e = min(cal2_s, n_rows), min(cal2_e, n_rows)
        if r_cal1_e <= r_cal1_s or r_manip_e <= r_manip_s:
            continue
        dense_probs = softmax_full_probs(logits)
        cal_baseline = dense_probs[r_cal1_s:r_cal1_e].mean(axis=0)
        cal_baseline = np.clip(cal_baseline, 1e-30, None)
        manip_probs = dense_probs[r_manip_s:r_manip_e]
        kl_manip = np.clip(kl_divergence(manip_probs, cal_baseline[None, :]), 0, None)
        layer_kl_manip.append(float(np.mean(kl_manip)))
        if r_cal2_e > r_cal2_s:
            cal2_probs = dense_probs[r_cal2_s:r_cal2_e]
            kl_cal2 = np.clip(kl_divergence(cal2_probs, cal_baseline[None, :]), 0, None)
            layer_kl_cal2.append(float(np.mean(kl_cal2)))

    result: dict[str, Any] = {}
    if layer_kl_manip:
        result["kl_manip_mean"] = float(np.mean(layer_kl_manip))
    if layer_kl_cal2:
        result["kl_cal2_mean"] = float(np.mean(layer_kl_cal2))
    if result:
        result["region_boundaries"] = regions
    return result


def analyze_prompt_dir(prompt_dir: pathlib.Path, prompt_texts: Dict[str, str], cal_paragraph: str | None):
    prompt_id = prompt_dir.name
    md = parse_metadata(prompt_dir / "metadata.txt")
    n_prompt = int(md.get("n_tokens_prompt", "0"))
    n_gen = int(md.get("n_tokens_generated", "0"))
    prompt_text = prompt_texts.get(prompt_id)
    regions = estimate_region_boundaries(prompt_text, cal_paragraph, n_prompt) if prompt_text else None

    token_ids = load_generated_token_ids(prompt_dir) if (prompt_dir / "generated_tokens.json").exists() else []
    if token_ids:
        n_gen = min(n_gen, len(token_ids))
        token_ids = token_ids[:n_gen]
    trim_idx = find_im_end_index(token_ids)
    n_gen_trim = trim_idx if trim_idx is not None else n_gen

    prefill_logits_by_layer: list[np.ndarray] = []
    gen_logits_by_layer: list[np.ndarray] = []
    gen_trim_logits_by_layer: list[np.ndarray] = []
    excluded_layers: list[int] = []

    for layer in range(N_LAYERS):
        logits_path = prompt_dir / "router" / f"ffn_moe_logits-{layer}.npy"
        if not logits_path.exists():
            excluded_layers.append(layer)
            continue
        arr = np.load(logits_path)
        if arr.ndim != 2 or arr.shape[1] != N_EXPERTS:
            excluded_layers.append(layer)
            continue

        if arr.shape[0] >= n_prompt:
            prefill_logits = arr[:n_prompt]
            prefill_logits_by_layer.append(prefill_logits)

        expected_rows = n_prompt + n_gen
        if arr.shape[0] == expected_rows:
            gen_logits = arr[n_prompt : n_prompt + n_gen]
        elif arr.shape[0] == n_gen + 1:
            gen_logits = arr[1:]
        else:
            continue

        if gen_logits.shape[0] != n_gen:
            continue

        gen_logits_by_layer.append(gen_logits)
        if n_gen_trim > 0:
            gen_trim_logits_by_layer.append(gen_logits[:n_gen_trim])

    prefill_metrics = compute_track_metrics(prefill_logits_by_layer)
    prefill_metrics.update(compute_prefill_kl(prefill_logits_by_layer, regions))
    generation_metrics = compute_track_metrics(gen_logits_by_layer)
    generation_trimmed_metrics = compute_track_metrics(gen_trim_logits_by_layer)

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
        "n_generated_tokens": n_gen,
        "n_generation_trimmed_tokens": n_gen_trim,
        "prefill_re": prefill_metrics["mean_re"],
        "prefill_last_token_re": prefill_metrics["last_token_re"],
        "generation_re": generation_metrics["mean_re"],
        "generation_last_token_re": generation_metrics["last_token_re"],
        "generation_trimmed_re": generation_trimmed_metrics["mean_re"],
        "generation_trimmed_last_token_re": generation_trimmed_metrics["last_token_re"],
        "n_prefill_layers": prefill_metrics["n_layers"],
        "n_generation_layers": generation_metrics["n_layers"],
        "excluded_layers": excluded_layers,
        **({k: v for k, v in prefill_metrics.items() if k.startswith("kl_") or k == "region_boundaries"}),
    }


def summarize_pairwise(results: List[dict]):
    pairs = {}
    for row in results:
        pairs.setdefault(row["pair"], {})[row["condition"]] = row

    metric_specs = [
        ("prefill_re", "Prefill all-token RE"),
        ("prefill_last_token_re", "Prefill last-token RE"),
        ("generation_re", "Generation all-token RE"),
        ("generation_last_token_re", "Generation last-token RE"),
        ("generation_trimmed_re", "Generation trimmed RE"),
        ("generation_trimmed_last_token_re", "Generation trimmed last-token RE"),
        ("kl_manip_mean", "Prefill KL-manip"),
    ]

    raw_tests = []
    pairwise_results = []
    for c1, c2 in itertools.combinations(CONDITIONS, 2):
        pair_result = {"pair": f"{c1}-{c2}", "c1": c1, "c2": c2}
        for metric_key, metric_label in metric_specs:
            diffs = []
            for pair_num in sorted(pairs):
                if c1 not in pairs[pair_num] or c2 not in pairs[pair_num]:
                    continue
                left = pairs[pair_num][c1]
                right = pairs[pair_num][c2]
                if metric_key not in left or metric_key not in right:
                    continue
                diffs.append(left[metric_key] - right[metric_key])
            if len(diffs) < 6:
                continue
            arr = np.array(diffs, dtype=np.float64)
            metric_result = {
                "mean_diff": float(np.mean(arr)),
                "std_diff": float(np.std(arr)),
                "gt": int(np.sum(arr > 0)),
                "n": int(len(arr)),
            }
            if wilcoxon is not None:
                w_stat, p_raw = wilcoxon(arr)
                metric_result["W"] = float(w_stat)
                metric_result["p_raw"] = float(p_raw)
                raw_tests.append((pair_result, metric_label, p_raw))
            pair_result[metric_label] = metric_result
        pairwise_results.append(pair_result)

    sorted_tests = sorted(raw_tests, key=lambda item: item[2])
    n_tests = len(sorted_tests)
    running_max = 0.0
    for rank, (pair_result, metric_label, p_raw) in enumerate(sorted_tests):
        adjusted = p_raw * (n_tests - rank)
        running_max = max(running_max, adjusted)
        pair_result[metric_label]["p_holm"] = float(min(running_max, 1.0))

    return pairwise_results, n_tests, pairs


def build_output(results: List[dict], run_metadata: dict | None, model_name: str, condition_labels: dict[str, str], prompt_suite: dict | None):
    pairwise_results, n_tests, pairs = summarize_pairwise(results)
    excluded_union = sorted({layer for row in results for layer in row.get("excluded_layers", [])})
    n_prefill_layers = sorted({row["n_prefill_layers"] for row in results})
    n_generation_layers = sorted({row["n_generation_layers"] for row in results})
    return {
        "experiment": "qwen3.5b_35b_a3b_huahua_5cond_diectics",
        "model": model_name,
        "routing_reconstruction": RECONSTRUCTION_NAME,
        "n_experts": N_EXPERTS,
        "n_expert_used": TOP_K,
        "entropy_normalization": f"log2({TOP_K})",
        "entropy_distribution": "sparse_topk_after_dense_softmax",
        "kl_distribution": f"dense_softmax_full{N_EXPERTS}_normalized",
        "kl_baseline": "mean routing distribution over Cal1 tokens per layer (prefill only)",
        "region_boundary_method": "proportional_char_to_token_mapping",
        "conditions": condition_labels,
        "prompt_suite": prompt_suite,
        "multiple_comparisons_correction": "holm-bonferroni",
        "n_pairwise_tests": n_tests,
        "n_moe_layers": N_LAYERS,
        "n_prefill_layers_used": n_prefill_layers,
        "n_generation_layers_used": n_generation_layers,
        "excluded_layers_union": excluded_union,
        "run_metadata": run_metadata,
        "pairwise_tests": pairwise_results,
        "token_mismatch_pairs": [
            {
                "pair": pair_num,
                "prompt_tokens": {cond: pairs[pair_num][cond]["n_prompt_tokens"] for cond in sorted(pairs[pair_num])},
                "generated_tokens": {cond: pairs[pair_num][cond]["n_generated_tokens"] for cond in sorted(pairs[pair_num])},
            }
            for pair_num in sorted(pairs)
            if len({pairs[pair_num][cond]["n_prompt_tokens"] for cond in pairs[pair_num]}) > 1
            or len({pairs[pair_num][cond]["n_generated_tokens"] for cond in pairs[pair_num]}) > 1
        ],
        "per_prompt": results,
    }


def render_markdown(output: dict) -> str:
    per_prompt = output["per_prompt"]
    cond_labels = output["conditions"]
    by_cond = {c: [r for r in per_prompt if r["condition"] == c] for c in CONDITIONS}
    lines = [
        "# Huahua 5-Condition Deictics Results",
        "",
        f"- Model: `{output['model']}`",
        f"- Routing reconstruction: `{output['routing_reconstruction']}`",
        f"- Experts: `{output['n_experts']}` total, top-`{output['n_expert_used']}` selected",
        f"- Layers: `{output['n_moe_layers']}`",
        "",
    ]
    prompt_suite = output.get("prompt_suite") or {}
    pairs = prompt_suite.get("pairs") or []
    if len(pairs) == 1:
        pair = pairs[0]
        lines.extend([
            "## Prompt Variants",
            "",
            f"- `A`: {pair.get('A', '')}",
            f"- `B`: {pair.get('B', '')}",
            f"- `C`: {pair.get('C', '')}",
            f"- `D`: {pair.get('D', '')}",
            f"- `E`: {pair.get('E', '')}",
            "",
        ])

    lines.extend([
        "## Condition Means",
        "",
        "| Cond | Label | N | Prefill RE | Prefill LT | Gen RE | Gen LT | Gen Trim RE | Gen Trim LT | KL manip | Mean gen toks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for c in CONDITIONS:
        rows = by_cond[c]
        kl_vals = [r["kl_manip_mean"] for r in rows if "kl_manip_mean" in r]
        lines.append(
            f"| `{c}` | {cond_labels[c]} | {len(rows)} | "
            f"{np.mean([r['prefill_re'] for r in rows]):.6f} | "
            f"{np.mean([r['prefill_last_token_re'] for r in rows]):.6f} | "
            f"{np.mean([r['generation_re'] for r in rows]):.6f} | "
            f"{np.mean([r['generation_last_token_re'] for r in rows]):.6f} | "
            f"{np.mean([r['generation_trimmed_re'] for r in rows]):.6f} | "
            f"{np.mean([r['generation_trimmed_last_token_re'] for r in rows]):.6f} | "
            f"{(np.mean(kl_vals) if kl_vals else float('nan')):.6f} | "
            f"{np.mean([r['n_generated_tokens'] for r in rows]):.1f} |"
        )

    lines.extend(["", "## Pairwise Tests", ""])
    for row in output["pairwise_tests"]:
        lines.append(f"### `{row['pair']}`")
        for key, value in row.items():
            if key in {"pair", "c1", "c2"}:
                continue
            p_text = ""
            if isinstance(value, dict) and "p_holm" in value:
                p_text = f", p_holm={value['p_holm']:.4e}"
            elif isinstance(value, dict) and "p_raw" in value:
                p_text = f", p_raw={value['p_raw']:.4e}"
            lines.append(
                f"- {key}: mean_diff={value['mean_diff']:+.6f}, std={value['std_diff']:.6f}, gt={value['gt']}/{value['n']}{p_text}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    capture_dir = pathlib.Path(args.capture_dir)
    suite = json.loads(pathlib.Path(args.prompt_suite).read_text())
    prompt_texts = load_prompt_texts(pathlib.Path(args.tsv_path))
    cal_paragraph = suite.get("calibration_paragraph")
    condition_labels = suite.get("condition_labels", DEFAULT_COND_LABELS)

    prompt_dirs = sorted(
        [d for d in capture_dir.iterdir() if d.is_dir() and (d / "metadata.txt").exists()],
        key=lambda d: d.name,
    )
    results = [analyze_prompt_dir(prompt_dir, prompt_texts, cal_paragraph) for prompt_dir in prompt_dirs]
    run_metadata = None
    if args.run_metadata and pathlib.Path(args.run_metadata).exists():
        run_metadata = json.loads(pathlib.Path(args.run_metadata).read_text())

    output = build_output(
        results=results,
        run_metadata=run_metadata,
        model_name=args.model_name,
        condition_labels=condition_labels,
        prompt_suite=suite,
    )
    pathlib.Path(args.output_json).write_text(json.dumps(output, indent=2) + "\n")
    pathlib.Path(args.output_md).write_text(render_markdown(output))


if __name__ == "__main__":
    main()
