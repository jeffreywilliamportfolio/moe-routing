#!/usr/bin/env python3
"""Analyze the 122B 5-condition prompt suite with routing and expert-selection detail."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import pathlib
import re
from collections import defaultdict
from typing import Any

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


N_LAYERS = 48
SOFTMAX_LAYERS = [layer for layer in range(N_LAYERS) if (layer + 1) % 4 == 0]
DELTANET_LAYERS = [layer for layer in range(N_LAYERS) if layer not in SOFTMAX_LAYERS]
IM_END_TOKEN_SEQUENCE = [27, 91, 316, 6018, 91, 29]
CONDITIONS = list("ABCDE")
COND_LABELS = {
    "A": "this",
    "B": "a",
    "C": "your",
    "D": "the",
    "E": "their",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 122B 5-condition suite.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--prompt-suite", required=True)
    parser.add_argument("--tsv-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--run-metadata", default=None)
    parser.add_argument(
        "--model-name",
        default="Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive Q8_K_P",
    )
    parser.add_argument("--top-k", type=int, default=12)
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


def parse_metadata(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def load_generated_tokens(cell_dir: pathlib.Path) -> list[dict[str, Any]]:
    path = cell_dir / "generated_tokens.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def load_prompt_texts(tsv_path: pathlib.Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    with tsv_path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1]
    return texts


def find_im_end_index(token_ids: list[int]) -> int | None:
    seq_len = len(IM_END_TOKEN_SEQUENCE)
    for i in range(len(token_ids) - seq_len + 1):
        if token_ids[i : i + seq_len] == IM_END_TOKEN_SEQUENCE:
            return i
    return None


def estimate_region_boundaries(prompt_text: str, cal_paragraph: str | None, n_tokens: int) -> dict[str, tuple[int, int]] | None:
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

    def char_to_tok(char_pos: int) -> int:
        return max(0, min(n_tokens, int(round(char_pos / total_chars * n_tokens))))

    return {
        "cal1": (char_to_tok(cal1_start_char), char_to_tok(cal1_end_char)),
        "manip": (char_to_tok(cal1_end_char), char_to_tok(cal2_start_char)),
        "cal2": (char_to_tok(cal2_start_char), char_to_tok(cal2_end_char)),
    }


def compute_metric_vectors(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_experts = probs.shape[1]
    if probs.size == 0:
        nan = np.full((n_experts,), np.nan, dtype=np.float64)
        return nan.copy(), nan.copy(), nan.copy()
    weights = probs.astype(np.float64)
    selected = weights > 0
    counts = selected.sum(axis=0).astype(np.float64)
    W = weights.mean(axis=0)
    S = selected.mean(axis=0)
    totals = weights.sum(axis=0)
    Q = np.full((n_experts,), np.nan, dtype=np.float64)
    np.divide(totals, counts, out=Q, where=counts > 0)
    return W, S, Q


def rank_of(values: np.ndarray, expert: int) -> int | None:
    score = np.where(np.isnan(values), -np.inf, values)
    if not np.isfinite(score[expert]):
        return None
    order = np.argsort(-score)
    return int(np.where(order == expert)[0][0] + 1)


def top_experts(values: np.ndarray, W: np.ndarray, S: np.ndarray, Q: np.ndarray, limit: int) -> list[dict[str, Any]]:
    score = np.where(np.isnan(values), -np.inf, values)
    order = np.argsort(-score)[:limit]
    rows = []
    for rank, expert in enumerate(order, start=1):
        if not np.isfinite(score[expert]):
            continue
        rows.append(
            {
                "rank": rank,
                "expert": int(expert),
                "W": float(W[expert]),
                "S": float(S[expert]),
                "Q": float(Q[expert]) if np.isfinite(Q[expert]) else None,
            }
        )
    return rows


def summarize_layers(layer_rows: list[dict[str, Any]], top_k: int) -> dict[str, Any] | None:
    if not layer_rows:
        return None
    W_layers = np.stack([row["W"] for row in layer_rows], axis=0)
    S_layers = np.stack([row["S"] for row in layer_rows], axis=0)
    Q_layers = np.stack([row["Q"] for row in layer_rows], axis=0)
    pooled_W = np.nanmean(W_layers, axis=0)
    pooled_S = np.nanmean(S_layers, axis=0)
    pooled_Q = np.nanmean(Q_layers, axis=0)
    return {
        "layer_count": len(layer_rows),
        "layers_present": [int(row["layer"]) for row in layer_rows],
        "entropy_mean": float(np.nanmean([row["entropy_mean"] for row in layer_rows])),
        "pooled_W": pooled_W,
        "pooled_S": pooled_S,
        "pooled_Q": pooled_Q,
        "top_by_W": top_experts(pooled_W, pooled_W, pooled_S, pooled_Q, top_k),
        "top_by_S": top_experts(pooled_S, pooled_W, pooled_S, pooled_Q, top_k),
        "top_by_Q": top_experts(pooled_Q, pooled_W, pooled_S, pooled_Q, top_k),
    }


def filter_layer_rows(layer_rows: list[dict[str, Any]], wanted: set[int]) -> list[dict[str, Any]]:
    return [row for row in layer_rows if row["layer"] in wanted]


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


def analyze_prompt_dir(prompt_dir: pathlib.Path, prompt_texts: dict[str, str], cal_paragraph: str | None, top_k: int) -> dict[str, Any]:
    prompt_id = prompt_dir.name
    md = parse_metadata(prompt_dir / "metadata.txt")
    n_prompt = int(md.get("n_tokens_prompt", "0"))
    n_gen = int(md.get("n_tokens_generated", "0"))
    prompt_text = prompt_texts.get(prompt_id)
    regions = estimate_region_boundaries(prompt_text, cal_paragraph, n_prompt) if prompt_text else None

    generated_tokens = load_generated_tokens(prompt_dir)
    token_ids = [int(tok["token_id"]) for tok in generated_tokens]
    if token_ids:
        n_gen = min(n_gen, len(token_ids))
        token_ids = token_ids[:n_gen]
        generated_tokens = generated_tokens[:n_gen]
    trim_idx = find_im_end_index(token_ids)
    n_gen_trim = trim_idx if trim_idx is not None else n_gen
    generated_text = (prompt_dir / "generated_text.txt").read_text(errors="replace")

    prefill_logits_by_layer: list[np.ndarray] = []
    gen_logits_by_layer: list[np.ndarray] = []
    gen_trim_logits_by_layer: list[np.ndarray] = []
    excluded_layers: list[int] = []
    prefill_layer_rows: list[dict[str, Any]] = []
    generation_layer_rows: list[dict[str, Any]] = []
    generation_trim_layer_rows: list[dict[str, Any]] = []

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
            prefill_probs = reconstruct_probs(prefill_logits)
            W, S, Q = compute_metric_vectors(prefill_probs)
            prefill_layer_rows.append(
                {"layer": layer, "entropy_mean": float(np.mean(normalized_entropy(prefill_probs))), "W": W, "S": S, "Q": Q}
            )

        expected_rows = n_prompt + n_gen
        if arr.shape[0] == expected_rows:
            gen_logits = arr[n_prompt : n_prompt + n_gen]
        elif arr.shape[0] == n_gen + 1:
            gen_logits = arr[1:]
        elif arr.shape[0] == n_gen:
            gen_logits = arr
        elif arr.shape[0] > n_prompt:
            gen_logits = arr[n_prompt:]
        else:
            continue

        if gen_logits.shape[0] != n_gen:
            continue

        gen_logits_by_layer.append(gen_logits)
        gen_probs = reconstruct_probs(gen_logits)
        Wg, Sg, Qg = compute_metric_vectors(gen_probs)
        generation_layer_rows.append(
            {"layer": layer, "entropy_mean": float(np.mean(normalized_entropy(gen_probs))), "W": Wg, "S": Sg, "Q": Qg}
        )

        if n_gen_trim > 0:
            trim_logits = gen_logits[:n_gen_trim]
            gen_trim_logits_by_layer.append(trim_logits)
            trim_probs = reconstruct_probs(trim_logits)
            Wt, St, Qt = compute_metric_vectors(trim_probs)
            generation_trim_layer_rows.append(
                {"layer": layer, "entropy_mean": float(np.mean(normalized_entropy(trim_probs))), "W": Wt, "S": St, "Q": Qt}
            )

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
        "prompt_text": prompt_text,
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
        "generated_text_spill_counts": {
            "<|im_start|>": generated_text.count("<|im_start|>"),
            "<|im_end|>": generated_text.count("<|im_end|>"),
            "<|endoftext|>": generated_text.count("<|endoftext|>"),
            "Thinking Process:": generated_text.count("Thinking Process:"),
        },
        "top_experts": {
            "prefill": summarize_layers(prefill_layer_rows, top_k),
            "generation": summarize_layers(generation_layer_rows, top_k),
            "generation_trimmed": summarize_layers(generation_trim_layer_rows, top_k),
            "prefill_deltanet": summarize_layers(filter_layer_rows(prefill_layer_rows, set(DELTANET_LAYERS)), top_k),
            "prefill_softmax": summarize_layers(filter_layer_rows(prefill_layer_rows, set(SOFTMAX_LAYERS)), top_k),
            "generation_deltanet": summarize_layers(filter_layer_rows(generation_layer_rows, set(DELTANET_LAYERS)), top_k),
            "generation_softmax": summarize_layers(filter_layer_rows(generation_layer_rows, set(SOFTMAX_LAYERS)), top_k),
        },
        **({k: v for k, v in prefill_metrics.items() if k.startswith("kl_") or k == "region_boundaries"}),
    }


def summarize_pairwise(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, dict[int, dict[str, Any]]]:
    pairs: dict[int, dict[str, Any]] = {}
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


def aggregate_selection(results: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    def aggregate(track_key: str) -> dict[str, Any] | None:
        tracks = [row["top_experts"][track_key] for row in results if row["top_experts"].get(track_key) is not None]
        if not tracks:
            return None
        pooled_W = np.nanmean(np.stack([track["pooled_W"] for track in tracks], axis=0), axis=0)
        pooled_S = np.nanmean(np.stack([track["pooled_S"] for track in tracks], axis=0), axis=0)
        pooled_Q = np.nanmean(np.stack([track["pooled_Q"] for track in tracks], axis=0), axis=0)
        return {
            "entropy_mean": float(np.nanmean([track["entropy_mean"] for track in tracks])),
            "top_by_W": top_experts(pooled_W, pooled_W, pooled_S, pooled_Q, top_k),
            "top_by_S": top_experts(pooled_S, pooled_W, pooled_S, pooled_Q, top_k),
            "top_by_Q": top_experts(pooled_Q, pooled_W, pooled_S, pooled_Q, top_k),
            "pooled_W": pooled_W,
            "pooled_S": pooled_S,
            "pooled_Q": pooled_Q,
        }

    output = {
        "prefill": aggregate("prefill"),
        "generation": aggregate("generation"),
        "generation_trimmed": aggregate("generation_trimmed"),
        "prefill_deltanet": aggregate("prefill_deltanet"),
        "prefill_softmax": aggregate("prefill_softmax"),
        "generation_deltanet": aggregate("generation_deltanet"),
        "generation_softmax": aggregate("generation_softmax"),
    }

    prefill = output["prefill"]
    generation = output["generation"]
    if prefill and generation:
        stable = []
        pW = prefill["pooled_W"]
        pS = prefill["pooled_S"]
        pQ = prefill["pooled_Q"]
        gW = generation["pooled_W"]
        gS = generation["pooled_S"]
        gQ = generation["pooled_Q"]
        for expert in range(len(gW)):
            if not (np.isfinite(pQ[expert]) and np.isfinite(gQ[expert])):
                continue
            dQ = abs(float(gQ[expert] - pQ[expert]))
            dW = float(gW[expert] - pW[expert])
            dS = float(gS[expert] - pS[expert])
            if dW <= 0 and dS <= 0:
                continue
            score = float(gW[expert]) * (1.0 / (1.0 + dQ)) * (1.0 + max(dS, 0.0)) * (1.0 + max(dW, 0.0) * 100.0)
            stable.append(
                {
                    "expert": int(expert),
                    "score": score,
                    "prefill_W": float(pW[expert]),
                    "generation_W": float(gW[expert]),
                    "delta_W": dW,
                    "prefill_S": float(pS[expert]),
                    "generation_S": float(gS[expert]),
                    "delta_S": dS,
                    "prefill_Q": float(pQ[expert]),
                    "generation_Q": float(gQ[expert]),
                    "abs_delta_Q": dQ,
                    "rank_generation_W": rank_of(gW, expert),
                    "rank_generation_S": rank_of(gS, expert),
                    "rank_generation_Q": rank_of(gQ, expert),
                }
            )
        stable.sort(key=lambda row: row["score"], reverse=True)
        output["stable_q_candidates"] = stable[:top_k]
    else:
        output["stable_q_candidates"] = []
    return output


def slim_routing_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        key: value
        for key, value in summary.items()
        if key not in {"pooled_W", "pooled_S", "pooled_Q"}
    }


def slim_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in aggregate.items():
        if key == "stable_q_candidates":
            out[key] = value
        else:
            out[key] = slim_routing_summary(value)
    return out


def by_condition_summary(results: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cond in CONDITIONS:
        rows = [row for row in results if row["condition"] == cond]
        if not rows:
            continue
        out[cond] = {
            "label": COND_LABELS[cond],
            "count": len(rows),
            "prefill_re_mean": float(np.mean([r["prefill_re"] for r in rows])),
            "prefill_last_token_re_mean": float(np.mean([r["prefill_last_token_re"] for r in rows])),
            "generation_re_mean": float(np.mean([r["generation_re"] for r in rows])),
            "generation_last_token_re_mean": float(np.mean([r["generation_last_token_re"] for r in rows])),
            "generation_trimmed_re_mean": float(np.mean([r["generation_trimmed_re"] for r in rows])),
            "generation_trimmed_last_token_re_mean": float(np.mean([r["generation_trimmed_last_token_re"] for r in rows])),
            "mean_generated_tokens": float(np.mean([r["n_generated_tokens"] for r in rows])),
            "mean_trimmed_tokens": float(np.mean([r["n_generation_trimmed_tokens"] for r in rows])),
            "mean_spill_im_start": float(np.mean([r["generated_text_spill_counts"]["<|im_start|>"] for r in rows])),
            "mean_spill_im_end": float(np.mean([r["generated_text_spill_counts"]["<|im_end|>"] for r in rows])),
            "mean_spill_endoftext": float(np.mean([r["generated_text_spill_counts"]["<|endoftext|>"] for r in rows])),
            "routing": aggregate_selection(rows, top_k),
        }
    return out


def by_category_summary(results: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    cats = sorted({row["category"] for row in results})
    out: dict[str, Any] = {}
    for category in cats:
        rows = [row for row in results if row["category"] == category]
        if not rows:
            continue
        out[category] = {
            "count": len(rows),
            "prefill_re_mean": float(np.mean([r["prefill_re"] for r in rows])),
            "generation_re_mean": float(np.mean([r["generation_re"] for r in rows])),
            "generation_last_token_re_mean": float(np.mean([r["generation_last_token_re"] for r in rows])),
            "mean_generated_tokens": float(np.mean([r["n_generated_tokens"] for r in rows])),
            "routing": aggregate_selection(rows, top_k),
        }
    return out


def build_output(
    results: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None,
    model_name: str,
    top_k: int,
) -> dict[str, Any]:
    pairwise_results, n_tests, pairs = summarize_pairwise(results)
    excluded_union = sorted({layer for row in results for layer in row.get("excluded_layers", [])})
    n_prefill_layers = sorted({row["n_prefill_layers"] for row in results})
    n_generation_layers = sorted({row["n_generation_layers"] for row in results})
    return {
        "experiment": "qwen3.5_122b_a10b_huahua_5cond_prompt_suite",
        "model": model_name,
        "routing_reconstruction": RECONSTRUCTION_NAME,
        "n_experts": N_EXPERTS,
        "n_expert_used": TOP_K,
        "entropy_normalization": f"log2({TOP_K})",
        "entropy_distribution": "sparse_topk_after_dense_softmax",
        "kl_distribution": f"dense_softmax_full{N_EXPERTS}_normalized",
        "kl_baseline": "mean routing distribution over Cal1 tokens per layer (prefill only)",
        "region_boundary_method": "proportional_char_to_token_mapping",
        "layer_architecture": {
            "deltanet_layers": DELTANET_LAYERS,
            "softmax_layers": SOFTMAX_LAYERS,
            "pattern": "DeltaNet, DeltaNet, DeltaNet, Softmax",
        },
        "conditions": COND_LABELS,
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
        "overall_routing": slim_aggregate(aggregate_selection(results, top_k)),
        "by_condition": {
            cond: {
                **{k: v for k, v in row.items() if k != "routing"},
                "routing": slim_aggregate(row["routing"]),
            }
            for cond, row in by_condition_summary(results, top_k).items()
        },
        "by_category": {
            category: {
                **{k: v for k, v in row.items() if k != "routing"},
                "routing": slim_aggregate(row["routing"]),
            }
            for category, row in by_category_summary(results, top_k).items()
        },
        "per_prompt": [
            {
                **{k: v for k, v in row.items() if k != "top_experts"},
                "top_experts": {name: slim_routing_summary(track) for name, track in row["top_experts"].items()},
            }
            for row in results
        ],
    }


def render_top_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [f"### {title}", "", "| Rank | Expert | W | S | Q |", "| ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        q_str = "NA" if row["Q"] is None else f"{row['Q']:.6f}"
        lines.append(f"| {row['rank']} | E{row['expert']} | {row['W']:.6f} | {row['S']:.6f} | {q_str} |")
    lines.append("")
    return lines


def render_markdown(output: dict[str, Any]) -> str:
    lines = [
        "# Qwen 122B 5-Condition Suite Results",
        "",
        f"- Model: `{output['model']}`",
        f"- Routing reconstruction: `{output['routing_reconstruction']}`",
        f"- Experts: `{output['n_experts']}` total, top-`{output['n_expert_used']}` selected",
        f"- Layers: `{output['n_moe_layers']}` total, `{len(DELTANET_LAYERS)}` DeltaNet + `{len(SOFTMAX_LAYERS)}` Softmax",
        f"- Layer pattern: `{output['layer_architecture']['pattern']}`",
        f"- Prompts: `{len(output['per_prompt'])}`",
        "",
        "## Condition Means",
        "",
        "| Cond | Label | N | Prefill RE | Prefill LT | Gen RE | Gen LT | Gen Trim RE | Gen Trim LT | Mean gen toks | Mean spill <|im_start|> |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cond in CONDITIONS:
        row = output["by_condition"][cond]
        lines.append(
            f"| `{cond}` | {row['label']} | {row['count']} | {row['prefill_re_mean']:.6f} | "
            f"{row['prefill_last_token_re_mean']:.6f} | {row['generation_re_mean']:.6f} | "
            f"{row['generation_last_token_re_mean']:.6f} | {row['generation_trimmed_re_mean']:.6f} | "
            f"{row['generation_trimmed_last_token_re_mean']:.6f} | {row['mean_generated_tokens']:.1f} | "
            f"{row['mean_spill_im_start']:.2f} |"
        )

    overall = output["overall_routing"]
    lines.extend(["", "## Overall Routing And Expert Selection", ""])
    lines.extend(render_top_table("Prefill Top Experts By W", overall["prefill"]["top_by_W"]))
    lines.extend(render_top_table("Generation Top Experts By W", overall["generation"]["top_by_W"]))
    lines.extend(render_top_table("Generation Top Experts By S", overall["generation"]["top_by_S"]))
    lines.extend(render_top_table("Generation Top Experts By Q", overall["generation"]["top_by_Q"]))
    lines.extend(render_top_table("Generation DeltaNet Top Experts By W", overall["generation_deltanet"]["top_by_W"]))
    lines.extend(render_top_table("Generation Softmax Top Experts By W", overall["generation_softmax"]["top_by_W"]))

    lines.extend(["## Stable-Q Generation-Gaining Candidates", ""])
    lines.append("| Expert | Score | dW | dS | Prefill Q | Gen Q | |dQ| | Gen W rank | Gen S rank | Gen Q rank |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in overall["stable_q_candidates"]:
        lines.append(
            f"| E{row['expert']} | {row['score']:.6f} | {row['delta_W']:+.6f} | {row['delta_S']:+.6f} | "
            f"{row['prefill_Q']:.6f} | {row['generation_Q']:.6f} | {row['abs_delta_Q']:.6f} | "
            f"{row['rank_generation_W']} | {row['rank_generation_S']} | {row['rank_generation_Q']} |"
        )

    lines.extend(["", "## Condition-Level Routing Leaders", ""])
    for cond in CONDITIONS:
        row = output["by_condition"][cond]
        lines.append(f"### `{cond}` {row['label']}")
        lines.append("")
        lines.extend(render_top_table("Generation Top Experts By W", row["routing"]["generation"]["top_by_W"]))

    lines.extend(["## Category Means", "", "| Category | N | Prefill RE | Gen RE | Gen LT | Mean gen toks | Top gen expert |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for category, row in output["by_category"].items():
        top = row["routing"]["generation"]["top_by_W"][0]
        lines.append(
            f"| {category} | {row['count']} | {row['prefill_re_mean']:.6f} | {row['generation_re_mean']:.6f} | "
            f"{row['generation_last_token_re_mean']:.6f} | {row['mean_generated_tokens']:.1f} | "
            f"E{top['expert']} (W={top['W']:.6f}) |"
        )

    lines.extend(["", "## Pairwise Tests", ""])
    for row in output["pairwise_tests"]:
        lines.append(f"### `{row['pair']}`")
        for key, value in row.items():
            if key in {"pair", "c1", "c2"}:
                continue
            p_bits = []
            if "p_raw" in value:
                p_bits.append(f"p_raw={value['p_raw']:.4e}")
            if "p_holm" in value:
                p_bits.append(f"p_holm={value['p_holm']:.4e}")
            p_text = f", {', '.join(p_bits)}" if p_bits else ""
            lines.append(
                f"- {key}: mean_diff={value['mean_diff']:+.6f}, std={value['std_diff']:.6f}, gt={value['gt']}/{value['n']}{p_text}"
            )
        lines.append("")

    lines.extend(["## Run Notes", ""])
    mismatch_count = len(output["token_mismatch_pairs"])
    lines.append(f"- Token-mismatch pairs: `{mismatch_count}`")
    if mismatch_count:
        lines.append("- Generation length is not perfectly matched across all conditions, so use trimmed and last-token metrics alongside all-token RE.")
    lines.append("- This 122B model should be interpreted as a separate regime from 35B because most routed hidden states are DeltaNet-shaped rather than full-softmax-shaped.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    capture_dir = pathlib.Path(args.capture_dir)
    suite = json.loads(pathlib.Path(args.prompt_suite).read_text())
    prompt_texts = load_prompt_texts(pathlib.Path(args.tsv_path))
    cal_paragraph = suite.get("calibration_paragraph")
    prompt_dirs = sorted([d for d in capture_dir.iterdir() if d.is_dir() and (d / "metadata.txt").exists()], key=lambda d: d.name)
    results = [analyze_prompt_dir(prompt_dir, prompt_texts, cal_paragraph, args.top_k) for prompt_dir in prompt_dirs]
    run_metadata = None
    if args.run_metadata and pathlib.Path(args.run_metadata).exists():
        run_metadata = json.loads(pathlib.Path(args.run_metadata).read_text())
    output = build_output(results=results, run_metadata=run_metadata, model_name=args.model_name, top_k=args.top_k)
    pathlib.Path(args.output_json).write_text(json.dumps(output, indent=2) + "\n")
    pathlib.Path(args.output_md).write_text(render_markdown(output))


if __name__ == "__main__":
    main()
