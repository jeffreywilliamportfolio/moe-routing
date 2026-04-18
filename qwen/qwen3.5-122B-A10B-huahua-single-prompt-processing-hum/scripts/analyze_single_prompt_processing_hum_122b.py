#!/usr/bin/env python3
"""Analyze a single 122B processing-hum run with routing and expert-selection detail."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
from typing import Any

import numpy as np


N_LAYERS = 48
SOFTMAX_LAYERS = [layer for layer in range(N_LAYERS) if (layer + 1) % 4 == 0]
DELTANET_LAYERS = [layer for layer in range(N_LAYERS) if layer not in SOFTMAX_LAYERS]
N_EXPERTS = 256
IM_END_TOKEN_SEQUENCE = [27, 91, 316, 6018, 91, 29]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a single 122B processing-hum run.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--prompt-id", default="S01_processing_hum_probe")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--run-metadata", default=None)
    parser.add_argument(
        "--model-name",
        default="Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive Q8_K_P",
    )
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def load_router_helpers() -> tuple[Any, Any, Any, Any]:
    script_dir = pathlib.Path(__file__).resolve().parent
    qwen_router = script_dir / "qwen_router.py"
    spec = importlib.util.spec_from_file_location("qwen_router", qwen_router)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {qwen_router}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TOP_K, module.RECONSTRUCTION_NAME, module.normalized_entropy, module.reconstruct_probs


TOP_K, RECONSTRUCTION_NAME, normalized_entropy, reconstruct_probs = load_router_helpers()


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


def find_im_end_index(token_ids: list[int]) -> int | None:
    seq_len = len(IM_END_TOKEN_SEQUENCE)
    for i in range(len(token_ids) - seq_len + 1):
        if token_ids[i : i + seq_len] == IM_END_TOKEN_SEQUENCE:
            return i
    return None


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


def slim_routing_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {key: value for key, value in summary.items() if key not in {"pooled_W", "pooled_S", "pooled_Q"}}


def analyze_prompt_dir(prompt_dir: pathlib.Path, top_k: int) -> dict[str, Any]:
    prompt_id = prompt_dir.name
    md = parse_metadata(prompt_dir / "metadata.txt")
    n_prompt = int(md.get("n_tokens_prompt", "0"))
    n_gen = int(md.get("n_tokens_generated", "0"))

    generated_tokens = load_generated_tokens(prompt_dir)
    token_ids = [int(tok["token_id"]) for tok in generated_tokens]
    if token_ids:
        n_gen = min(n_gen, len(token_ids))
        token_ids = token_ids[:n_gen]
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
    generation_metrics = compute_track_metrics(gen_logits_by_layer)
    generation_trimmed_metrics = compute_track_metrics(gen_trim_logits_by_layer)

    routing = {
        "prefill": summarize_layers(prefill_layer_rows, top_k),
        "generation": summarize_layers(generation_layer_rows, top_k),
        "generation_trimmed": summarize_layers(generation_trim_layer_rows, top_k),
        "prefill_deltanet": summarize_layers(filter_layer_rows(prefill_layer_rows, set(DELTANET_LAYERS)), top_k),
        "prefill_softmax": summarize_layers(filter_layer_rows(prefill_layer_rows, set(SOFTMAX_LAYERS)), top_k),
        "generation_deltanet": summarize_layers(filter_layer_rows(generation_layer_rows, set(DELTANET_LAYERS)), top_k),
        "generation_softmax": summarize_layers(filter_layer_rows(generation_layer_rows, set(SOFTMAX_LAYERS)), top_k),
    }

    stable = []
    pre = routing["prefill"]
    gen = routing["generation"]
    if pre and gen:
        pW, pS, pQ = pre["pooled_W"], pre["pooled_S"], pre["pooled_Q"]
        gW, gS, gQ = gen["pooled_W"], gen["pooled_S"], gen["pooled_Q"]
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

    return {
        "id": prompt_id,
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
        "top_experts": {name: slim_routing_summary(track) for name, track in routing.items()},
        "stable_q_candidates": stable[:top_k],
    }


def render_top_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [f"### {title}", "", "| Rank | Expert | W | S | Q |", "| ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        q_str = "NA" if row["Q"] is None else f"{row['Q']:.6f}"
        lines.append(f"| {row['rank']} | E{row['expert']} | {row['W']:.6f} | {row['S']:.6f} | {q_str} |")
    lines.append("")
    return lines


def render_markdown(output: dict[str, Any]) -> str:
    prompt = output["prompt"]
    lines = [
        "# Qwen 122B Single Prompt Processing-Hum Results",
        "",
        f"- Model: `{output['model']}`",
        f"- Routing reconstruction: `{output['routing_reconstruction']}`",
        f"- Experts: `{output['n_experts']}` total, top-`{output['n_expert_used']}` selected",
        f"- Layers: `{output['n_moe_layers']}` total, `{len(DELTANET_LAYERS)}` DeltaNet + `{len(SOFTMAX_LAYERS)}` Softmax",
        f"- Prompt id: `{prompt['id']}`",
        "",
        "## Core Metrics",
        "",
        f"- Prompt tokens: `{prompt['n_prompt_tokens']}`",
        f"- Generated tokens: `{prompt['n_generated_tokens']}`",
        f"- Trimmed generated tokens: `{prompt['n_generation_trimmed_tokens']}`",
        f"- Prefill RE: `{prompt['prefill_re']:.6f}`",
        f"- Prefill last-token RE: `{prompt['prefill_last_token_re']:.6f}`",
        f"- Generation RE: `{prompt['generation_re']:.6f}`",
        f"- Generation last-token RE: `{prompt['generation_last_token_re']:.6f}`",
        f"- Generation trimmed RE: `{prompt['generation_trimmed_re']:.6f}`",
        f"- Generation trimmed last-token RE: `{prompt['generation_trimmed_last_token_re']:.6f}`",
        "",
        "## Spill Counts",
        "",
        f"- `<|im_start|>`: `{prompt['generated_text_spill_counts']['<|im_start|>']}`",
        f"- `<|im_end|>`: `{prompt['generated_text_spill_counts']['<|im_end|>']}`",
        f"- `<|endoftext|>`: `{prompt['generated_text_spill_counts']['<|endoftext|>']}`",
        f"- `Thinking Process:`: `{prompt['generated_text_spill_counts']['Thinking Process:']}`",
        "",
        "## Expert Leaders",
        "",
    ]
    lines.extend(render_top_table("Prefill Top Experts By W", prompt["top_experts"]["prefill"]["top_by_W"]))
    lines.extend(render_top_table("Generation Top Experts By W", prompt["top_experts"]["generation"]["top_by_W"]))
    lines.extend(render_top_table("Generation Top Experts By S", prompt["top_experts"]["generation"]["top_by_S"]))
    lines.extend(render_top_table("Generation Top Experts By Q", prompt["top_experts"]["generation"]["top_by_Q"]))
    lines.extend(render_top_table("Generation DeltaNet Top Experts By W", prompt["top_experts"]["generation_deltanet"]["top_by_W"]))
    lines.extend(render_top_table("Generation Softmax Top Experts By W", prompt["top_experts"]["generation_softmax"]["top_by_W"]))
    lines.extend(["## Stable-Q Generation-Gaining Candidates", ""])
    lines.append("| Expert | Score | dW | dS | Prefill Q | Gen Q | |dQ| | Gen W rank | Gen S rank | Gen Q rank |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in prompt["stable_q_candidates"]:
        lines.append(
            f"| E{row['expert']} | {row['score']:.6f} | {row['delta_W']:+.6f} | {row['delta_S']:+.6f} | "
            f"{row['prefill_Q']:.6f} | {row['generation_Q']:.6f} | {row['abs_delta_Q']:.6f} | "
            f"{row['rank_generation_W']} | {row['rank_generation_S']} | {row['rank_generation_Q']} |"
        )
    lines.extend(["", "## Run Notes", ""])
    lines.append("- This is a single-prompt exploratory run, so expert rankings here are prompt-specific rather than condition-comparative.")
    lines.append("- The 122B architecture should be read through the DeltaNet vs softmax split, not as a direct extension of the 35B full-softmax intuition.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    capture_dir = pathlib.Path(args.capture_dir)
    prompt_dir = capture_dir / args.prompt_id
    if not prompt_dir.exists():
        raise SystemExit(f"Prompt directory not found: {prompt_dir}")
    result = analyze_prompt_dir(prompt_dir, args.top_k)
    run_metadata = None
    if args.run_metadata and pathlib.Path(args.run_metadata).exists():
        run_metadata = json.loads(pathlib.Path(args.run_metadata).read_text())
    output = {
        "experiment": "qwen3.5_122b_a10b_huahua_single_prompt_processing_hum",
        "model": args.model_name,
        "routing_reconstruction": RECONSTRUCTION_NAME,
        "n_experts": N_EXPERTS,
        "n_expert_used": TOP_K,
        "n_moe_layers": N_LAYERS,
        "layer_architecture": {
            "deltanet_layers": DELTANET_LAYERS,
            "softmax_layers": SOFTMAX_LAYERS,
            "pattern": "DeltaNet, DeltaNet, DeltaNet, Softmax",
        },
        "run_metadata": run_metadata,
        "prompt": result,
    }
    pathlib.Path(args.output_json).write_text(json.dumps(output, indent=2) + "\n")
    pathlib.Path(args.output_md).write_text(render_markdown(output))


if __name__ == "__main__":
    main()
