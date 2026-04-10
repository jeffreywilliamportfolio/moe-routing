#!/usr/bin/env python3
"""
Analyze single-prompt HauhauCS capture runs with full-token layer decomposition.

This localized copy keeps the Strangeloop bundle self-contained.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
QWEN_ROUTER = SCRIPT_DIR / "qwen_router.py"
TARGET_EXPERT = 114


def load_reconstruct_probs():
    spec = importlib.util.spec_from_file_location("qwen_router", QWEN_ROUTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load qwen_router from {QWEN_ROUTER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reconstruct_probs


reconstruct_probs = load_reconstruct_probs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze single-prompt capture runs.")
    parser.add_argument(
        "--capture-root",
        required=True,
        help="Directory containing run capture subdirectories.",
    )
    parser.add_argument(
        "--run-pattern",
        default="*single*",
        help="Glob pattern for run directories inside capture-root.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory where per-run JSON/MD summaries should be written.",
    )
    parser.add_argument(
        "--model-label",
        default="HauhauCS Qwen3.5-35B-A3B Q8_0",
        help="Label recorded in output.",
    )
    return parser.parse_args()


def parse_metadata(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def compute_layer_track_metrics(
    logits: np.ndarray,
    n_prompt: int,
    n_gen: int,
    layer_idx: int,
) -> dict[str, Any]:
    total_rows = int(logits.shape[0])
    layer_info: dict[str, Any] = {
        "layer": layer_idx,
        "total_rows": total_rows,
    }

    prefill_metrics = None
    generation_metrics = None
    generation_rows_used = 0

    if total_rows >= n_prompt:
        prefill_logits = logits[:n_prompt]
        if prefill_logits.size > 0:
            prefill_probs = reconstruct_probs(prefill_logits)
            prefill_w = prefill_probs[:, TARGET_EXPERT]
            prefill_s = (prefill_w > 0).astype(np.float64)
            prefill_metrics = {
                "W": float(prefill_w.mean()),
                "S": float(prefill_s.mean()),
                "Q": float(prefill_w[prefill_w > 0].mean()) if np.any(prefill_w > 0) else 0.0,
                "selected_tokens": int(prefill_s.sum()),
                "n_tokens": int(n_prompt),
            }

    if total_rows == n_prompt + n_gen:
        gen_logits = logits[n_prompt:]
    elif total_rows == n_gen + 1:
        gen_logits = logits[1:]
    elif total_rows == n_gen:
        gen_logits = logits
    elif total_rows > n_prompt:
        gen_logits = logits[n_prompt:]
    else:
        gen_logits = np.empty((0, logits.shape[1]), dtype=logits.dtype)

    generation_rows_used = int(gen_logits.shape[0])
    if gen_logits.size > 0:
        gen_probs = reconstruct_probs(gen_logits)
        gen_w = gen_probs[:, TARGET_EXPERT]
        gen_s = (gen_w > 0).astype(np.float64)
        generation_metrics = {
            "W": float(gen_w.mean()),
            "S": float(gen_s.mean()),
            "Q": float(gen_w[gen_w > 0].mean()) if np.any(gen_w > 0) else 0.0,
            "selected_tokens": int(gen_s.sum()),
            "n_tokens": int(gen_logits.shape[0]),
        }

    layer_info["generation_rows_used"] = generation_rows_used
    layer_info["prefill"] = prefill_metrics
    layer_info["generation"] = generation_metrics
    return layer_info


def pooled_track(layers: list[dict[str, Any]], track: str) -> dict[str, float] | None:
    rows = [layer[track] for layer in layers if layer.get(track) is not None]
    if not rows:
        return None
    return {
        "W_mean": float(np.mean([row["W"] for row in rows])),
        "S_mean": float(np.mean([row["S"] for row in rows])),
        "Q_mean": float(np.mean([row["Q"] for row in rows])),
    }


def best_layer(layers: list[dict[str, Any]], track: str) -> dict[str, Any] | None:
    rows = [layer for layer in layers if layer.get(track) is not None]
    if not rows:
        return None
    best = max(rows, key=lambda layer: layer[track]["W"])
    return {
        "layer": best["layer"],
        **best[track],
    }


def top_layers(layers: list[dict[str, Any]], track: str, limit: int = 8) -> list[dict[str, Any]]:
    rows = [layer for layer in layers if layer.get(track) is not None]
    rows.sort(key=lambda layer: layer[track]["W"], reverse=True)
    return [
        {
            "layer": layer["layer"],
            "W": layer[track]["W"],
            "S": layer[track]["S"],
            "Q": layer[track]["Q"],
        }
        for layer in rows[:limit]
    ]


def identity_residual(layers: list[dict[str, Any]]) -> float:
    residuals = []
    for layer in layers:
        for track in ("prefill", "generation"):
            row = layer.get(track)
            if row is None:
                continue
            residuals.append(abs(row["W"] - (row["S"] * row["Q"])))
    return float(max(residuals)) if residuals else 0.0


def spill_counts(generated_text: str) -> dict[str, int]:
    return {
        "<|im_start|>": generated_text.count("<|im_start|>"),
        "<|im_end|>": generated_text.count("<|im_end|>"),
        "<|endoftext|>": generated_text.count("<|endoftext|>"),
        "Thinking Process:": generated_text.count("Thinking Process:"),
    }


def analyze_run(run_dir: Path, model_label: str) -> dict[str, Any]:
    cells = [p for p in run_dir.iterdir() if p.is_dir()]
    if len(cells) != 1:
        raise ValueError(f"{run_dir}: expected exactly one prompt cell, found {len(cells)}")
    cell_dir = cells[0]
    metadata = parse_metadata(cell_dir / "metadata.txt")
    n_prompt = int(metadata["n_tokens_prompt"])
    n_gen = int(metadata["n_tokens_generated"])
    generated_text = (cell_dir / "generated_text.txt").read_text(errors="replace")

    layers = []
    for layer_idx in range(40):
        logits_path = cell_dir / "router" / f"ffn_moe_logits-{layer_idx}.npy"
        logits = np.load(logits_path)
        layers.append(compute_layer_track_metrics(logits, n_prompt, n_gen, layer_idx))

    return {
        "run_id": run_dir.name,
        "prompt_id": metadata.get("prompt_id", cell_dir.name),
        "model_label": model_label,
        "capture_dir": str(cell_dir),
        "n_tokens_prompt": n_prompt,
        "n_tokens_generated": n_gen,
        "elapsed_ms": int(metadata.get("elapsed_ms", "0")),
        "router_tensors": int(metadata.get("n_router_tensors", "0")),
        "generated_text_spill_counts": spill_counts(generated_text),
        "identity_residual": identity_residual(layers),
        "pooled": {
            "prefill": pooled_track(layers, "prefill"),
            "generation": pooled_track(layers, "generation"),
        },
        "best_layer": {
            "prefill": best_layer(layers, "prefill"),
            "generation": best_layer(layers, "generation"),
        },
        "top_layers_prefill": top_layers(layers, "prefill"),
        "top_layers_generation": top_layers(layers, "generation"),
        "layers": layers,
    }


def render_md(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Single Prompt Analysis: {result['run_id']}\n\n")
    lines.append(f"Prompt ID: `{result['prompt_id']}`\n\n")
    lines.append(f"Model: {result['model_label']}\n\n")
    lines.append(f"Capture dir: `{result['capture_dir']}`\n\n")
    lines.append("## Capture\n\n")
    lines.append(f"- Prompt tokens: `{result['n_tokens_prompt']}`\n")
    lines.append(f"- Generated tokens: `{result['n_tokens_generated']}`\n")
    lines.append(f"- Elapsed capture time: `{result['elapsed_ms']} ms`\n")
    lines.append(f"- Router tensors captured: `{result['router_tensors']}`\n")
    lines.append(f"- `W = S x Q` max residual: `{result['identity_residual']:.3e}`\n\n")

    lines.append("## Spill Counts\n\n")
    for key, value in result["generated_text_spill_counts"].items():
        lines.append(f"- `{key}`: `{value}`\n")
    lines.append("\n")

    lines.append("## Pooled Decomposition\n\n")
    lines.append("| Track | mean W_114 | mean S_114 | mean Q_114 |\n")
    lines.append("|---|---:|---:|---:|\n")
    for track in ("prefill", "generation"):
        row = result["pooled"][track]
        if row is None:
            continue
        lines.append(
            f"| {track} | {row['W_mean']:.6f} | {row['S_mean']:.6f} | {row['Q_mean']:.6f} |\n"
        )
    lines.append("\n")

    lines.append("## Best Layer By W_114\n\n")
    for track in ("prefill", "generation"):
        row = result["best_layer"][track]
        if row is None:
            continue
        lines.append(
            f"- {track}: layer `{row['layer']}` with "
            f"`W={row['W']:.6f}`, `S={row['S']:.6f}`, `Q={row['Q']:.6f}`\n"
        )
    lines.append("\n")

    lines.append("## Top Layers Prefill\n\n")
    lines.append("| Layer | W_114 | S_114 | Q_114 |\n")
    lines.append("|---:|---:|---:|---:|\n")
    for row in result["top_layers_prefill"]:
        lines.append(f"| {row['layer']} | {row['W']:.6f} | {row['S']:.6f} | {row['Q']:.6f} |\n")
    lines.append("\n")

    lines.append("## Top Layers Generation\n\n")
    lines.append("| Layer | W_114 | S_114 | Q_114 |\n")
    lines.append("|---:|---:|---:|---:|\n")
    for row in result["top_layers_generation"]:
        lines.append(f"| {row['layer']} | {row['W']:.6f} | {row['S']:.6f} | {row['Q']:.6f} |\n")
    lines.append("\n")
    return "".join(lines)


def main() -> None:
    args = parse_args()
    capture_root = Path(args.capture_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in capture_root.glob(args.run_pattern) if p.is_dir()])
    if not run_dirs:
        raise SystemExit(f"No run directories matched {args.run_pattern!r} in {capture_root}")

    for run_dir in run_dirs:
        result = analyze_run(run_dir, args.model_label)
        stem = f"results_{run_dir.name}"
        json_path = results_dir / f"{stem}.json"
        md_path = results_dir / f"{stem}.md"
        json_path.write_text(json.dumps(result, indent=2))
        md_path.write_text(render_md(result))
        print(json_path)
        print(md_path)


if __name__ == "__main__":
    main()
