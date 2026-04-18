#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qwen_router import reconstruct_probs


def parse_meta(path: pathlib.Path) -> dict[str, int]:
    data: dict[str, int] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            data[key] = int(value)
        except ValueError:
            continue
    return data


def infer_group(prompt_id: str) -> str:
    if "_" in prompt_id:
        return prompt_id.split("_", 1)[1]
    return prompt_id


def sorted_router_files(router_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        router_dir.glob("ffn_moe_logits-*.npy"),
        key=lambda path: int(path.stem.split("-")[1]),
    )


def aggregate_group(
    run_dir: pathlib.Path,
    condition: str,
    group: str,
    focus_expert: int = 114,
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    capture_dir = run_dir / "capture" / condition
    for prompt_dir in sorted(capture_dir.iterdir()):
        if not prompt_dir.is_dir() or not (prompt_dir / "metadata.txt").exists():
            continue
        if infer_group(prompt_dir.name) != group:
            continue
        meta = parse_meta(prompt_dir / "metadata.txt")
        start = meta.get("n_tokens_prompt", 0)
        n_generated = meta.get("n_tokens_generated", 0)
        for router_path in sorted_router_files(prompt_dir / "router"):
            layer = int(router_path.stem.split("-")[1])
            logits = np.load(router_path)
            begin = min(start, logits.shape[0])
            end = min(start + n_generated, logits.shape[0])
            if end <= begin:
                continue
            probs = reconstruct_probs(logits[begin:end])
            selected = probs > 0
            layer_totals = out.setdefault(
                layer,
                {"selection_count": 0.0, "weight_sum": 0.0, "total_slots": 0.0, "total_weight": 0.0},
            )
            layer_totals["selection_count"] += float(selected[:, focus_expert].sum())
            layer_totals["weight_sum"] += float(probs[:, focus_expert].sum())
            layer_totals["total_slots"] += float(selected.sum())
            layer_totals["total_weight"] += float(probs.sum())
    return out


def compute_rows(
    run_dir: pathlib.Path,
    group: str,
    condition: str = "expert_114_soft_bias_1.0",
    baseline: str = "baseline",
) -> list[dict[str, float]]:
    base = aggregate_group(run_dir, baseline, group)
    cond = aggregate_group(run_dir, condition, group)
    rows: list[dict[str, float]] = []
    for layer in sorted(set(base) & set(cond)):
        base_totals = base[layer]
        cond_totals = cond[layer]
        base_weight_rate = base_totals["weight_sum"] / base_totals["total_weight"]
        cond_weight_rate = cond_totals["weight_sum"] / cond_totals["total_weight"]
        base_selection_rate = base_totals["selection_count"] / base_totals["total_slots"]
        cond_selection_rate = cond_totals["selection_count"] / cond_totals["total_slots"]
        rows.append(
            {
                "layer": layer,
                "weight_rate_delta": cond_weight_rate - base_weight_rate,
                "selection_rate_delta": cond_selection_rate - base_selection_rate,
            }
        )
    return rows


def window_mean(rows: list[dict[str, float]], start: int, end: int) -> float:
    values = [row["weight_rate_delta"] for row in rows if start <= row["layer"] <= end]
    return float(np.mean(values))


def summarize_top(rows: list[dict[str, float]], limit: int = 5) -> list[tuple[int, float]]:
    top_rows = sorted(rows, key=lambda row: (-row["weight_rate_delta"], row["layer"]))[:limit]
    return [(row["layer"], row["weight_rate_delta"]) for row in top_rows]


def verify_saved_json(
    json_path: pathlib.Path,
    run_dir: pathlib.Path,
    groups: list[str],
    condition: str = "expert_114_soft_bias_1.0",
) -> dict[str, object]:
    saved = json.loads(json_path.read_text())
    out: dict[str, object] = {}
    for group in groups:
        rows = compute_rows(run_dir, group, condition=condition)
        saved_rows = saved["group_comparisons"][group]["per_layer"]
        max_abs_diff = max(
            abs(lhs["weight_rate_delta"] - rhs["weight_rate_delta"])
            for lhs, rhs in zip(rows, saved_rows)
        )
        out[group] = {
            "max_abs_weight_delta_diff": max_abs_diff,
            "windows": {
                "early_0_14": window_mean(rows, 0, 14),
                "middle_15_25": window_mean(rows, 15, 25),
                "late_26_39": window_mean(rows, 26, 39),
            },
            "top5": summarize_top(rows, limit=5),
        }
    return out


def main() -> None:
    smoke_run = HERE / "runs" / "smoke-20260323b"
    smoke_json = HERE / "reanalysis" / "20260327-focus-layers" / "smoke-expert_114_soft_bias_1.0.json"
    smoke_summary = verify_saved_json(
        smoke_json,
        smoke_run,
        groups=["process", "regulation", "static_fact"],
    )
    print("SMOKE")
    print(json.dumps(smoke_summary, indent=2))

    five_run = HERE / "runs" / "nothink-5cond-boost-1024-20260323"
    for level in ("0.25", "0.5", "1.0"):
        condition = f"expert_114_soft_bias_{level}"
        five_json = HERE / "reanalysis" / "20260327-focus-layers" / f"5cond-{condition}.json"
        five_summary = verify_saved_json(
            five_json,
            five_run,
            groups=["experience_probe", "recursive_selfref", "routing_selfref"],
            condition=condition,
        )
        print(f"5COND {condition}")
        print(json.dumps(five_summary, indent=2))


if __name__ == "__main__":
    main()
