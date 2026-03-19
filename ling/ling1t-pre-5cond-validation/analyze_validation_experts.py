#!/usr/bin/env python3
"""
Expert-identity analysis for preserved Ling-1T validation probes.

Consumes the 6 full router bundles inside the validation artifacts root and
produces:
  1. A machine-readable JSON summary.
  2. A Markdown report focused on which experts were selected.

The script uses exact `ffn_moe_topk-*` + `ffn_moe_weights_norm-*` captures.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from typing import Dict, List

import numpy as np

TOP_K = 8
TARGET_PROMPTS = [
    "P01A_basic_selfref",
    "P99A_deep_selfref_custom",
    "P100A_conscious_probe",
    "P104_mixed_system_probe",
    "P106B_want_answer_probe",
    "P108D_the_system_like_something_probe",
]


def sorted_files(router_dir: pathlib.Path, pattern: str) -> List[pathlib.Path]:
    return sorted(
        router_dir.glob(pattern),
        key=lambda path: int(path.stem.split("-")[1]),
    )


def load_prompt_map(results_json: pathlib.Path) -> Dict[str, dict]:
    obj = json.loads(results_json.read_text())
    return {
        row["prompt_id"]: row
        for row in obj["full_metric_prompts"]
        if row["prompt_id"] in TARGET_PROMPTS
    }


def find_router_dir(artifacts_root: pathlib.Path, prompt_id: str) -> pathlib.Path:
    matches = list(artifacts_root.rglob(f"{prompt_id}/router"))
    if not matches:
        raise FileNotFoundError(f"router dir not found for {prompt_id}")
    matches.sort(key=lambda path: len(str(path)))
    return matches[0]


def init_region_stats() -> Dict[str, object]:
    return {
        "selected_count": Counter(),
        "weight_sum": Counter(),
        "token_count": 0,
        "token_layer_count": 0,
    }


def finalize_region(stats: Dict[str, object]) -> Dict[str, object]:
    token_layer_count = max(int(stats["token_layer_count"]), 1)
    token_count = max(int(stats["token_count"]), 1)
    selected_count: Counter = stats["selected_count"]
    weight_sum: Counter = stats["weight_sum"]

    top_by_selection = [
        {
            "expert": int(expert),
            "selected_count": int(count),
            "selection_rate": float(count / token_layer_count),
            "mean_weight_per_selected": float(weight_sum[expert] / count),
            "mean_weight_per_token_layer": float(weight_sum[expert] / token_layer_count),
        }
        for expert, count in selected_count.most_common(15)
    ]

    top_by_weight = [
        {
            "expert": int(expert),
            "weight_sum": float(weight),
            "mean_weight_per_token_layer": float(weight / token_layer_count),
            "selection_rate": float(selected_count[expert] / token_layer_count),
        }
        for expert, weight in weight_sum.most_common(15)
    ]

    return {
        "token_count": token_count,
        "token_layer_count": token_layer_count,
        "n_unique_experts": len(selected_count),
        "top_by_selection": top_by_selection,
        "top_by_weight": top_by_weight,
        "selected_count": {str(k): int(v) for k, v in selected_count.items()},
        "weight_sum": {str(k): float(v) for k, v in weight_sum.items()},
    }


def analyze_prompt(prompt_row: dict, router_dir: pathlib.Path) -> Dict[str, object]:
    cal1_s, cal1_e = prompt_row["region_boundaries"]["cal1"]
    manip_s, manip_e = prompt_row["region_boundaries"]["manip"]
    cal2_s, cal2_e = prompt_row["region_boundaries"]["cal2"]

    regions = {
        "full_prompt": init_region_stats(),
        "cal1": init_region_stats(),
        "manip": init_region_stats(),
        "cal2": init_region_stats(),
    }

    weights_files = sorted_files(router_dir, "ffn_moe_weights_norm-*.npy")
    topk_files = sorted_files(router_dir, "ffn_moe_topk-*.npy")
    topk_map = {int(path.stem.split("-")[1]): path for path in topk_files}

    layers = []
    manip_only = Counter()
    manip_weight = Counter()
    cal1_only = Counter()
    cal1_weight = Counter()

    for weights_path in weights_files:
        layer = int(weights_path.stem.split("-")[1])
        if layer == 79 or layer not in topk_map:
            continue

        weights = np.load(weights_path)
        topk = np.load(topk_map[layer])
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        if topk.ndim == 1:
            topk = topk.reshape(1, -1)
        n_rows = min(weights.shape[0], topk.shape[0], prompt_row["prompt_tokens"])
        if n_rows <= 1:
            continue

        layers.append(layer)
        weights = weights[:n_rows]
        topk = topk[:n_rows]

        slices = {
            "full_prompt": slice(0, n_rows),
            "cal1": slice(cal1_s, min(cal1_e, n_rows)),
            "manip": slice(manip_s, min(manip_e, n_rows)),
            "cal2": slice(cal2_s, min(cal2_e, n_rows)),
        }

        for region_name, region_slice in slices.items():
            region_topk = topk[region_slice]
            region_weights = weights[region_slice]
            if region_topk.size == 0:
                continue
            stats = regions[region_name]
            stats["token_count"] += int(region_topk.shape[0])
            stats["token_layer_count"] += int(region_topk.shape[0] * region_topk.shape[1])
            for row_experts, row_weights in zip(region_topk, region_weights):
                for expert, weight in zip(row_experts.tolist(), row_weights.tolist()):
                    stats["selected_count"][int(expert)] += 1
                    stats["weight_sum"][int(expert)] += float(weight)

        manip_topk = topk[manip_s:min(manip_e, n_rows)]
        manip_weights = weights[manip_s:min(manip_e, n_rows)]
        for row_experts, row_weights in zip(manip_topk, manip_weights):
            for expert, weight in zip(row_experts.tolist(), row_weights.tolist()):
                manip_only[int(expert)] += 1
                manip_weight[int(expert)] += float(weight)

        cal1_topk = topk[cal1_s:min(cal1_e, n_rows)]
        cal1_weights = weights[cal1_s:min(cal1_e, n_rows)]
        for row_experts, row_weights in zip(cal1_topk, cal1_weights):
            for expert, weight in zip(row_experts.tolist(), row_weights.tolist()):
                cal1_only[int(expert)] += 1
                cal1_weight[int(expert)] += float(weight)

    manip_token_layer_count = max(int(regions["manip"]["token_layer_count"]), 1)
    cal1_token_layer_count = max(int(regions["cal1"]["token_layer_count"]), 1)
    all_experts = set(manip_only) | set(cal1_only)
    manip_vs_cal1 = sorted(
        (
            {
                "expert": int(expert),
                "selection_rate_diff": float(
                    manip_only[expert] / manip_token_layer_count
                    - cal1_only[expert] / cal1_token_layer_count
                ),
                "manip_selection_rate": float(manip_only[expert] / manip_token_layer_count),
                "cal1_selection_rate": float(cal1_only[expert] / cal1_token_layer_count),
                "manip_weight_per_token_layer": float(manip_weight[expert] / manip_token_layer_count),
                "cal1_weight_per_token_layer": float(cal1_weight[expert] / cal1_token_layer_count),
            }
            for expert in all_experts
        ),
        key=lambda row: row["selection_rate_diff"],
        reverse=True,
    )

    finalized = {name: finalize_region(stats) for name, stats in regions.items()}
    finalized["manip_vs_cal1_top_positive"] = manip_vs_cal1[:15]
    finalized["manip_vs_cal1_top_negative"] = manip_vs_cal1[-15:]

    return {
        "prompt_id": prompt_row["prompt_id"],
        "short_label": prompt_row["short_label"],
        "source_bundle": prompt_row["source_bundle"],
        "prompt_tokens": prompt_row["prompt_tokens"],
        "segment_token_counts": prompt_row["segment_token_counts"],
        "layers_analyzed": layers,
        "regions": finalized,
    }


def build_cross_prompt_summary(prompt_results: List[dict]) -> Dict[str, object]:
    top_positive = []
    expert_probe_presence: Counter = Counter()
    expert_probe_top_rank: Counter = Counter()

    for row in prompt_results:
        positives = row["regions"]["manip_vs_cal1_top_positive"][:10]
        top_positive.append(
            {
                "prompt_id": row["prompt_id"],
                "top_positive": positives,
            }
        )
        seen = set()
        for item in positives:
            expert = item["expert"]
            seen.add(expert)
        for expert in seen:
            expert_probe_presence[expert] += 1
        for item in positives[:3]:
            expert_probe_top_rank[item["expert"]] += 1

    recurrent = sorted(
        (
            {
                "expert": int(expert),
                "n_prompts_in_top10": int(expert_probe_presence[expert]),
                "n_prompts_in_top3": int(expert_probe_top_rank[expert]),
            }
            for expert in expert_probe_presence
        ),
        key=lambda row: (row["n_prompts_in_top10"], row["n_prompts_in_top3"], -row["expert"]),
        reverse=True,
    )

    return {
        "per_prompt_top_positive": top_positive,
        "recurrent_positive_experts": recurrent[:20],
    }


def render_markdown(results: Dict[str, object]) -> str:
    lines = []
    lines.append("# Ling-1T Validation Expert Activation Report")
    lines.append("")
    lines.append(
        f"Generated from `{results['artifacts_root']}` for the 6 validation probes with "
        "full exact router tensors."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Prompts analyzed: `{len(results['prompt_results'])}`")
    lines.append("- Exact tensors used: `ffn_moe_topk-*` + `ffn_moe_weights_norm-*`")
    lines.append("- Expert universe: `256` experts, `top-8` per token per layer")
    lines.append("- Layer `79` excluded, matching the validation RE/KL report")
    lines.append("")
    lines.append("## Recurrent Manipulation-Boosted Experts")
    lines.append("")
    lines.append("| Expert | Prompts in Top-10 | Prompts in Top-3 |")
    lines.append("|---|---:|---:|")
    for row in results["cross_prompt_summary"]["recurrent_positive_experts"]:
        lines.append(
            f"| `E{row['expert']}` | {row['n_prompts_in_top10']} | {row['n_prompts_in_top3']} |"
        )

    for prompt in results["prompt_results"]:
        lines.append("")
        lines.append(f"## `{prompt['prompt_id']}`")
        lines.append("")
        lines.append(f"- Short label: `{prompt['short_label']}`")
        lines.append(f"- Source bundle: `{prompt['source_bundle']}`")
        lines.append(f"- Prompt tokens: `{prompt['prompt_tokens']}`")
        lines.append(
            "- Segment tokens: "
            f"Cal1 `{prompt['segment_token_counts']['cal1_tokens']}`, "
            f"Manip `{prompt['segment_token_counts']['manip_tokens']}`, "
            f"Cal2 `{prompt['segment_token_counts']['cal2_tokens']}`"
        )
        lines.append("")
        lines.append("### Top Manipulation-Selected Experts")
        lines.append("")
        lines.append("| Expert | Selection Rate | Mean Weight / Token-Layer | Mean Weight / Selected |")
        lines.append("|---|---:|---:|---:|")
        for row in prompt["regions"]["manip"]["top_by_selection"][:10]:
            lines.append(
                f"| `E{row['expert']}` | {row['selection_rate']:.6f} | "
                f"{row['mean_weight_per_token_layer']:.6f} | {row['mean_weight_per_selected']:.6f} |"
            )
        lines.append("")
        lines.append("### Strongest Manipulation vs Cal1 Selection Shifts")
        lines.append("")
        lines.append("| Expert | Manip Rate | Cal1 Rate | Diff |")
        lines.append("|---|---:|---:|---:|")
        for row in prompt["regions"]["manip_vs_cal1_top_positive"][:10]:
            lines.append(
                f"| `E{row['expert']}` | {row['manip_selection_rate']:.6f} | "
                f"{row['cal1_selection_rate']:.6f} | {row['selection_rate_diff']:+.6f} |"
            )
        lines.append("")
        lines.append("### Strongest Cal1 over Manip Selection Shifts")
        lines.append("")
        lines.append("| Expert | Manip Rate | Cal1 Rate | Diff |")
        lines.append("|---|---:|---:|---:|")
        for row in reversed(prompt["regions"]["manip_vs_cal1_top_negative"][-10:]):
            lines.append(
                f"| `E{row['expert']}` | {row['manip_selection_rate']:.6f} | "
                f"{row['cal1_selection_rate']:.6f} | {row['selection_rate_diff']:+.6f} |"
            )

    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- This is an expert-identity analysis for the 6 validation probes only, not the failed 150-prompt run.")
    lines.append("- The 6 metadata-only validation probes cannot be analyzed at expert level because they do not retain router tensors.")
    lines.append("- Selection-rate comparisons are normalized by token-layer opportunities within each region.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", required=True)
    parser.add_argument("--results-json", required=True)
    parser.add_argument(
        "--output-json",
        default="validation_expert_activation_results.json",
    )
    parser.add_argument(
        "--output-md",
        default="RESULTS-EXPERTS-VALIDATION.md",
    )
    args = parser.parse_args()

    artifacts_root = pathlib.Path(args.artifacts_root)
    prompt_map = load_prompt_map(pathlib.Path(args.results_json))

    prompt_results = []
    for prompt_id in TARGET_PROMPTS:
        if prompt_id not in prompt_map:
            raise KeyError(f"Prompt missing from results JSON: {prompt_id}")
        router_dir = find_router_dir(artifacts_root, prompt_id)
        prompt_results.append(analyze_prompt(prompt_map[prompt_id], router_dir))

    output = {
        "artifacts_root": str(artifacts_root),
        "prompt_ids": TARGET_PROMPTS,
        "prompt_results": prompt_results,
        "cross_prompt_summary": build_cross_prompt_summary(prompt_results),
    }

    output_json = pathlib.Path(args.output_json)
    output_md = pathlib.Path(args.output_md)
    output_json.write_text(json.dumps(output, indent=2))
    output_md.write_text(render_markdown(output))

    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
