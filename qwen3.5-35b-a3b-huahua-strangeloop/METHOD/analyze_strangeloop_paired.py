#!/usr/bin/env python3
"""Analyze HauhauCS Strangeloop paired captures for Cameron-style paired metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except ImportError:
    scipy_wilcoxon = None

from qwen_router import N_EXPERTS, RECONSTRUCTION_NAME, TOP_K, normalized_entropy, reconstruct_probs, softmax_full_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Strangeloop paired capture outputs.")
    parser.add_argument("--capture-dir", required=True, help="Run capture directory containing per-prompt cells.")
    parser.add_argument("--prompt-json", required=True, help="Prompt metadata JSON generated from prompt_suite.json.")
    parser.add_argument("--report", required=True, help="Markdown report output path.")
    parser.add_argument("--results-json", required=True, help="Machine-readable JSON output path.")
    parser.add_argument("--model-label", default="Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0")
    return parser.parse_args()


def parse_metadata(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def estimate_region_boundaries(prompt_text: str, calibration_paragraph: str, n_tokens: int) -> dict[str, tuple[int, int]] | None:
    cal1_start = prompt_text.find(calibration_paragraph)
    if cal1_start < 0:
        return None
    cal1_end = cal1_start + len(calibration_paragraph)
    cal2_start = prompt_text.find(calibration_paragraph, cal1_end)
    if cal2_start < 0:
        return None
    cal2_end = cal2_start + len(calibration_paragraph)
    total_chars = len(prompt_text)
    if total_chars <= 0:
        return None

    def c2t(char_index: int) -> int:
        return max(0, min(n_tokens, int(round(char_index / total_chars * n_tokens))))

    return {
        "cal1": (c2t(cal1_start), c2t(cal1_end)),
        "manip": (c2t(cal1_end), c2t(cal2_start)),
        "cal2": (c2t(cal2_start), c2t(cal2_end)),
    }


def kl_divergence_dense(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return np.sum(p * np.log2((p + 1e-30) / (q + 1e-30)), axis=-1)


def compute_metrics(prompt_dir: Path, n_prompt: int, regions: dict[str, tuple[int, int]] | None) -> dict | None:
    router_dir = prompt_dir / "router"
    if not router_dir.exists() or n_prompt <= 0:
        return None

    files = sorted(router_dir.glob("ffn_moe_logits-*.npy"), key=lambda p: int(p.stem.split("-")[1]))
    if not files:
        return None

    shapes = {int(file_path.stem.split("-")[1]): np.load(file_path).shape[0] for file_path in files}
    median_rows = np.median(list(shapes.values()))
    good_layers = sorted(layer for layer, rows in shapes.items() if rows >= median_rows * 0.5)
    excluded_layers = sorted(set(shapes) - set(good_layers))

    all_ent: list[float] = []
    last_token_ents: list[float] = []
    layer_kl_manip: list[float] = []
    layer_kl_cal2: list[float] = []
    per_layer: list[dict] = []

    for layer_idx in good_layers:
        logits = np.load(router_dir / f"ffn_moe_logits-{layer_idx}.npy")
        n_rows = min(logits.shape[0], n_prompt)
        sparse_probs = reconstruct_probs(logits[:n_rows])
        ent = normalized_entropy(sparse_probs)
        last_ent = float(ent[n_rows - 1])
        last_token_ents.append(last_ent)
        valid = ent > 0
        if valid.any():
            all_ent.extend(ent[valid].tolist())

        layer_info = {
            "layer": layer_idx,
            "mean_entropy": float(np.mean(ent)),
            "std_entropy": float(np.std(ent)),
            "last_token_entropy": last_ent,
            "n_rows": int(logits.shape[0]),
        }

        if regions is not None:
            cal1_s, cal1_e = regions["cal1"]
            manip_s, manip_e = regions["manip"]
            cal2_s, cal2_e = regions["cal2"]

            r_cal1_s = min(cal1_s, n_rows)
            r_cal1_e = min(cal1_e, n_rows)
            r_manip_s = min(manip_s, n_rows)
            r_manip_e = min(manip_e, n_rows)
            r_cal2_s = min(cal2_s, n_rows)
            r_cal2_e = min(cal2_e, n_rows)

            if r_cal1_e > r_cal1_s and r_manip_e > r_manip_s:
                dense_probs = softmax_full_probs(logits[:n_rows])
                cal_baseline = dense_probs[r_cal1_s:r_cal1_e].mean(axis=0)
                cal_baseline = np.clip(cal_baseline, 1e-30, None)

                manip_probs = dense_probs[r_manip_s:r_manip_e]
                kl_manip = np.clip(kl_divergence_dense(manip_probs, cal_baseline[None, :]), 0, None)
                layer_info["kl_manip_mean"] = float(np.mean(kl_manip))
                layer_kl_manip.append(float(np.mean(kl_manip)))

                if r_cal2_e > r_cal2_s:
                    cal2_probs = dense_probs[r_cal2_s:r_cal2_e]
                    kl_cal2 = np.clip(kl_divergence_dense(cal2_probs, cal_baseline[None, :]), 0, None)
                    layer_info["kl_cal2_mean"] = float(np.mean(kl_cal2))
                    layer_kl_cal2.append(float(np.mean(kl_cal2)))

        per_layer.append(layer_info)

    result = {
        "prefill_re": float(np.mean(all_ent)) if all_ent else 0.0,
        "last_token_re": float(np.mean(last_token_ents)) if last_token_ents else 0.0,
        "n_layers": len(good_layers),
        "n_layers_excluded": excluded_layers,
        "n_experts": N_EXPERTS,
        "n_expert_used": TOP_K,
        "reconstruction": RECONSTRUCTION_NAME,
        "per_layer": per_layer,
    }
    if layer_kl_manip:
        result["kl_manip_mean"] = float(np.mean(layer_kl_manip))
    if layer_kl_cal2:
        result["kl_cal2_mean"] = float(np.mean(layer_kl_cal2))
    if regions is not None:
        result["region_boundaries"] = {key: [int(v[0]), int(v[1])] for key, v in regions.items()}
    return result


def summarize_paired(results: list[dict]) -> tuple[dict, list[dict]]:
    by_pair: dict[int, dict[str, dict]] = {}
    for row in results:
        by_pair.setdefault(row["pair"], {})[row["condition"]] = row

    diffs_re: list[float] = []
    diffs_lt: list[float] = []
    diffs_kl: list[float] = []
    pair_rows: list[dict] = []

    a_gt_b_re = 0
    a_gt_b_lt = 0
    a_gt_b_kl = 0

    for pair_num in sorted(by_pair):
        pair = by_pair[pair_num]
        if "A" not in pair or "B" not in pair:
            continue
        row_a = pair["A"]
        row_b = pair["B"]
        diff_re = row_a["prefill_re"] - row_b["prefill_re"]
        diff_lt = row_a["last_token_re"] - row_b["last_token_re"]
        diffs_re.append(diff_re)
        diffs_lt.append(diff_lt)
        if diff_re > 0:
            a_gt_b_re += 1
        if diff_lt > 0:
            a_gt_b_lt += 1

        pair_row = {
            "pair": pair_num,
            "category": row_a["category"],
            "a_tokens": row_a["n_prompt_tokens"],
            "b_tokens": row_b["n_prompt_tokens"],
            "a_re": row_a["prefill_re"],
            "b_re": row_b["prefill_re"],
            "diff_re": diff_re,
            "a_lt": row_a["last_token_re"],
            "b_lt": row_b["last_token_re"],
            "diff_lt": diff_lt,
        }

        if "kl_manip_mean" in row_a and "kl_manip_mean" in row_b:
            diff_kl = row_a["kl_manip_mean"] - row_b["kl_manip_mean"]
            diffs_kl.append(diff_kl)
            if diff_kl > 0:
                a_gt_b_kl += 1
            pair_row["a_kl"] = row_a["kl_manip_mean"]
            pair_row["b_kl"] = row_b["kl_manip_mean"]
            pair_row["diff_kl"] = diff_kl

        pair_rows.append(pair_row)

    summary = {
        "n_pairs": len(pair_rows),
        "all_token_re": {
            "mean_diff": float(np.mean(diffs_re)) if diffs_re else 0.0,
            "std_diff": float(np.std(diffs_re)) if diffs_re else 0.0,
            "a_gt_b": int(a_gt_b_re),
            "n": len(diffs_re),
        },
        "last_token_re": {
            "mean_diff": float(np.mean(diffs_lt)) if diffs_lt else 0.0,
            "std_diff": float(np.std(diffs_lt)) if diffs_lt else 0.0,
            "a_gt_b": int(a_gt_b_lt),
            "n": len(diffs_lt),
        },
    }
    if diffs_kl:
        summary["kl_manip"] = {
            "mean_diff": float(np.mean(diffs_kl)),
            "std_diff": float(np.std(diffs_kl)),
            "a_gt_b": int(a_gt_b_kl),
            "n": len(diffs_kl),
        }

    if len(diffs_re) >= 6 and scipy_wilcoxon is not None:
        w_re, p_re = scipy_wilcoxon(np.array(diffs_re))
        w_lt, p_lt = scipy_wilcoxon(np.array(diffs_lt))
        summary["wilcoxon"] = {
            "all_token_re": {"W": float(w_re), "p": float(p_re)},
            "last_token_re": {"W": float(w_lt), "p": float(p_lt)},
        }
        if diffs_kl:
            w_kl, p_kl = scipy_wilcoxon(np.array(diffs_kl))
            summary["wilcoxon"]["kl_manip"] = {"W": float(w_kl), "p": float(p_kl)}

    return summary, pair_rows


def format_report(results: list[dict], paired_summary: dict, pair_rows: list[dict], model_label: str, capture_dir: Path) -> str:
    lines = []
    lines.append("# Results — qwen3.5-35b-a3b-huahua-strangeloop")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"60-prompt paired routing experiment on **{model_label}** using 30 A/B strange-loop prompt pairs.")
    lines.append("All prompts use a **Cal-Manip-Cal** sandwich and **prefill-only** capture (`n_predict=0`).")
    lines.append("")
    lines.append("This report is targeted at Cameron's 35B definiteness-control ask:")
    lines.append("- Does the A=`this ...` vs B=`a ...` contrast show a paired effect on HauhauCS 35B?")
    lines.append("- Do all-token and last-token metrics tell the same story?")
    lines.append("- Does prompt-local KL-to-Cal1 provide a clearer separation than entropy alone?")
    lines.append("")
    lines.append("## Main Paired Results")
    lines.append("")
    lines.append("| Metric | A-B Mean Diff | Std Diff | A>B | n |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| All-token RE | {paired_summary['all_token_re']['mean_diff']:+.6f} | {paired_summary['all_token_re']['std_diff']:.6f} | {paired_summary['all_token_re']['a_gt_b']}/{paired_summary['all_token_re']['n']} | {paired_summary['all_token_re']['n']} |")
    lines.append(f"| Last-token RE | {paired_summary['last_token_re']['mean_diff']:+.6f} | {paired_summary['last_token_re']['std_diff']:.6f} | {paired_summary['last_token_re']['a_gt_b']}/{paired_summary['last_token_re']['n']} | {paired_summary['last_token_re']['n']} |")
    if "kl_manip" in paired_summary:
        lines.append(f"| KL-to-baseline (manip region) | {paired_summary['kl_manip']['mean_diff']:+.6f} | {paired_summary['kl_manip']['std_diff']:.6f} | {paired_summary['kl_manip']['a_gt_b']}/{paired_summary['kl_manip']['n']} | {paired_summary['kl_manip']['n']} |")
    lines.append("")
    if "wilcoxon" in paired_summary:
        lines.append("## Wilcoxon")
        lines.append("")
        lines.append(f"- All-token RE: W={paired_summary['wilcoxon']['all_token_re']['W']:.0f}, p={paired_summary['wilcoxon']['all_token_re']['p']:.4e}")
        lines.append(f"- Last-token RE: W={paired_summary['wilcoxon']['last_token_re']['W']:.0f}, p={paired_summary['wilcoxon']['last_token_re']['p']:.4e}")
        if "kl_manip" in paired_summary["wilcoxon"]:
            lines.append(f"- KL-to-baseline (manip): W={paired_summary['wilcoxon']['kl_manip']['W']:.0f}, p={paired_summary['wilcoxon']['kl_manip']['p']:.4e}")
        lines.append("")
    lines.append("## Pair Detail")
    lines.append("")
    lines.append("| Pair | Category | A_tok | B_tok | A_RE | B_RE | A-B RE | A_LT | B_LT | A-B LT | A_KL | B_KL | A-B KL |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in pair_rows:
        lines.append(
            f"| {row['pair']} | {row['category']} | {row['a_tokens']} | {row['b_tokens']} | "
            f"{row['a_re']:.6f} | {row['b_re']:.6f} | {row['diff_re']:+.6f} | "
            f"{row['a_lt']:.6f} | {row['b_lt']:.6f} | {row['diff_lt']:+.6f} | "
            f"{row.get('a_kl', float('nan')):.6f} | {row.get('b_kl', float('nan')):.6f} | {row.get('diff_kl', float('nan')):+.6f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- Capture directory: `{capture_dir}`")
    lines.append(f"- Routing reconstruction: `{RECONSTRUCTION_NAME}`")
    lines.append(f"- Entropy normalization: `log2({TOP_K})`")
    lines.append(f"- KL distribution: dense softmax over all `{N_EXPERTS}` experts")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    capture_dir = Path(args.capture_dir)
    prompt_rows = json.loads(Path(args.prompt_json).read_text())
    pair_suite = json.loads((Path(args.prompt_json).parents[1] / "prompt_suite.json").read_text())
    calibration_paragraph = pair_suite["calibration_paragraph"]
    prompt_lookup = {row["id"]: row for row in prompt_rows}

    prompt_dirs = sorted(
        [d for d in capture_dir.iterdir() if d.is_dir() and (d / "metadata.txt").exists()],
        key=lambda d: d.name,
    )

    results: list[dict] = []
    for prompt_dir in prompt_dirs:
        prompt_id = prompt_dir.name
        if prompt_id not in prompt_lookup:
            continue
        metadata = parse_metadata(prompt_dir / "metadata.txt")
        n_prompt = int(metadata.get("n_tokens_prompt", "0"))
        prompt_text = prompt_lookup[prompt_id]["prompt"]
        regions = estimate_region_boundaries(prompt_text, calibration_paragraph, n_prompt)
        metrics = compute_metrics(prompt_dir, n_prompt, regions)
        if metrics is None:
            continue
        row = {
            "id": prompt_id,
            "condition": prompt_lookup[prompt_id]["condition"],
            "pair": int(prompt_lookup[prompt_id]["pair_id"]),
            "category": prompt_lookup[prompt_id]["category"],
            "n_prompt_tokens": n_prompt,
            **metrics,
        }
        results.append(row)

    paired_summary, pair_rows = summarize_paired(results)
    output = {
        "experiment": "qwen_strangeloop_paired_1_hauhau",
        "model": args.model_label,
        "architecture": "qwen35moe",
        "design": "Cal-Manip-Cal sandwich, 30 paired prompts, cold cache",
        "capture_dir": str(capture_dir),
        "n_prompts": len(results),
        "n_pairs": paired_summary["n_pairs"],
        "reconstruction": RECONSTRUCTION_NAME,
        "entropy_normalization": f"log2({TOP_K})",
        "kl_distribution": f"dense softmax over all {N_EXPERTS} experts",
        "paired_test": paired_summary,
        "paired_rows": pair_rows,
        "results": results,
    }

    report_text = format_report(results, paired_summary, pair_rows, args.model_label, capture_dir)
    Path(args.results_json).write_text(json.dumps(output, indent=2) + "\n")
    Path(args.report).write_text(report_text)
    print(f"Wrote {args.results_json}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
