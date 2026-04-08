#!/usr/bin/env python3
"""Analyze generated-token routing and rubric drift for the HauhauCS steering run."""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

from qwen_router import N_EXPERTS, js_divergence, probability_from_counts, reconstruct_probs

_HERE = pathlib.Path(__file__).parent

BAND_ALIASES = {
    "static_fact_control": "static_fact",
    "process_explanation": "process",
    "implicatio_probe": "regulation",
}


def parse_metadata(path: pathlib.Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value
    return meta


def load_prompt_suite(path: pathlib.Path) -> dict:
    suite = json.loads(path.read_text())
    if "prompts" in suite:
        prompts = {prompt["id"]: prompt for prompt in suite["prompts"]}
        return {"raw": suite, "prompts": prompts}

    prompts = {}
    for band in suite.get("bands", []):
        source_band = band["name"]
        normalized_band = BAND_ALIASES.get(source_band, source_band)
        for prompt in band.get("prompts", []):
            prompts[prompt["id"]] = {
                "id": prompt["id"],
                "band": normalized_band,
                "source_band": source_band,
                "text": prompt["text"],
            }
    return {"raw": suite, "prompts": prompts}


def load_rubric(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def score_rubric(text: str, rubric: dict) -> dict:
    lowered = text.lower()
    categories = {}
    total = 0.0
    for name, cfg in rubric["categories"].items():
        matched = [pattern for pattern in cfg["patterns"] if pattern in lowered]
        score = float(cfg["weight"]) * len(matched)
        categories[name] = {
            "score": score,
            "matched_patterns": matched,
        }
        total += score
    return {
        "total_score": total,
        "categories": categories,
    }


def sorted_router_files(router_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        router_dir.glob("ffn_moe_logits-*.npy"),
        key=lambda path: int(path.stem.split("-")[1]),
    )


def analyze_prompt_dir(prompt_dir: pathlib.Path, band: str, focus_expert: int, rubric: dict) -> dict:
    metadata = parse_metadata(prompt_dir / "metadata.txt")
    n_prompt = int(metadata["n_tokens_prompt"])
    n_generated = int(metadata.get("n_tokens_generated", 0))
    generated_text_path = prompt_dir / "generated_text.txt"
    generated_text = generated_text_path.read_text() if generated_text_path.exists() else ""

    counts = np.zeros(N_EXPERTS, dtype=np.float64)
    weights = np.zeros(N_EXPERTS, dtype=np.float64)
    cooccurrence = np.zeros(N_EXPERTS, dtype=np.float64)
    used_layers = []

    router_dir = prompt_dir / "router"
    if router_dir.exists() and n_generated > 0:
        for path in sorted_router_files(router_dir):
            logits = np.load(path)
            start = min(n_prompt, logits.shape[0])
            end = min(n_prompt + n_generated, logits.shape[0])
            if end <= start:
                continue
            probs = reconstruct_probs(logits[start:end])
            selected = probs > 0
            counts += selected.sum(axis=0)
            weights += probs.sum(axis=0)
            if 0 <= focus_expert < N_EXPERTS:
                active_rows = selected[:, focus_expert]
                if np.any(active_rows):
                    cooccurrence += selected[active_rows].sum(axis=0)
            used_layers.append(int(path.stem.split("-")[1]))

    if 0 <= focus_expert < N_EXPERTS:
        cooccurrence[focus_expert] = 0.0

    rubric_score = score_rubric(generated_text, rubric)
    total_slots = float(counts.sum())
    total_weight = float(weights.sum())

    return {
        "prompt_id": prompt_dir.name,
        "band": band,
        "n_prompt_tokens": n_prompt,
        "n_generated_tokens": n_generated,
        "generated_text": generated_text,
        "rubric": rubric_score,
        "generated_selection_counts": counts.astype(float).tolist(),
        "generated_weight_sums": weights.astype(float).tolist(),
        "focus_expert_selection_rate": float(counts[focus_expert] / total_slots) if total_slots > 0 else 0.0,
        "focus_expert_weight_rate": float(weights[focus_expert] / total_weight) if total_weight > 0 else 0.0,
        "focus_expert_cooccurrence_counts": cooccurrence.astype(float).tolist(),
        "used_router_layers": used_layers,
    }


def aggregate_band(prompt_rows: list[dict]) -> dict:
    counts = np.sum([np.asarray(row["generated_selection_counts"], dtype=np.float64) for row in prompt_rows], axis=0)
    weights = np.sum([np.asarray(row["generated_weight_sums"], dtype=np.float64) for row in prompt_rows], axis=0)
    return {
        "counts": counts,
        "weights": weights,
    }


def compare_to_baseline(condition_rows: list[dict], baseline_rows: list[dict], focus_expert: int) -> dict:
    baseline_by_prompt = {row["prompt_id"]: row for row in baseline_rows}
    bands = sorted({row["band"] for row in condition_rows})

    prompt_level = []
    for row in condition_rows:
        base = baseline_by_prompt[row["prompt_id"]]
        prompt_level.append({
            "prompt_id": row["prompt_id"],
            "band": row["band"],
            "rubric_total": row["rubric"]["total_score"],
            "baseline_rubric_total": base["rubric"]["total_score"],
            "rubric_delta": row["rubric"]["total_score"] - base["rubric"]["total_score"],
            "focus_expert_selection_rate_delta": row["focus_expert_selection_rate"] - base["focus_expert_selection_rate"],
            "focus_expert_weight_rate_delta": row["focus_expert_weight_rate"] - base["focus_expert_weight_rate"],
        })

    band_level = {}
    for band in bands:
        cond_band_rows = [row for row in condition_rows if row["band"] == band]
        base_band_rows = [row for row in baseline_rows if row["band"] == band]
        cond_agg = aggregate_band(cond_band_rows)
        base_agg = aggregate_band(base_band_rows)

        cond_co = np.sum([np.asarray(row["focus_expert_cooccurrence_counts"], dtype=np.float64) for row in cond_band_rows], axis=0)
        base_co = np.sum([np.asarray(row["focus_expert_cooccurrence_counts"], dtype=np.float64) for row in base_band_rows], axis=0)

        band_level[band] = {
            "primary_jsd": js_divergence(cond_agg["counts"], base_agg["counts"]),
            "focus_expert_selection_rate_delta": float(
                probability_from_counts(cond_agg["counts"])[focus_expert] -
                probability_from_counts(base_agg["counts"])[focus_expert]
            ),
            "focus_expert_weight_rate_delta": float(
                probability_from_counts(cond_agg["weights"])[focus_expert] -
                probability_from_counts(base_agg["weights"])[focus_expert]
            ),
            "focus_expert_cooccurrence_jsd": js_divergence(cond_co, base_co),
            "prompt_count": len(cond_band_rows),
        }

    return {
        "prompt_level": prompt_level,
        "band_level": band_level,
    }


def evaluate_acceptance(analysis: dict, sham_controls: dict | None) -> dict:
    if sham_controls is None:
        return {"status": "not_evaluated", "reason": "no sham_controls.json provided"}

    shams = sham_controls["selected"]
    baseline = analysis["conditions"].get("baseline")
    if baseline is None:
        return {"status": "not_evaluated", "reason": "baseline condition missing"}

    outcomes = {}
    for level in ["soft_bias_0.25", "soft_bias_0.5", "soft_bias_1.0", "soft_bias_2.0", "soft_bias_3.0", "forced_inclusion"]:
        focus_key = f"expert_114_{level}"
        sham_keys = [f"expert_{sham}_{level}" for sham in shams]
        if focus_key not in analysis["comparisons"] or any(key not in analysis["comparisons"] for key in sham_keys):
            continue

        focus_cmp = analysis["comparisons"][focus_key]
        sham_cmps = [analysis["comparisons"][key] for key in sham_keys]
        band_pass = {}
        for band in ["process", "regulation"]:
            focus_jsd = focus_cmp["band_level"][band]["primary_jsd"]
            band_pass[band] = all(focus_jsd > sham_cmp["band_level"][band]["primary_jsd"] for sham_cmp in sham_cmps)

        prompt_pass = {}
        for band in ["process", "regulation"]:
            focus_prompt = {row["prompt_id"]: row["rubric_delta"] for row in focus_cmp["prompt_level"] if row["band"] == band}
            sham_prompt = [
                {row["prompt_id"]: row["rubric_delta"] for row in sham_cmp["prompt_level"] if row["band"] == band}
                for sham_cmp in sham_cmps
            ]
            wins = 0
            for prompt_id, value in focus_prompt.items():
                if all(value > sham_map.get(prompt_id, float("-inf")) for sham_map in sham_prompt):
                    wins += 1
            prompt_pass[band] = {
                "wins": wins,
                "threshold": 6,
                "pass": wins >= 6,
            }

        static_median = np.median([
            row["rubric_delta"]
            for row in focus_cmp["prompt_level"]
            if row["band"] == "static_fact"
        ]) if any(row["band"] == "static_fact" for row in focus_cmp["prompt_level"]) else 0.0
        process_reg_values = [
            focus_cmp["band_level"][band]["primary_jsd"]
            for band in ["process", "regulation"]
        ]
        static_guard = float(focus_cmp["band_level"]["static_fact"]["primary_jsd"]) < (0.5 * float(np.median(process_reg_values)))

        outcomes[level] = {
            "band_level_pass": band_pass,
            "prompt_level_pass": prompt_pass,
            "static_fact_guard_pass": static_guard,
            "static_fact_median_rubric_delta": float(static_median),
        }

    if not outcomes:
        return {"status": "not_evaluated", "reason": "insufficient condition coverage for acceptance test"}

    return {
        "status": "evaluated",
        "outcomes": outcomes,
    }


def analyze_run(run_dir: pathlib.Path, prompt_suite_path: pathlib.Path, rubric_path: pathlib.Path, focus_expert: int, sham_controls_path: pathlib.Path | None = None) -> dict:
    suite = load_prompt_suite(prompt_suite_path)
    rubric = load_rubric(rubric_path)
    sham_controls = json.loads(sham_controls_path.read_text()) if sham_controls_path and sham_controls_path.exists() else None

    condition_dirs = sorted([
        path for path in (run_dir / "capture").iterdir()
        if path.is_dir()
    ])

    conditions = {}
    for condition_dir in condition_dirs:
        prompt_rows = []
        for prompt_dir in sorted(
            [path for path in condition_dir.iterdir() if path.is_dir() and (path / "metadata.txt").exists()],
            key=lambda path: path.name,
        ):
            prompt_info = suite["prompts"][prompt_dir.name]
            prompt_rows.append(analyze_prompt_dir(prompt_dir, prompt_info["band"], focus_expert, rubric))
        conditions[condition_dir.name] = prompt_rows

    baseline_rows = conditions.get("baseline", [])
    comparisons = {}
    for name, prompt_rows in conditions.items():
        if name == "baseline":
            continue
        comparisons[name] = compare_to_baseline(prompt_rows, baseline_rows, focus_expert)

    analysis = {
        "run_dir": str(run_dir),
        "focus_expert": focus_expert,
        "conditions": conditions,
        "comparisons": comparisons,
    }
    analysis["acceptance"] = evaluate_acceptance(analysis, sham_controls)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prompt-suite", default=str(_HERE / "prompt-suite-3band.json"))
    parser.add_argument("--rubric", default=str(_HERE / "rubric_markers.json"))
    parser.add_argument("--focus-expert", type=int, default=114)
    parser.add_argument("--sham-controls", default=str(_HERE / "sham_controls.json"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    output = pathlib.Path(args.output) if args.output else run_dir / "analysis.json"
    analysis = analyze_run(
        run_dir=run_dir,
        prompt_suite_path=pathlib.Path(args.prompt_suite),
        rubric_path=pathlib.Path(args.rubric),
        focus_expert=args.focus_expert,
        sham_controls_path=pathlib.Path(args.sham_controls),
    )
    output.write_text(json.dumps(analysis, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
