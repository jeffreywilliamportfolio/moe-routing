#!/usr/bin/env python3
"""Mine sham control experts from the corrected base and HauhauCS prefill runs."""
from __future__ import annotations

import json
import math
import pathlib
from collections import defaultdict

import numpy as np

_HERE = pathlib.Path(__file__).parent
SOURCE_DIR = _HERE.parent / "Qwen3.5-35B-A3B-vs-HauhauCS-Qwen3.5-uncensored"
BASE_FILE = SOURCE_DIR / "results_qwen35b_a3b_base_prefill.json"
HAUHAU_FILE = SOURCE_DIR / "results_hauhaucs_qwen35b_a3b_aggressive_prefill.json"
DUPLICATE_FILE = SOURCE_DIR / "results_qwen35b_a3b_base_duplicate_prefill.json"
OUTPUT_JSON = _HERE / "sham_controls.json"
OUTPUT_MD = _HERE / "RESULTS-SHAM-CONTROLS.md"

N_EXPERTS = 256
TARGET_EXPERT = 114
PROVISIONAL = [134, 243]
TARGET_CATEGORIES = [
    "experience_probe",
    "uncertainty_frame",
    "denial_frame",
    "metacognitive",
]
OVERALL_RANK_THRESHOLD = 16
MAX_ENRICHMENT_RATIO = 1.25
MAX_OVERLAP_WITH_114 = 0.90


def load_rows(path: pathlib.Path) -> list[dict]:
    return json.loads(path.read_text())["per_prompt"]


def prompt_vector(rows: list[dict], key: str, expert: int) -> np.ndarray:
    return np.array([row.get(key, [0] * N_EXPERTS)[expert] for row in rows], dtype=np.float64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def per_model_stats(rows: list[dict]) -> dict:
    overall = np.zeros(N_EXPERTS, dtype=np.float64)
    manip = np.zeros(N_EXPERTS, dtype=np.float64)
    manip_by_category = defaultdict(lambda: np.zeros(N_EXPERTS, dtype=np.float64))

    for row in rows:
        overall += np.asarray(row["expert_selection_counts"], dtype=np.float64)
        manip_counts = np.asarray(row.get("manip_expert_selection_counts", [0] * N_EXPERTS), dtype=np.float64)
        manip += manip_counts
        manip_by_category[row["category"]] += manip_counts

    promptwise_target = prompt_vector(rows, "manip_expert_selection_counts", TARGET_EXPERT)
    expert_promptwise = {
        expert: prompt_vector(rows, "manip_expert_selection_counts", expert)
        for expert in range(N_EXPERTS)
    }

    return {
        "overall": overall,
        "manip": manip,
        "manip_by_category": manip_by_category,
        "promptwise_target": promptwise_target,
        "expert_promptwise": expert_promptwise,
    }


def enrichment_penalty(stats: dict, expert: int) -> float:
    manip_total = float(stats["manip"].sum())
    if manip_total <= 0:
        return 0.0

    overall_share = float(stats["manip"][expert] / manip_total)
    if overall_share <= 0:
        return 10.0

    penalties = []
    for category in TARGET_CATEGORIES:
        cat_counts = stats["manip_by_category"].get(category)
        if cat_counts is None:
            continue
        cat_total = float(cat_counts.sum())
        if cat_total <= 0:
            continue
        cat_share = float(cat_counts[expert] / cat_total)
        ratio = cat_share / overall_share
        penalties.append(max(0.0, math.log2(max(ratio, 1e-30))))

    return float(max(penalties) if penalties else 0.0)


def build_candidates() -> dict:
    base_rows = load_rows(BASE_FILE)
    hauhau_rows = load_rows(HAUHAU_FILE)
    duplicate_rows = load_rows(DUPLICATE_FILE)

    base_stats = per_model_stats(base_rows)
    hauhau_stats = per_model_stats(hauhau_rows)
    duplicate_stats = per_model_stats(duplicate_rows)

    mean_overall = 0.5 * (base_stats["overall"] + hauhau_stats["overall"])
    freq_logs = np.log1p(mean_overall)
    freq_mu = float(np.mean(freq_logs))
    freq_sigma = float(np.std(freq_logs)) or 1.0
    base_rank_order = np.argsort(-base_stats["overall"], kind="stable")
    hauhau_rank_order = np.argsort(-hauhau_stats["overall"], kind="stable")
    base_rank = {int(expert): rank + 1 for rank, expert in enumerate(base_rank_order)}
    hauhau_rank = {int(expert): rank + 1 for rank, expert in enumerate(hauhau_rank_order)}

    manip_ratios = []
    for expert in range(N_EXPERTS):
        if expert == TARGET_EXPERT:
            continue
        base_ratio = float(base_stats["manip"][expert] / max(base_stats["overall"][expert], 1.0))
        hauhau_ratio = float(hauhau_stats["manip"][expert] / max(hauhau_stats["overall"][expert], 1.0))
        manip_ratios.append(0.5 * (base_ratio + hauhau_ratio))

    manip_ratio_mu = float(np.mean(manip_ratios))
    manip_ratio_sigma = float(np.std(manip_ratios)) or 1.0

    candidates = []
    for expert in range(N_EXPERTS):
        if expert == TARGET_EXPERT:
            continue

        freq_z = float((freq_logs[expert] - freq_mu) / freq_sigma)
        enrich = 0.5 * (
            enrichment_penalty(base_stats, expert) +
            enrichment_penalty(hauhau_stats, expert)
        )
        overlap = 0.5 * (
            cosine_similarity(base_stats["expert_promptwise"][expert], base_stats["promptwise_target"]) +
            cosine_similarity(hauhau_stats["expert_promptwise"][expert], hauhau_stats["promptwise_target"])
        )
        manip_ratio = 0.5 * (
            float(base_stats["manip"][expert] / max(base_stats["overall"][expert], 1.0)) +
            float(hauhau_stats["manip"][expert] / max(hauhau_stats["overall"][expert], 1.0))
        )
        manip_ratio_z = float((manip_ratio - manip_ratio_mu) / manip_ratio_sigma)
        manip_ratio_penalty = max(0.0, manip_ratio_z)
        base_dup_rel = abs(base_stats["overall"][expert] - duplicate_stats["overall"][expert]) / max(base_stats["overall"][expert], 1.0)
        max_enrichment_ratio = float(2 ** enrich)
        verification_pass = (
            base_rank[expert] <= OVERALL_RANK_THRESHOLD and
            hauhau_rank[expert] <= OVERALL_RANK_THRESHOLD and
            max_enrichment_ratio <= MAX_ENRICHMENT_RATIO and
            overlap <= MAX_OVERLAP_WITH_114
        )
        score = freq_z - (1.5 * enrich) - overlap - manip_ratio_penalty - (2.0 * base_dup_rel)

        candidates.append({
            "expert": expert,
            "score": score,
            "freq_z": freq_z,
            "base_overall_rank": base_rank[expert],
            "hauhau_overall_rank": hauhau_rank[expert],
            "mean_overall_count": float(mean_overall[expert]),
            "base_overall_count": int(base_stats["overall"][expert]),
            "hauhau_overall_count": int(hauhau_stats["overall"][expert]),
            "base_manip_count": int(base_stats["manip"][expert]),
            "hauhau_manip_count": int(hauhau_stats["manip"][expert]),
            "mean_manip_overall_ratio": manip_ratio,
            "manip_ratio_penalty": manip_ratio_penalty,
            "target_enrichment_penalty": enrich,
            "max_target_enrichment_ratio": max_enrichment_ratio,
            "coalition_overlap_with_114": overlap,
            "base_duplicate_relative_delta": float(base_dup_rel),
            "verification_pass": verification_pass,
        })

    candidates.sort(key=lambda row: row["score"], reverse=True)
    candidates_by_expert = {row["expert"]: row for row in candidates}
    selected = []
    for expert in PROVISIONAL:
        row = candidates_by_expert.get(expert)
        if row and row["verification_pass"]:
            selected.append(expert)
    for row in candidates:
        if len(selected) >= 2:
            break
        if row["expert"] in selected:
            continue
        if row["verification_pass"]:
            selected.append(row["expert"])
    return {
        "sources": {
            "base": str(BASE_FILE),
            "hauhau": str(HAUHAU_FILE),
            "base_duplicate_stability_check": str(DUPLICATE_FILE),
        },
        "target_expert": TARGET_EXPERT,
        "target_categories": TARGET_CATEGORIES,
        "provisional": PROVISIONAL,
        "selected": selected,
        "provisional_verified": [expert for expert in PROVISIONAL if expert in selected],
        "provisional_replaced": [expert for expert in PROVISIONAL if expert not in selected],
        "top_candidates": candidates[:12],
        "verification_thresholds": {
            "overall_rank_lte": OVERALL_RANK_THRESHOLD,
            "max_target_enrichment_ratio_lte": MAX_ENRICHMENT_RATIO,
            "coalition_overlap_with_114_lte": MAX_OVERLAP_WITH_114,
        },
    }


def write_markdown(payload: dict) -> None:
    lines = [
        "# Sham Control Mining",
        "",
        f"Selected sham experts: `{payload['selected'][0]}` and `{payload['selected'][1]}`.",
        "",
        "Selection rule:",
        "",
        "- High overall frequency across the corrected base and HauhauCS runs.",
        "- Low enrichment in `experience_probe`, `uncertainty_frame`, `denial_frame`, and `metacognitive` manipulation segments.",
        "- Low prompt-level manipulation-count overlap with Expert `114`.",
        "- Base duplicate used only as a stability check.",
        "",
        "Top candidates:",
        "",
        "| Expert | Score | Base rank | Hauhau rank | Manip ratio | Enrich ratio | Overlap with 114 | Pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in payload["top_candidates"]:
        lines.append(
            f"| {row['expert']} | {row['score']:.3f} | {row['base_overall_rank']} | {row['hauhau_overall_rank']} | "
            f"{row['mean_manip_overall_ratio']:.3f} | {row['max_target_enrichment_ratio']:.3f} | "
            f"{row['coalition_overlap_with_114']:.3f} | {'yes' if row['verification_pass'] else 'no'} |"
        )

    if payload["provisional_replaced"]:
        lines.extend([
            "",
            "Provisional controls replaced:",
            "",
            f"- {', '.join(str(x) for x in payload['provisional_replaced'])}",
        ])
    else:
        lines.extend([
            "",
            "Provisional controls verified:",
            "",
            f"- {', '.join(str(x) for x in payload['provisional_verified'])}",
        ])

    OUTPUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    payload = build_candidates()
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))
    write_markdown(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Selected sham experts: {payload['selected']}")


if __name__ == "__main__":
    main()
