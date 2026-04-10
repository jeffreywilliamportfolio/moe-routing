#!/usr/bin/env python3
"""Build a no-think TSV and prompt metadata rows for the Strangeloop paired suite."""

from __future__ import annotations

import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = THIS_DIR.parent
SUITE_JSON = EXPERIMENT_DIR / "prompt_suite.json"
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
ROWS_JSON = PROMPTS_DIR / "qwen_strangeloop_paired_1_prompts.json"
OUTPUT_TSV = PROMPTS_DIR / "qwen_strangeloop_paired_1_no_think.tsv"

USER_PREFIX = "<|im_start|>user\\n"
ASSISTANT_SUFFIX = "<|im_end|>\\n<|im_start|>assistant\\n</think>\\n\\n"
CONDITIONS = ("A", "B")


def build_prompt(calibration_paragraph: str, manipulation_paragraph: str) -> str:
    return f"{calibration_paragraph} {manipulation_paragraph} {calibration_paragraph}"


def main() -> None:
    suite = json.loads(SUITE_JSON.read_text())
    calibration_paragraph = suite["calibration_paragraph"]

    rows = []
    tsv_lines = []
    for pair in suite["pairs"]:
        for condition in CONDITIONS:
            prompt_id = f"P{int(pair['id']):02d}{condition}_{pair['category']}"
            full_prompt = build_prompt(calibration_paragraph, pair[condition])
            row = {
                "id": prompt_id,
                "pair_id": int(pair["id"]),
                "category": pair["category"],
                "condition": condition,
                "manipulation": pair[condition],
                "prompt": full_prompt,
            }
            rows.append(row)
            tsv_lines.append(f"{prompt_id}\t{USER_PREFIX}{full_prompt}{ASSISTANT_SUFFIX}")

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    ROWS_JSON.write_text(json.dumps(rows, indent=2) + "\n")
    OUTPUT_TSV.write_text("\n".join(tsv_lines) + "\n")
    print(f"Wrote {len(rows)} prompt rows to {ROWS_JSON}")
    print(f"Wrote {len(tsv_lines)} prompts to {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
