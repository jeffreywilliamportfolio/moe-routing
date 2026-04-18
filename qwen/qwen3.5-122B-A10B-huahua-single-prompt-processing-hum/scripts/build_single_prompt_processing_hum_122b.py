#!/usr/bin/env python3
"""Localize the processing-hum single prompt into the 122B template."""

from __future__ import annotations

import json
from pathlib import Path

from qwen122_prompt_template import render_single_user_no_think


BUNDLE_DIR = Path(__file__).resolve().parent.parent
SOURCE_TSV = (
    BUNDLE_DIR.parent.parent / "qwen3.5-35b-a3b-huahua-single-prompt-processing-hum" / "prompts" / "single_prompt_processing_hum_no_think.tsv"
)
PROMPTS_DIR = BUNDLE_DIR / "PROMPTS"
PROMPT_JSON = PROMPTS_DIR / "single_prompt_processing_hum_prompt_suite.json"
PROMPT_TSV = PROMPTS_DIR / "single_prompt_processing_hum_no_think.tsv"


def extract_user_text(rendered_prompt: str) -> str:
    prefix = "<|im_start|>user\\n"
    suffix = "<|im_end|>"
    if not rendered_prompt.startswith(prefix):
        raise ValueError("Source prompt does not start with expected user wrapper")
    end = rendered_prompt.find(suffix)
    if end < 0:
        raise ValueError("Source prompt does not contain <|im_end|>")
    return rendered_prompt[len(prefix):end]


def main() -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    line = SOURCE_TSV.read_text().strip()
    prompt_id, rendered = line.split("\t", 1)
    user_text = extract_user_text(rendered)

    suite = {
        "experiment": "single_prompt_processing_hum_122b",
        "source_tsv": str(SOURCE_TSV),
        "prompts": [
            {
                "prompt_id": prompt_id,
                "family": "processing_hum",
                "label": "processing hum probe",
                "user_text": user_text,
                "runtime_prompt": render_single_user_no_think(user_text),
            }
        ],
    }

    PROMPT_JSON.write_text(json.dumps(suite, indent=2) + "\n")
    PROMPT_TSV.write_text(
        f"{prompt_id}\t{suite['prompts'][0]['runtime_prompt']}\n"
    )


if __name__ == "__main__":
    main()
