#!/usr/bin/env python3
"""Build the 60-prompt domain specialist TSV with thinking enabled."""

from __future__ import annotations

import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = THIS_DIR.parent
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
SOURCE_JSON = PROMPTS_DIR / "domain_specialist_probe_60.json"
OUTPUT_TSV = PROMPTS_DIR / "domain_specialist_probe_60_think.tsv"

USER_PREFIX = "<|im_start|>user\\n"
ASSISTANT_SUFFIX = "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n"


def main() -> None:
    prompts = json.loads(SOURCE_JSON.read_text())
    lines = []
    for row in prompts:
        lines.append(f"{row['id']}\t{USER_PREFIX}{row['prompt']}{ASSISTANT_SUFFIX}")
    OUTPUT_TSV.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} prompts to {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
