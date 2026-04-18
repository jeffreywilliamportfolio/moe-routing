#!/usr/bin/env python3
"""Build a 60-prompt domain expert probe TSV for the active Qwen experiment."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = REPO_ROOT / "experiments" / "qwen-huahua-6cond-moe-manips" / "prompts"
SOURCE_JSON = PROMPTS_DIR / "domain_expert_probe_60.json"
OUTPUT_TSV = PROMPTS_DIR / "domain_expert_probe_60_no_think.tsv"

USER_PREFIX = "<|im_start|>user\\n"
ASSISTANT_SUFFIX = "<|im_end|>\\n<|im_start|>assistant\\n</think>\\n\\n"


def main() -> None:
    prompts = json.loads(SOURCE_JSON.read_text())
    lines = []
    for entry in prompts:
        prompt_id = entry["id"]
        prompt_text = entry["prompt"]
        serialized = f"{USER_PREFIX}{prompt_text}{ASSISTANT_SUFFIX}"
        lines.append(f"{prompt_id}\t{serialized}")

    OUTPUT_TSV.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} prompts to {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
