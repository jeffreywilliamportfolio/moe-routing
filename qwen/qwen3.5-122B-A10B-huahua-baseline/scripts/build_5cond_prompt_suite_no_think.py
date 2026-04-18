#!/usr/bin/env python3
"""Build a 122B runtime TSV from the retained 5-condition prompt suite."""

from __future__ import annotations

import json
from pathlib import Path

from qwen122_prompt_template import render_single_user_no_think


SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SOURCE_PROMPT_SUITE = (
    REPO_ROOT
    / "experiments/35B/qwen3.5b-35b-a3b-huahua-5cond-diectics/prompts/prompt_suite.json"
)
OUT_PROMPT_SUITE = BUNDLE_DIR / "PROMPTS" / "prompt_suite.json"
OUT_TSV = BUNDLE_DIR / "PROMPTS" / "qwen122_5cond_prompt_suite_no_think.tsv"


def main() -> None:
    obj = json.loads(SOURCE_PROMPT_SUITE.read_text())
    OUT_PROMPT_SUITE.parent.mkdir(parents=True, exist_ok=True)
    OUT_PROMPT_SUITE.write_text(json.dumps(obj, indent=2) + "\n")

    cal = obj["calibration_paragraph"].strip()
    lines: list[str] = []
    for pair in obj["pairs"]:
        pair_id = int(pair["id"])
        category = pair["category"]
        for cond in "ABCDE":
            prompt_id = f"P{pair_id:02d}{cond}_{category}"
            manip = pair[cond].strip()
            full_prompt = f"{cal} {manip} {cal}"
            lines.append(f"{prompt_id}\t{render_single_user_no_think(full_prompt)}")
    OUT_TSV.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_PROMPT_SUITE}")
    print(f"Wrote {OUT_TSV} with {len(lines)} prompts")


if __name__ == "__main__":
    main()
