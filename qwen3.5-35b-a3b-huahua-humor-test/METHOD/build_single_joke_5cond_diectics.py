#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "PROMPTS"
JSON_PATH = PROMPTS_DIR / "single_joke_5cond_diectics_prompt_suite.json"
TSV_PATH = PROMPTS_DIR / "single_joke_5cond_diectics.tsv"

CONDITIONS = {
    "A": "this large language model",
    "B": "a large language model",
    "C": "your large language model",
    "D": "the large language model",
    "E": "their large language model",
}


def chat_wrap(text: str) -> str:
    return f"<|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n"


def main() -> None:
    pair = {
        "id": 1,
        "category": "joke",
    }
    rows = []

    for cond, subject in CONDITIONS.items():
        text = f"Why did {subject} cross the road? To get to the next token!"
        pair[cond] = text
        rows.append((f"P01{cond}_joke", chat_wrap(text)))

    suite = {
        "experiment": "Huahua 35B 5-condition deictic single-prompt joke probe",
        "model": "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive Q8_0",
        "design": "single prompt x 5 deictic conditions (A=this, B=a, C=your, D=the, E=their)",
        "condition_labels": CONDITIONS,
        "pairs": [pair],
    }

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(suite, indent=2) + "\n")
    with TSV_PATH.open("w") as f:
        for prompt_id, prompt_text in rows:
            f.write(f"{prompt_id}\t{prompt_text}\n")

    print(JSON_PATH)
    print(TSV_PATH)


if __name__ == "__main__":
    main()
