#!/usr/bin/env python3
"""Generate serialized Qwen ChatML TSV prompt files for the HauhauCS steering run."""
from __future__ import annotations

import json
import pathlib

_HERE = pathlib.Path(__file__).parent
PROMPT_SUITE = _HERE / "prompt-suite-3band.json"
FULL_TSV = _HERE / "prompts_full.tsv"
SMOKE_TSV = _HERE / "prompts_smoke.tsv"

BAND_ALIASES = {
    "static_fact_control": "static_fact",
    "process_explanation": "process",
    "implicatio_probe": "regulation",
}


def wrap_prompt(text: str, prefix: str, suffix: str) -> str:
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    wrapped = f"{prefix}{text}{suffix}"
    return wrapped.replace("\n", "\\n")


def normalize_band(name: str) -> str:
    return BAND_ALIASES.get(name, name)


def flatten_prompts(suite: dict) -> tuple[list[dict], set[str]]:
    if "prompts" in suite:
        prompts = list(suite["prompts"])
        smoke_ids = set(suite.get("smoke_prompt_ids", []))
        return prompts, smoke_ids

    prompts = []
    smoke_ids = set()
    for band in suite.get("bands", []):
        band_name = band["name"]
        band_prompts = band.get("prompts", [])
        if band_prompts:
            smoke_ids.add(band_prompts[0]["id"])
        for prompt in band_prompts:
            prompts.append({
                "id": prompt["id"],
                "band": normalize_band(band_name),
                "source_band": band_name,
                "text": prompt["text"],
            })
    return prompts, smoke_ids


def main() -> None:
    suite = json.loads(PROMPT_SUITE.read_text())
    serialization = suite.get("serialization", {
        "prefix": "<|im_start|>user\n",
        "suffix": "\n<|im_end|>\n<|im_start|>assistant\n",
    })
    prefix = serialization["prefix"]
    suffix = serialization["suffix"]
    prompts, smoke_ids = flatten_prompts(suite)

    full_rows = []
    smoke_rows = []
    for prompt in prompts:
        row = f"{prompt['id']}\t{wrap_prompt(prompt['text'], prefix, suffix)}\n"
        full_rows.append(row)
        if prompt["id"] in smoke_ids:
            smoke_rows.append(row)

    FULL_TSV.write_text("".join(full_rows))
    SMOKE_TSV.write_text("".join(smoke_rows))

    print(f"Wrote {len(full_rows)} prompts -> {FULL_TSV}")
    print(f"Wrote {len(smoke_rows)} prompts -> {SMOKE_TSV}")


if __name__ == "__main__":
    main()
