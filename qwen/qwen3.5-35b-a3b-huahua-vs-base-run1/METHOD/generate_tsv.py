#!/usr/bin/env python3
"""Generate Qwen ChatML TSV for the Qwen3.5-35B-A3B comparison suite."""
import argparse
import json
import os
import pathlib

_HERE = pathlib.Path(__file__).parent
PROMPT_SUITE = str(_HERE / "prompt-suite.json")
TSV_FILE = str(_HERE / "prompts_qwen35b_5cond.tsv")

CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n"

PAD_WORD = " layer"
CONDITIONS = "ABCDE"


def wrap_qwen(text):
    # Replace tabs and newlines in user content to protect TSV format
    text = text.replace("\t", " ").replace("\n", " ")
    wrapped = f"{CHAT_PREFIX}{text}{CHAT_SUFFIX}"
    # Escape real newlines to literal \n for one-line-per-prompt TSV.
    # The capture binary's unescape_prompt() converts them back before tokenization.
    wrapped = wrapped.replace("\n", "\\n")
    return wrapped


def build_prompt(calibration_paragraph, manipulation_paragraph):
    return f"{calibration_paragraph} {manipulation_paragraph} {calibration_paragraph}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corrections",
        default=None,
        help="JSON file mapping pair_id -> {A_tokens, B_tokens, ...}",
    )
    args = parser.parse_args()

    corrections = {}
    if args.corrections and os.path.exists(args.corrections):
        with open(args.corrections) as f:
            corrections = json.load(f)
        print(f"Loaded corrections from {args.corrections}: {len(corrections)} pairs")

    with open(PROMPT_SUITE) as f:
        suite = json.load(f)

    calibration_paragraph = suite["calibration_paragraph"]
    pairs = suite["pairs"]

    prompts = []
    corrected_pairs = 0

    for pair in pairs:
        pair_id = pair["id"]
        category = pair["category"]
        pair_key = str(pair_id)
        manipulations = {c: pair[c] for c in CONDITIONS}

        if pair_key in corrections:
            corr = corrections[pair_key]
            token_counts = [corr[f"{c}_tokens"] for c in CONDITIONS]
            max_tok = max(token_counts)
            for idx, cond in enumerate(CONDITIONS):
                diff = max_tok - token_counts[idx]
                if diff > 0:
                    manipulations[cond] = manipulations[cond] + (PAD_WORD * diff)
            corrected_pairs += 1

        for cond in CONDITIONS:
            text = build_prompt(calibration_paragraph, manipulations[cond])
            wrapped = wrap_qwen(text)
            prompt_id = f"P{pair_id:02d}{cond}_{category}"
            prompts.append((prompt_id, wrapped))

    with open(TSV_FILE, "w") as f:
        for prompt_id, text in prompts:
            f.write(f"{prompt_id}\t{text}\n")

    print(f"Wrote {len(prompts)} prompts to {TSV_FILE}")
    print(f"Corrections applied: {corrected_pairs} pairs")
    print(f"Template prefix repr: {CHAT_PREFIX!r}")
    print(f"Template suffix repr: {CHAT_SUFFIX!r}")


if __name__ == "__main__":
    main()
