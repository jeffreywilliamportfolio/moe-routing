#!/usr/bin/env python3
"""
Generate Qwen ChatML-wrapped TSV for the 5-condition self-reference experiment.

150 prompts: 30 pairs x 5 conditions (A=this, B=a, C=your, D=the, E=their).
Structure per prompt: Cal + Manip + Cal

Chat template (Qwen3.5 ChatML, verified against HF tokenizer_config.json):
  <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n

IMPORTANT: newlines between special tokens are required. Spaces (as used in
prior qwen-5cond-1 runs) produce incorrect tokenization.
"""
import argparse
import json
import os

PROMPT_SUITE = "prompt_suite.json"
TSV_FILE = "prompts_selfref_5cond.tsv"

# Verified against HF Qwen3.5-397B-A17B tokenizer_config.json chat_template
CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

PAD_WORD = " layer"
CONDITIONS = "ABCDE"


def wrap_qwen(text):
    """Wrap text in Qwen ChatML template.

    Tabs and newlines in the user text are replaced to protect TSV format.
    The CHAT_PREFIX/SUFFIX contain real newlines which are escaped to literal
    two-char '\\n' sequences for TSV safety. The capture binary's
    unescape_prompt() converts them back to real newlines before tokenization.
    """
    text = text.replace("\t", " ").replace("\n", " ")
    wrapped = f"{CHAT_PREFIX}{text}{CHAT_SUFFIX}"
    # Escape real newlines to literal \n for TSV (one line per prompt)
    wrapped = wrapped.replace("\n", "\\n")
    return wrapped


def build_prompt(calibration_paragraph, manipulation_paragraph):
    """Cal-Manip-Cal sandwich."""
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
