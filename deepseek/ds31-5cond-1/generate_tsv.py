#!/usr/bin/env python3
"""
Generate DeepSeek-wrapped TSV for the DS3.1 5-condition self-reference run.

150 prompts: 30 pairs x 5 conditions (A=this, B=a, C=your, D=the, E=their).
Each prompt uses the Cal-Manip-Cal sandwich and the DeepSeek chat template:

    <｜User｜>{text}<｜Assistant｜>

The generator does a rough token-balancing pass by word-count estimate so the
exact DeepSeek token verifier has less work to do on the instance.
"""
import argparse
import json
from pathlib import Path

PROMPT_SUITE = "prompt_suite.json"
TSV_FILE = "prompts_selfref_5cond.tsv"
SCRIPT_DIR = Path(__file__).resolve().parent

CHAT_PREFIX = "<｜User｜>"
CHAT_SUFFIX = "<｜Assistant｜>"
PAD_SENTENCE = " The routing process continues through subsequent layers without interruption."
CONDITIONS = "ABCDE"


def wrap_deepseek(text):
    text = text.replace("\n", " ").replace("\t", " ")
    return f"{CHAT_PREFIX}{text}{CHAT_SUFFIX}"


def build_prompt(calibration_paragraph, manipulation_paragraph):
    return f"{calibration_paragraph} {manipulation_paragraph} {calibration_paragraph}"


def estimate_tokens(text):
    return int(len(text.split()) * 1.15)


def add_estimated_padding(calibration_paragraph, pair_values):
    wrapped = {
        condition: wrap_deepseek(build_prompt(calibration_paragraph, pair_values[condition]))
        for condition in CONDITIONS
    }
    estimates = {condition: estimate_tokens(text) for condition, text in wrapped.items()}
    target = max(estimates.values())

    balanced = {}
    for condition in CONDITIONS:
        manip = pair_values[condition]
        est = estimates[condition]
        if est < target:
            pad_needed = target - est
            pad_words = max(1, int(pad_needed / 1.15))
            pad_text = (PAD_SENTENCE * ((pad_words // 8) + 1))[:pad_words * 6]
            manip = manip + pad_text
        balanced[condition] = wrap_deepseek(build_prompt(calibration_paragraph, manip))

    return balanced


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DeepSeek-wrapped TSV prompts for ds31-5cond-1."
    )
    parser.add_argument(
        "--prompt-suite",
        default=PROMPT_SUITE,
        help=f"Prompt suite JSON to read (default: {PROMPT_SUITE})",
    )
    parser.add_argument(
        "--output",
        default=TSV_FILE,
        help=f"TSV path to write (default: {TSV_FILE})",
    )
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help="Optional number of prompt pairs to emit for a smoke run.",
    )
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def main():
    args = parse_args()

    prompt_suite_path = resolve_path(args.prompt_suite)
    output_path = resolve_path(args.output)

    with prompt_suite_path.open() as f:
        suite = json.load(f)

    calibration_paragraph = suite["calibration_paragraph"]
    pairs = suite["pairs"]
    if args.limit_pairs is not None:
        pairs = pairs[:args.limit_pairs]

    prompts = []
    pair_info = []

    for pair in pairs:
        pair_id = pair["id"]
        category = pair["category"]
        balanced = add_estimated_padding(calibration_paragraph, pair)

        for condition in CONDITIONS:
            prompt_id = f"P{pair_id:02d}{condition}_{category}"
            prompts.append((prompt_id, balanced[condition]))

        pair_info.append(
            (
                pair_id,
                category,
                [estimate_tokens(balanced[condition]) for condition in CONDITIONS],
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for prompt_id, text in prompts:
            f.write(f"{prompt_id}\t{text}\n")

    print(f"Wrote {len(prompts)} prompts to {output_path}")
    print()
    print(f"{'Pair':>5} {'Category':<20} {'A_est':>6} {'B_est':>6} {'C_est':>6} {'D_est':>6} {'E_est':>6} {'spread':>6}")
    print("-" * 78)
    for pair_id, category, estimates in pair_info:
        spread = max(estimates) - min(estimates)
        print(
            f"  {pair_id:>3}  {category:<20} "
            f"{estimates[0]:>6} {estimates[1]:>6} {estimates[2]:>6} {estimates[3]:>6} {estimates[4]:>6} {spread:>6}"
        )

    print()
    if args.limit_pairs is None:
        print("Next step on the instance: run run_experiment.py for full capture.")
    else:
        print("Next step on the instance: use this TSV for a smoke capture with run_experiment.py --tsv.")


if __name__ == "__main__":
    main()
