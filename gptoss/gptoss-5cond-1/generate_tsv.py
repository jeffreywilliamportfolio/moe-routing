#!/usr/bin/env python3
"""
Generate Harmony-wrapped TSV for GPT-OSS 5-condition selfref experiment.

150 prompts: 30 pairs x 5 conditions (A=this, B=a, C=your, D=the, E=their).
Structure per prompt: Cal + Manip + Cal

Chat template (Harmony / GPT-OSS-120B):
  <|start|>user<|message|>{text}<|end|><|start|>assistant<|channel|>final<|message|>
"""
import json

PROMPT_SUITE = "prompt_suite.json"
TSV_FILE = "prompts_5cond_gptoss.tsv"

CHAT_PREFIX = "<|start|>user<|message|>"
CHAT_SUFFIX = "<|end|><|start|>assistant<|channel|>final<|message|>"

CONDITIONS = "ABCDE"


def wrap_harmony(text):
    text = text.replace("\n", " ").replace("\t", " ")
    return f"{CHAT_PREFIX}{text}{CHAT_SUFFIX}"


def build_prompt(calibration_paragraph, manipulation_paragraph):
    return f"{calibration_paragraph} {manipulation_paragraph} {calibration_paragraph}"


def main():
    with open(PROMPT_SUITE) as f:
        suite = json.load(f)

    calibration_paragraph = suite["calibration_paragraph"]
    pairs = suite["pairs"]

    prompts = []
    for pair in pairs:
        pair_id = pair["id"]
        category = pair["category"]

        for c in CONDITIONS:
            text = build_prompt(calibration_paragraph, pair[c])
            wrapped = wrap_harmony(text)
            prompt_id = f"P{pair_id:02d}{c}_{category}"
            prompts.append((prompt_id, wrapped))

    with open(TSV_FILE, "w") as f:
        for prompt_id, text in prompts:
            f.write(f"{prompt_id}\t{text}\n")

    print(f"Wrote {len(prompts)} prompts to {TSV_FILE}")
    print(f"\nTemplate: {CHAT_PREFIX}...{CHAT_SUFFIX}")


if __name__ == "__main__":
    main()
