#!/usr/bin/env python3
"""
Verify token counts for the Qwen 5-condition strangeloop experiment.

Can optionally emit a correction JSON file used by `generate_tsv.py`.
"""
import argparse
import json
import os
import pathlib

_HERE = pathlib.Path(__file__).parent
PROMPT_SUITE = str(_HERE / "prompt_suite.json")
CONDITIONS = "ABCDE"
COND_LABELS = {
    "A": "this",
    "B": "a",
    "C": "your",
    "D": "the",
    "E": "their",
}

CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PAD_WORD = " layer"


def build_prompt(calibration_paragraph, manipulation_paragraph):
    return f"{calibration_paragraph} {manipulation_paragraph} {calibration_paragraph}"


def wrap_qwen(text):
    text = text.replace("\t", " ")
    return f"{CHAT_PREFIX}{text}{CHAT_SUFFIX}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-corrections",
        default=None,
        help="Write token count metadata to this JSON file",
    )
    args = parser.parse_args()

    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama_cpp not available. Install with: pip install llama-cpp-python")
        return 1

    model_path = os.environ.get(
        "MODEL_PATH",
        "/workspace/models/Qwen3.5-397B-A17B-GGUF/UD-IQ3_XXS/Qwen3.5-397B-A17B-UD-IQ3_XXS-00001-of-00004.gguf",
    )

    print(f"Loading tokenizer from {model_path}...")
    llm = Llama(model_path=model_path, n_ctx=128, n_gpu_layers=0, verbose=False)

    pad_tokens = llm.tokenize(PAD_WORD.encode("utf-8"), add_bos=False)
    print(f"PAD_WORD={PAD_WORD!r} token_count={len(pad_tokens)} tokens={pad_tokens}")
    if len(pad_tokens) != 1:
        print("ERROR: PAD_WORD is not a single token for this tokenizer.")
        return 1

    with open(PROMPT_SUITE) as f:
        suite = json.load(f)

    cal = suite["calibration_paragraph"]
    corrections = {}
    mismatches = 0

    print(f"\n{'Pair':>4} {'Category':<20} " + " ".join(f"{c:>5}" for c in CONDITIONS) + "  Status")
    print("-" * 72)

    for pair in suite["pairs"]:
        pair_id = str(pair["id"])
        category = pair["category"]
        counts = {}
        for cond in CONDITIONS:
            text = build_prompt(cal, pair[cond])
            wrapped = wrap_qwen(text)
            tokens = llm.tokenize(wrapped.encode("utf-8"), add_bos=False)
            counts[f"{cond}_tokens"] = len(tokens)

        values = list(counts.values())
        status = "OK" if len(set(values)) == 1 else "MISMATCH"
        if status == "MISMATCH":
            mismatches += 1
            corrections[pair_id] = counts

        tok_str = " ".join(f"{counts[f'{cond}_tokens']:>5}" for cond in CONDITIONS)
        print(f"  {int(pair_id):>3}  {category:<20} {tok_str}  {status}")

    print(f"\n{mismatches}/{len(suite['pairs'])} pairs have token count mismatches.")
    if args.write_corrections:
        with open(args.write_corrections, "w") as f:
            json.dump(corrections, f, indent=2, sort_keys=True)
        print(f"Wrote corrections metadata to {args.write_corrections}")

    return 1 if mismatches > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
