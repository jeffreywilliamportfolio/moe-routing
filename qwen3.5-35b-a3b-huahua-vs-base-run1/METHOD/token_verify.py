#!/usr/bin/env python3
"""Verify token counts for the Qwen3.5-35B-A3B comparison suite."""
import argparse
import json
import os
import pathlib
import subprocess

_HERE = pathlib.Path(__file__).parent
PROMPT_SUITE = str(_HERE / "prompt-suite.json")
CORRECTIONS_FILE = _HERE / "token_corrections.json"
CONDITIONS = "ABCDE"
COND_LABELS = {
    "A": "this",
    "B": "a",
    "C": "your",
    "D": "the",
    "E": "their",
}

CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
PAD_WORD = " layer"
LLAMA_TOKENIZE = os.environ.get(
    "LLAMA_TOKENIZE_BIN",
    "/workspace/llama.cpp/build/bin/llama-tokenize",
)


def build_prompt(calibration_paragraph, manipulation_paragraph):
    return f"{calibration_paragraph} {manipulation_paragraph} {calibration_paragraph}"


def wrap_qwen(text):
    text = text.replace("\t", " ")
    return f"{CHAT_PREFIX}{text}{CHAT_SUFFIX}"


def apply_corrections(pair: dict, corrections: dict) -> dict[str, str]:
    manipulations = {cond: pair[cond] for cond in CONDITIONS}
    pair_key = str(pair["id"])
    corr = corrections.get(pair_key)
    if not corr:
        return manipulations

    token_counts = [corr[f"{cond}_tokens"] for cond in CONDITIONS]
    max_tok = max(token_counts)
    for idx, cond in enumerate(CONDITIONS):
        diff = max_tok - token_counts[idx]
        if diff > 0:
            manipulations[cond] = manipulations[cond] + (PAD_WORD * diff)
    return manipulations


def get_llama_cpp_counter(model_path: str):
    try:
        from llama_cpp import Llama
    except ImportError:
        return None

    try:
        llm = Llama(model_path=model_path, n_ctx=128, n_gpu_layers=0, verbose=False)
    except Exception:
        return None

    def counter(text: str) -> int:
        return len(llm.tokenize(text.encode("utf-8"), add_bos=False))

    return counter


def _count_tokens_cli(model_path: str, text: str) -> int:
    cmd = [
        LLAMA_TOKENIZE,
        "-m", model_path,
        "--stdin",
        "--no-bos",
        "--show-count",
        "--ids",
        "--log-disable",
    ]
    completed = subprocess.run(
        cmd,
        input=text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"llama-tokenize failed ({completed.returncode}): {completed.stderr.strip()}"
        )

    for line in completed.stdout.splitlines():
        if line.startswith("Total number of tokens:"):
            return int(line.rsplit(":", 1)[1].strip())
    raise RuntimeError("llama-tokenize did not report token count")


def get_token_counter(model_path: str):
    llama_cpp_counter = get_llama_cpp_counter(model_path)
    if llama_cpp_counter is not None:
        return "llama_cpp", llama_cpp_counter
    return "llama-tokenize", lambda text: _count_tokens_cli(model_path, text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-corrections",
        default=None,
        help="Write token count metadata to this JSON file",
    )
    parser.add_argument(
        "--corrections",
        default=str(CORRECTIONS_FILE) if CORRECTIONS_FILE.exists() else None,
        help="Optional correction JSON to apply before token verification",
    )
    args = parser.parse_args()

    model_path = os.environ.get(
        "TOKENIZER_MODEL_PATH",
        os.environ.get(
        "MODEL_PATH",
        "/workspace/models/Qwen3.5-35B-A3B-BF16/BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf",
        ),
    )

    backend, count_tokens = get_token_counter(model_path)
    print(f"Loading tokenizer from {model_path} via {backend}...")

    pad_count = count_tokens(PAD_WORD)
    print(f"PAD_WORD={PAD_WORD!r} token_count={pad_count}")
    if pad_count != 1:
        print("ERROR: PAD_WORD is not a single token for this tokenizer.")
        return 1

    with open(PROMPT_SUITE) as f:
        suite = json.load(f)

    cal = suite["calibration_paragraph"]
    input_corrections = {}
    if args.corrections and pathlib.Path(args.corrections).exists():
        with open(args.corrections) as f:
            input_corrections = json.load(f)
        print(f"Applying corrections from {args.corrections}: {len(input_corrections)} pairs")
    corrections = {}
    mismatches = 0

    print(f"\n{'Pair':>4} {'Category':<20} " + " ".join(f"{c:>5}" for c in CONDITIONS) + "  Status")
    print("-" * 72)

    for pair in suite["pairs"]:
        pair_id = str(pair["id"])
        category = pair["category"]
        counts = {}
        manipulations = apply_corrections(pair, input_corrections)
        for cond in CONDITIONS:
            text = build_prompt(cal, manipulations[cond])
            wrapped = wrap_qwen(text)
            counts[f"{cond}_tokens"] = count_tokens(wrapped)

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
