#!/usr/bin/env python3
"""
Derive tokenizer-verified A/B correction deltas for gptoss-strangeloop-paired-1.

This script uses the exact GPT-OSS capture path to measure `n_tokens_prompt`
for each prompt, writes `token_corrections.json`, regenerates the corrected TSV,
and verifies that every A/B pair is token-matched.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

MODEL = os.environ.get(
    "MODEL_PATH",
    "/workspace/models/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf",
)
BINARY = os.environ.get(
    "CAPTURE_BINARY",
    "/workspace/consciousness-experiment/capture_activations",
)
LLAMA_BUILD_BIN = os.environ.get(
    "LLAMA_BUILD_BIN",
    "/workspace/src/llama.cpp-b8123/build-cuda/bin",
)

TSV = "prompts_strangeloop_paired.tsv"
CORRECTIONS_FILE = "token_corrections.json"
WORK_DIR = pathlib.Path(".tmp_token_corrections")
RAW_OUTPUT_DIR = WORK_DIR / "raw_output"
VERIFY_OUTPUT_DIR = WORK_DIR / "verify_output"

N_PREDICT = 0
NGL = int(os.environ.get("NGL", "999"))
CTX = int(os.environ.get("CTX", "4096"))
THREADS = int(os.environ.get("THREADS", "16"))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "off")
CACHE_TYPE_K = os.environ.get("CACHE_TYPE_K", "f16")
CACHE_TYPE_V = os.environ.get("CACHE_TYPE_V", "f16")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "15"))


def ensure_inputs() -> None:
    if not pathlib.Path(BINARY).exists():
        raise FileNotFoundError(f"Capture binary not found: {BINARY}")
    if not pathlib.Path(MODEL).exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("Running:", " ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, check=True, env=env)


def read_tsv_rows(tsv_path: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(tsv_path) as f:
        for line in f:
            prompt_id, prompt_text = line.rstrip("\n").split("\t", 1)
            rows.append((prompt_id, prompt_text))
    return rows


def capture_prompt_counts(tsv_path: str, output_dir: pathlib.Path) -> dict[str, int]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_tsv_rows(tsv_path)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_BUILD_BIN + ":" + env.get("LD_LIBRARY_PATH", "")
    counts: dict[str, int] = {}

    n_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(n_batches):
        batch_rows = rows[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
        batch_tsv = WORK_DIR / f"batch_{batch_idx}.tsv"
        batch_output_dir = output_dir / f"batch_{batch_idx}"

        with open(batch_tsv, "w") as f:
            for prompt_id, prompt_text in batch_rows:
                f.write(f"{prompt_id}\t{prompt_text}\n")

        batch_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            BINARY,
            "-m", MODEL,
            "--prompt-file", str(batch_tsv),
            "-o", str(batch_output_dir),
            "-n", str(N_PREDICT),
            "-ngl", str(NGL),
            "-c", str(CTX),
            "-t", str(THREADS),
            "-fa", FLASH_ATTN,
            "--cache-type-k", CACHE_TYPE_K,
            "--cache-type-v", CACHE_TYPE_V,
            "--routing-only",
            "--no-stream",
        ]
        run(cmd, env=env)

        for prompt_dir in sorted(batch_output_dir.iterdir()):
            meta = prompt_dir / "metadata.txt"
            if not prompt_dir.is_dir() or not meta.exists():
                continue
            for line in meta.read_text().splitlines():
                if line.startswith("n_tokens_prompt="):
                    counts[prompt_dir.name] = int(line.split("=", 1)[1].strip())
                    break

    return counts


def pair_counts(counts: dict[str, int]) -> dict[int, dict[str, int]]:
    pairs: dict[int, dict[str, int]] = {}
    for prompt_id, n_tokens in counts.items():
        prefix = prompt_id.split("_", 1)[0]
        pair_num = int(prefix[1:3])
        condition = prefix[3]
        pairs.setdefault(pair_num, {})[condition] = n_tokens
    return pairs


def build_corrections(pairs: dict[int, dict[str, int]]) -> dict[str, dict[str, int]]:
    corrections: dict[str, dict[str, int]] = {}
    for pair_num in sorted(pairs):
        if "A" not in pairs[pair_num] or "B" not in pairs[pair_num]:
            raise RuntimeError(f"Missing A/B prompt counts for pair {pair_num}")
        a_tokens = pairs[pair_num]["A"]
        b_tokens = pairs[pair_num]["B"]
        if a_tokens != b_tokens:
            corrections[str(pair_num)] = {
                "A_tokens": a_tokens,
                "B_tokens": b_tokens,
            }
    return corrections


def verify_pairs_equal(pairs: dict[int, dict[str, int]]) -> list[int]:
    mismatches = []
    for pair_num in sorted(pairs):
        if pairs[pair_num].get("A") != pairs[pair_num].get("B"):
            mismatches.append(pair_num)
    return mismatches


def main() -> None:
    ensure_inputs()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "generate_tsv.py"])
    raw_counts = capture_prompt_counts(TSV, RAW_OUTPUT_DIR)
    raw_pairs = pair_counts(raw_counts)
    corrections = build_corrections(raw_pairs)

    with open(CORRECTIONS_FILE, "w") as f:
        json.dump(corrections, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {CORRECTIONS_FILE} with {len(corrections)} corrected pairs")

    run([sys.executable, "generate_tsv.py", "--corrections", CORRECTIONS_FILE])
    verify_counts = capture_prompt_counts(TSV, VERIFY_OUTPUT_DIR)
    verify_pairs = pair_counts(verify_counts)
    mismatches = verify_pairs_equal(verify_pairs)
    if mismatches:
        raise RuntimeError(f"Token corrections did not fully resolve pair mismatches: {mismatches}")

    print("All A/B pairs verified token-matched after correction.")


if __name__ == "__main__":
    main()
