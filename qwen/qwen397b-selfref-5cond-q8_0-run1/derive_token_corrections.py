#!/usr/bin/env python3
"""
Derive tokenizer-verified 5-condition correction deltas for qwen397b-selfref-5cond-q8_0-run1.

Two-pass self-verifying workflow:
  1. Generate uncorrected TSV, capture all 150 prompts, read n_tokens_prompt
  2. Write token_corrections.json with per-pair padding deltas
  3. Regenerate corrected TSV with padding applied
  4. Capture all 150 prompts again, verify ALL conditions within each pair match

Hard-aborts if verification fails — do not proceed to the main experiment
with mismatched token counts.
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
    "/workspace/models/Qwen3.5-397B-A17B-Q8_0/Qwen3.5-397B-A17B-Q8_0-00001-of-00010.gguf",
)
BINARY = os.environ.get(
    "CAPTURE_BINARY",
    "/workspace/consciousness-experiment/capture_activations",
)
LLAMA_BUILD_BIN = os.environ.get(
    "LLAMA_BUILD_BIN",
    "/workspace/src/llama.cpp-b8123/build-cuda/bin",
)

TSV = "prompts_selfref_5cond.tsv"
CORRECTIONS_FILE = "token_corrections.json"
WORK_DIR = pathlib.Path(".tmp_token_corrections")
RAW_OUTPUT_DIR = WORK_DIR / "raw_output"
VERIFY_OUTPUT_DIR = WORK_DIR / "verify_output"

CONDITIONS = "ABCDE"
N_PREDICT = 0
NGL = int(os.environ.get("NGL", "999"))
CTX = int(os.environ.get("CTX", "16384"))
THREADS = int(os.environ.get("THREADS", "16"))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "on")
CACHE_TYPE_K = os.environ.get("CACHE_TYPE_K", "q8_0")
CACHE_TYPE_V = os.environ.get("CACHE_TYPE_V", "q8_0")
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
    """Run capture binary on all prompts, return {prompt_id: n_tokens_prompt}.

    Only metadata.txt is needed — router .npy files are deleted immediately
    after reading each batch to avoid accumulating ~GB of unused tensors.
    """
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
            if not prompt_dir.is_dir():
                continue
            meta = prompt_dir / "metadata.txt"
            if meta.exists():
                for line in meta.read_text().splitlines():
                    if line.startswith("n_tokens_prompt="):
                        counts[prompt_dir.name] = int(line.split("=", 1)[1].strip())
                        break
            # Delete router tensors immediately — only metadata was needed.
            # Also cleans up incomplete/malformed prompt dirs that lack metadata.
            router_dir = prompt_dir / "router"
            if router_dir.exists():
                shutil.rmtree(router_dir)

    return counts


def group_by_pair(counts: dict[str, int]) -> dict[int, dict[str, int]]:
    """Group prompt counts by pair number and condition letter."""
    pairs: dict[int, dict[str, int]] = {}
    for prompt_id, n_tokens in counts.items():
        prefix = prompt_id.split("_", 1)[0]  # e.g., "P01A"
        pair_num = int(prefix[1:3])
        condition = prefix[3]
        pairs.setdefault(pair_num, {})[condition] = n_tokens
    return pairs


def build_corrections(pairs: dict[int, dict[str, int]]) -> dict[str, dict[str, int]]:
    """Build correction deltas: for each pair, record token counts if any mismatch."""
    corrections: dict[str, dict[str, int]] = {}
    for pair_num in sorted(pairs):
        cond_counts = pairs[pair_num]
        # Require all 5 conditions present
        for c in CONDITIONS:
            if c not in cond_counts:
                raise RuntimeError(f"Missing condition {c} for pair {pair_num}")
        values = [cond_counts[c] for c in CONDITIONS]
        if len(set(values)) > 1:
            corrections[str(pair_num)] = {
                f"{c}_tokens": cond_counts[c] for c in CONDITIONS
            }
    return corrections


def verify_all_matched(pairs: dict[int, dict[str, int]]) -> list[int]:
    """Return list of pair numbers where conditions have mismatched token counts."""
    mismatches = []
    for pair_num in sorted(pairs):
        cond_counts = pairs[pair_num]
        values = [cond_counts.get(c) for c in CONDITIONS]
        if len(set(values)) > 1:
            mismatches.append(pair_num)
    return mismatches


def main() -> None:
    ensure_inputs()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: capture raw token counts ---
    print("=== PASS 1: Capture raw token counts ===")
    run([sys.executable, "generate_tsv.py"])
    raw_counts = capture_prompt_counts(TSV, RAW_OUTPUT_DIR)
    print(f"Captured {len(raw_counts)} prompt token counts")

    if len(raw_counts) != 150:
        raise RuntimeError(
            f"Expected 150 prompts in pass 1, got {len(raw_counts)}. "
            f"Missing: check batch outputs in {RAW_OUTPUT_DIR}"
        )

    raw_pairs = group_by_pair(raw_counts)
    corrections = build_corrections(raw_pairs)

    with open(CORRECTIONS_FILE, "w") as f:
        json.dump(corrections, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {CORRECTIONS_FILE} with {len(corrections)} corrected pairs")

    if len(corrections) == 0:
        print("All pairs naturally token-matched. No corrections needed.")
        # Still do pass 2 to confirm
    else:
        for pair_key, corr in sorted(corrections.items()):
            counts_str = ", ".join(f"{c}={corr[f'{c}_tokens']}" for c in CONDITIONS)
            print(f"  Pair {pair_key}: {counts_str}")

    # --- Pass 2: regenerate with corrections and verify ---
    print("\n=== PASS 2: Verify corrected token counts ===")
    run([sys.executable, "generate_tsv.py", "--corrections", CORRECTIONS_FILE])
    verify_counts = capture_prompt_counts(TSV, VERIFY_OUTPUT_DIR)
    print(f"Captured {len(verify_counts)} prompt token counts (verification)")

    if len(verify_counts) != 150:
        raise RuntimeError(
            f"Expected 150 prompts in pass 2, got {len(verify_counts)}. "
            f"Missing: check batch outputs in {VERIFY_OUTPUT_DIR}"
        )

    verify_pairs = group_by_pair(verify_counts)
    mismatches = verify_all_matched(verify_pairs)
    if mismatches:
        for p in mismatches:
            counts_str = ", ".join(
                f"{c}={verify_pairs[p].get(c, '?')}" for c in CONDITIONS
            )
            print(f"  FAIL pair {p}: {counts_str}")
        raise RuntimeError(
            f"Token corrections did not fully resolve pair mismatches: {mismatches}. "
            f"DO NOT proceed to the main experiment."
        )

    print("All 30 pairs verified: all 5 conditions token-matched after correction.")

    # Clean up work dir (keep corrections file)
    shutil.rmtree(WORK_DIR)
    print(f"Cleaned up {WORK_DIR}")


if __name__ == "__main__":
    main()
