#!/usr/bin/env python3
"""
GPT-OSS 120B -- 5-Condition Self-Referential Experiment (Prefill-Only).

CAPTURE ONLY -- no entropy computation on instance.
150 prompts run in batches of 15. Raw .npy files preserved for local analysis.

Architecture: 128 experts, top-4 routing, 36 MoE layers (layer 35 excluded).
Gating: softmax (standard MoE gating, NOT sigmoid like Ling-1T).
"""
import os
import pathlib
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
TSV = "prompts_5cond_gptoss.tsv"
OUTPUT_DIR = "output"

N_PREDICT = 0
NGL = int(os.environ.get("NGL", "999"))
CTX = int(os.environ.get("CTX", "4096"))
THREADS = int(os.environ.get("THREADS", "16"))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "off")
CACHE_TYPE_K = os.environ.get("CACHE_TYPE_K", "f16")
CACHE_TYPE_V = os.environ.get("CACHE_TYPE_V", "f16")

CONDITIONS = "ABCDE"
COND_LABELS = {
    "A": "this system",
    "B": "a system",
    "C": "your system",
    "D": "the system",
    "E": "their system",
}
BATCH_SIZE = 15


def run_capture(tsv_file=TSV):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_BUILD_BIN + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        BINARY,
        "-m", MODEL,
        "--prompt-file", tsv_file,
        "-o", OUTPUT_DIR,
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
    print("Running:", " ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, env=env, check=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== GPT-OSS 120B -- 5-Condition Self-Referential Experiment ===")
    print(f"n_predict={N_PREDICT}, ctx={CTX}, ngl={NGL}")
    print(f"flash_attn={FLASH_ATTN}, cache_type_k={CACHE_TYPE_K}, cache_type_v={CACHE_TYPE_V}")
    print(f"150 prompts: 30 pairs x 5 conditions ({', '.join(f'{c}={COND_LABELS[c]}' for c in CONDITIONS)})")
    print("Cal-Manip-Cal sandwich, cold KV cache, Harmony template")
    print("CAPTURE ONLY -- .npy files preserved, analysis runs locally")
    print()

    with open(TSV) as f:
        all_lines = f.readlines()
    n_prompts = len(all_lines)
    n_batches = (n_prompts + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Loaded {n_prompts} prompts, {n_batches} batches of {BATCH_SIZE}")
    print()

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_prompts)
        batch_lines = all_lines[start:end]

        batch_tsv = f"batch_{batch_idx}.tsv"
        with open(batch_tsv, "w") as f:
            f.writelines(batch_lines)

        print(f"=== BATCH {batch_idx+1}/{n_batches}: prompts {start+1}-{end} ===")
        sys.stdout.flush()
        run_capture(tsv_file=batch_tsv)
        os.remove(batch_tsv)

    output_path = pathlib.Path(OUTPUT_DIR)
    captured = sorted([
        d for d in output_path.iterdir()
        if d.is_dir() and (d / "metadata.txt").exists()
    ])
    print(f"\nRun complete. {len(captured)}/{n_prompts} prompts captured.")
    print()
    print("Download output/ and run analysis locally.")


if __name__ == "__main__":
    main()
