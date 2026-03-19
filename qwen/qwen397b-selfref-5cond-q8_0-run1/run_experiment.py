#!/usr/bin/env python3
"""
Qwen3.5-397B-A17B Q8_0 -- 5-Condition Self-Referential Experiment (Prefill-Only).

Capture router tensors in batches, compute prompt-level metrics immediately,
write reproducibility artifacts, then delete `.npy` files once analysis is
successfully committed to disk.

Bug-fix hardening (from methodology audit):
  1. STALE DATA: output/ is nuked at start. Hard fail if pre-existing data found
     without --clean flag.
  2. COMPLETE COVERAGE: assert exactly 150 prompt dirs after all batches.
     Hard abort on any skip — do not write results with missing prompts.
  3. TOKEN CORRECTIONS: run derive_token_corrections.py first (separate script).
     This script asserts token_corrections.json exists before starting.

Architecture: 512 experts, top-10 routing, 60 MoE layers (all layers have MoE).
Routing: softmax(512) -> topk(10) -> renormalize. Entropy norm log2(10).
45/60 layers use Gated DeltaNet (linear attention), 15/60 standard attention.
1 shared expert (sigmoid-gated, always active) not captured in ffn_moe_logits.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Dict, List

from analyze_5cond import (
    PROMPT_SUITE,
    RESULTS_FILE,
    TSV,
    analyze_prompt_dir,
    build_output,
    load_prompt_texts,
    write_manifest,
    write_results,
)

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
OUTPUT_DIR = "output"

N_PREDICT = 0
NGL = int(os.environ.get("NGL", "999"))
CTX = int(os.environ.get("CTX", "16384"))
THREADS = int(os.environ.get("THREADS", "16"))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "on")
CACHE_TYPE_K = os.environ.get("CACHE_TYPE_K", "q8_0")
CACHE_TYPE_V = os.environ.get("CACHE_TYPE_V", "q8_0")
BATCH_SIZE = 15
MODEL_NAME = "Qwen3.5-397B-A17B-Q8_0"
N_EXPECTED_PROMPTS = 150


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
    completed = subprocess.run(cmd, env=env, check=False)
    return completed.returncode


def save_partial_results(results: List[dict]):
    inference = {
        "n_predict": N_PREDICT,
        "ngl": NGL,
        "ctx": CTX,
        "flash_attn": FLASH_ATTN,
        "cache_type_k": CACHE_TYPE_K,
        "cache_type_v": CACHE_TYPE_V,
        "sampling": "greedy_argmax",
        "routing_only": True,
    }
    output = build_output(
        results=results,
        inference=inference,
        model_name=MODEL_NAME,
        model_path=MODEL,
    )
    write_results(output, RESULTS_FILE)
    extra_files = [pathlib.Path("experiment.log")] if pathlib.Path("experiment.log").exists() else []
    write_manifest(extra_files=extra_files)


def main():
    # -----------------------------------------------------------
    # BUG FIX 1: Prevent stale data contamination
    # -----------------------------------------------------------
    output_path = pathlib.Path(OUTPUT_DIR)
    if output_path.exists():
        existing = [d for d in output_path.iterdir() if d.is_dir()]
        if existing:
            if "--clean" in sys.argv:
                print(f"WARNING: --clean flag set. Removing {len(existing)} existing prompt dirs in {OUTPUT_DIR}/")
                shutil.rmtree(OUTPUT_DIR)
            else:
                print(f"FATAL: {OUTPUT_DIR}/ contains {len(existing)} existing directories.")
                print(f"This could contaminate results with stale captures from an earlier run.")
                print(f"Either:")
                print(f"  1. Delete {OUTPUT_DIR}/ manually: rm -rf {OUTPUT_DIR}")
                print(f"  2. Re-run with --clean flag: python3 run_experiment.py --clean")
                sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------------
    # Verify prerequisites
    # -----------------------------------------------------------
    if not pathlib.Path("token_corrections.json").exists():
        print("FATAL: token_corrections.json not found.")
        print("Run derive_token_corrections.py first to generate and verify corrections.")
        sys.exit(1)

    if not pathlib.Path(TSV).exists():
        print("FATAL: TSV file not found. Run generate_tsv.py --corrections token_corrections.json first.")
        sys.exit(1)

    with open(PROMPT_SUITE) as f:
        suite = json.load(f)
    cal_paragraph = suite["calibration_paragraph"]
    prompt_texts: Dict[str, str] = load_prompt_texts(TSV)

    print("=== Qwen3.5-397B-A17B Q8_0 -- 5-Condition Self-Referential Experiment ===")
    print(f"model={MODEL_NAME}")
    print(f"n_predict={N_PREDICT}, ctx={CTX}, ngl={NGL}, threads={THREADS}")
    print(f"flash_attn={FLASH_ATTN}, cache_type_k={CACHE_TYPE_K}, cache_type_v={CACHE_TYPE_V}")
    print("Routing reconstruction: softmax(512) -> topk(10) -> renormalize")
    print("Entropy normalization: log2(10)")
    print("Chat template: '<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n'")
    print("On-instance analysis enabled; router tensors deleted after successful metric write")
    print()

    with open(TSV) as f:
        all_lines = f.readlines()
    n_prompts = len(all_lines)
    if n_prompts != N_EXPECTED_PROMPTS:
        print(f"FATAL: Expected {N_EXPECTED_PROMPTS} prompts in TSV, got {n_prompts}")
        sys.exit(1)

    n_batches = (n_prompts + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Loaded {n_prompts} prompts, {n_batches} batches of {BATCH_SIZE}")

    results = []
    seen_ids = set()
    failed_prompts = []

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_prompts)
        batch_lines = all_lines[start:end]
        batch_tsv = f"batch_{batch_idx}.tsv"

        with open(batch_tsv, "w") as f:
            f.writelines(batch_lines)

        print(f"\n=== BATCH {batch_idx + 1}/{n_batches}: prompts {start + 1}-{end} ===")
        sys.stdout.flush()
        return_code = run_capture(tsv_file=batch_tsv)
        os.remove(batch_tsv)
        if return_code != 0:
            print(f"FATAL: Capture binary returned {return_code} on batch {batch_idx + 1}")
            raise SystemExit(return_code)

        prompt_dirs = sorted(
            [
                d for d in pathlib.Path(OUTPUT_DIR).iterdir()
                if d.is_dir() and (d / "metadata.txt").exists() and d.name not in seen_ids
            ],
            key=lambda d: d.name,
        )

        for prompt_dir in prompt_dirs:
            row = analyze_prompt_dir(prompt_dir, prompt_texts, cal_paragraph)
            if row is None:
                print(f"  SKIP {prompt_dir.name}: no valid data")
                failed_prompts.append(prompt_dir.name)
                continue

            results.append(row)
            seen_ids.add(prompt_dir.name)
            kl_text = ""
            if "kl_manip_mean" in row:
                kl_text = f" KL_manip={row['kl_manip_mean']:.6f}"
            kl_cal2_text = ""
            if "kl_cal2_mean" in row:
                kl_cal2_text = f" KL_cal2={row['kl_cal2_mean']:.6f}"
            print(
                f"  {row['id']}: RE={row['prefill_re']:.6f} "
                f"last_tok={row['last_token_re']:.6f}{kl_text}{kl_cal2_text} "
                f"tokens={row['n_prompt_tokens']}"
            )

            # Save after every prompt for crash resilience
            save_partial_results(results)

            # Delete .npy files to free disk space
            router_dir = prompt_dir / "router"
            if router_dir.exists():
                shutil.rmtree(router_dir)

    # -----------------------------------------------------------
    # BUG FIX 2: Assert complete prompt coverage
    # -----------------------------------------------------------
    if len(results) != N_EXPECTED_PROMPTS:
        print(f"\nFATAL: Expected {N_EXPECTED_PROMPTS} analyzed prompts, got {len(results)}")
        if failed_prompts:
            print(f"Failed/skipped prompts: {failed_prompts}")
        print("Results file contains INCOMPLETE data. DO NOT use for analysis.")
        print(f"Partial results saved to {RESULTS_FILE} for debugging only.")
        sys.exit(1)

    save_partial_results(results)
    print(f"\n=== DONE. {len(results)}/{n_prompts} prompts -> {RESULTS_FILE} ===")
    print("All prompts captured and analyzed successfully.")


if __name__ == "__main__":
    main()
