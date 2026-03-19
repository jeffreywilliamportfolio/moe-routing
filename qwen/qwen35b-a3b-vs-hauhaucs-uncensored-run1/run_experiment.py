#!/usr/bin/env python3
"""
Run the Qwen3.5-35B-A3B comparison suite.

Capture router tensors in batches, compute prompt-level metrics immediately,
and preserve raw `.npy` routing artifacts for later transfer off-instance.
"""
import json
import os
import pathlib
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
    "/workspace/models/Qwen3.5-35B-A3B-BF16/BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf",
)
BINARY = os.environ.get(
    "CAPTURE_BINARY",
    str(pathlib.Path(__file__).parent / "capture_activations"),
)
LLAMA_BUILD_BIN = os.environ.get(
    "LLAMA_BUILD_BIN",
    "/workspace/llama.cpp/build/bin",
)
_HERE = pathlib.Path(__file__).parent
OUTPUT_DIR = str(_HERE / "output")

N_PREDICT = 0
NGL = int(os.environ.get("NGL", "999"))
CTX = int(os.environ.get("CTX", "16384"))
THREADS = int(os.environ.get("THREADS", "16"))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "on")
CACHE_TYPE_K = os.environ.get("CACHE_TYPE_K", "q8_0")
CACHE_TYPE_V = os.environ.get("CACHE_TYPE_V", "q8_0")
BATCH_SIZE = 15
MODEL_NAME = os.environ.get("MODEL_TAG", "qwen35b_a3b_base")
CORRECTIONS_FILE = _HERE / "token_corrections.json"


def run_token_preflight():
    cmd = [sys.executable, str(_HERE / "token_verify.py")]
    env = os.environ.copy()
    if CORRECTIONS_FILE.exists():
        cmd.extend(["--corrections", str(CORRECTIONS_FILE)])
    print("=== Token preflight ===")
    print("Running:", " ".join(cmd))
    sys.stdout.flush()
    return subprocess.run(cmd, env=env, check=False).returncode


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
    extra_files = [_HERE / "experiment.log"] if (_HERE / "experiment.log").exists() else []
    write_manifest(extra_files=extra_files)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Guard against stale output from a prior aborted run contaminating results.
    existing = [
        d for d in pathlib.Path(OUTPUT_DIR).iterdir()
        if d.is_dir() and (d / "metadata.txt").exists()
    ] if pathlib.Path(OUTPUT_DIR).exists() else []
    if existing:
        print(
            f"ERROR: output/ already contains {len(existing)} prompt dir(s). "
            "Delete or move output/ before re-running to avoid stale data contamination."
        )
        raise SystemExit(1)

    if not pathlib.Path(TSV).exists():
        print("FATAL: TSV not found. Run: python3 generate_tsv.py")
        raise SystemExit(1)

    preflight_rc = run_token_preflight()
    if preflight_rc != 0:
        print(f"FATAL: token preflight failed with status {preflight_rc}")
        raise SystemExit(preflight_rc)

    with open(PROMPT_SUITE) as f:
        suite = json.load(f)
    cal_paragraph = suite["calibration_paragraph"]
    prompt_texts: Dict[str, str] = load_prompt_texts(TSV)

    print("=== Qwen3.5-35B-A3B Comparison Experiment ===")
    print(f"model={MODEL_NAME}")
    print(f"n_predict={N_PREDICT}, ctx={CTX}, ngl={NGL}, threads={THREADS}")
    print(f"flash_attn={FLASH_ATTN}, cache_type_k={CACHE_TYPE_K}, cache_type_v={CACHE_TYPE_V}")
    print("Routing reconstruction: softmax(256) -> topk(8) -> renormalize")
    print("Entropy normalization: log2(8)")
    print("Chat template: '<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n<think>\\n'")
    print("Content: prompt-suite categories from prompt-suite.json")
    print("On-instance analysis enabled; raw router tensors preserved for later SCP")
    print()

    with open(TSV) as f:
        all_lines = f.readlines()
    n_prompts = len(all_lines)
    n_batches = (n_prompts + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Loaded {n_prompts} prompts, {n_batches} batches of {BATCH_SIZE}")

    results = []
    seen_ids = set()
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_prompts)
        batch_lines = all_lines[start:end]
        batch_tsv = str(_HERE / f"batch_{batch_idx}.tsv")

        with open(batch_tsv, "w") as f:
            f.writelines(batch_lines)

        print(f"\n=== BATCH {batch_idx + 1}/{n_batches}: prompts {start + 1}-{end} ===")
        sys.stdout.flush()
        return_code = run_capture(tsv_file=batch_tsv)
        os.remove(batch_tsv)
        if return_code != 0:
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
                continue

            results.append(row)
            seen_ids.add(prompt_dir.name)
            kl_text = ""
            if "kl_manip_mean" in row:
                kl_text = f" KL={row['kl_manip_mean']:.6f}"
            print(
                f"  {row['id']}: RE={row['prefill_re']:.6f} "
                f"LT={row['last_token_re']:.6f}{kl_text} tok={row['n_prompt_tokens']}"
            )

            save_partial_results(results)

    save_partial_results(results)
    print(f"\n=== DONE. {len(results)}/{n_prompts} prompts -> {RESULTS_FILE} ===")


if __name__ == "__main__":
    main()
