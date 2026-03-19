#!/usr/bin/env python3
"""
DeepSeek V3.1 -- 5-Condition Self-Referential Experiment (Prefill-Only).

Capture only. Analysis runs after capture so the instance can emit the JSON
summary before artifacts are copied back locally.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

from analysis_common import write_region_boundaries

MODEL = os.environ.get(
    "MODEL_PATH",
    "/workspace/models/DeepSeek-V3-0324-UD-Q2_K_XL/UD-Q2_K_XL/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf",
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
PROMPT_SUITE = "prompt_suite.json"
OUTPUT_DIR = "output"
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

N_PREDICT = 0
NGL = int(os.environ.get("NGL", "999"))
CTX = int(os.environ.get("CTX", "4096"))
THREADS = int(os.environ.get("THREADS", "16"))

CONDITIONS = "ABCDE"
COND_LABELS = {
    "A": "this system",
    "B": "a system",
    "C": "your system",
    "D": "the system",
    "E": "their system",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the DeepSeek V3.1 5-condition capture experiment."
    )
    parser.add_argument(
        "--tsv",
        default=TSV,
        help=f"Prompt TSV to capture (default: {TSV})",
    )
    parser.add_argument(
        "--prompt-suite",
        default=PROMPT_SUITE,
        help=f"Prompt suite JSON used for calibration metadata (default: {PROMPT_SUITE})",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output directory for captured prompts (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


def run_capture(tsv_file, output_dir):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_BUILD_BIN + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        BINARY,
        "-m", MODEL,
        "--prompt-file", tsv_file,
        "-o", output_dir,
        "-n", str(N_PREDICT),
        "-ngl", str(NGL),
        "-c", str(CTX),
        "-t", str(THREADS),
        "--routing-only",
        "--no-stream",
    ]
    print("Running:", " ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, env=env, check=True)


def resolve_input_path(path_str):
    path = pathlib.Path(path_str)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def resolve_output_dir(path_str):
    path = pathlib.Path(path_str)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def inspect_capture_outputs(output_dir):
    output_path = pathlib.Path(output_dir)
    prompt_dirs = sorted(d for d in output_path.iterdir() if d.is_dir())
    with_metadata = [d for d in prompt_dirs if (d / "metadata.txt").exists()]
    with_tokens = [d for d in with_metadata if (d / "prompt_tokens.json").exists()]
    return prompt_dirs, with_metadata, with_tokens


def main():
    args = parse_args()
    tsv_path = resolve_input_path(args.tsv)
    prompt_suite_path = resolve_input_path(args.prompt_suite)
    output_dir = resolve_output_dir(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=== DeepSeek V3.1 -- 5-Condition Self-Referential Experiment ===")
    print(f"n_predict={N_PREDICT}, ctx={CTX}, ngl={NGL}")
    print(f"Expected full design: 150 prompts = 30 pairs x 5 conditions ({', '.join(f'{c}={COND_LABELS[c]}' for c in CONDITIONS)})")
    print("Cal-Manip-Cal sandwich, cold KV cache, DeepSeek chat template")
    print("Gating reconstruction downstream: sigmoid + noaux_tc-style grouped top-k (bias-free approximation)")
    print("CAPTURE ONLY -- analysis runs after capture")
    print(f"Prompt TSV: {tsv_path}")
    print(f"Prompt suite: {prompt_suite_path}")
    print(f"Output dir: {output_dir}")
    print()

    with tsv_path.open() as f:
        n_prompts = sum(1 for _ in f)
    print(f"Loaded {n_prompts} prompts in a single TSV")
    if n_prompts != 150:
        print("Run mode  : smoke / partial capture")
    else:
        print("Run mode  : full 150-prompt capture")
    print()

    run_capture(str(tsv_path), str(output_dir))

    output_path = output_dir
    _, captured, with_tokens = inspect_capture_outputs(output_dir)

    with prompt_suite_path.open() as f:
        suite = json.load(f)
    n_with_boundaries = write_region_boundaries(output_path, suite["calibration_paragraph"])

    print(f"\nRun complete. {len(captured)}/{n_prompts} prompts captured.")
    print(f"prompt_tokens.json present for {len(with_tokens)}/{len(captured)} captured prompts.")
    print(f"Exact token boundaries written to metadata for {n_with_boundaries} prompts.")
    if len(with_tokens) != len(captured):
        print("WARNING: some captured prompts are missing prompt_tokens.json; exact-boundary analysis will fail.")
    print()
    print("Run analysis on-instance:")
    print(f"  python3 analyze_local.py --output-dir {output_dir} --prompt-suite {prompt_suite_path}")
    print(f"  python3 analyze_kl_baseline_sigmoid.py --output-dir {output_dir} --prompt-suite {prompt_suite_path}")


if __name__ == "__main__":
    main()
