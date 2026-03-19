#!/usr/bin/env python3
"""
One-prompt DeepSeek tensor verification for the DS3.1 5-condition run.

This script intentionally spends a single prompt to verify the capture binary
produces `ffn_moe_logits-*` tensors with the expected expert dimension and a
non-zero token axis before the full run starts.
"""
import argparse
import json
import os
import pathlib
import subprocess
import tempfile

import numpy as np

N_ROUTED_EXPERTS = 256
CHAT_PREFIX = "<｜User｜>"
CHAT_SUFFIX = "<｜Assistant｜>"


def build_prompt(prompt_suite_path):
    with open(prompt_suite_path) as f:
        suite = json.load(f)
    pair = suite["pairs"][0]
    text = f"{suite['calibration_paragraph']} {pair['A']} {suite['calibration_paragraph']}"
    text = text.replace("\n", " ").replace("\t", " ")
    return f"P01A_verify\t{CHAT_PREFIX}{text}{CHAT_SUFFIX}\n"


def main():
    parser = argparse.ArgumentParser(description="Verify DeepSeek capture tensor names and shapes")
    parser.add_argument("--prompt-suite", default="prompt_suite.json")
    parser.add_argument("--tensor-names-out", default="tensor_names.txt")
    args = parser.parse_args()

    model = os.environ.get(
        "MODEL_PATH",
        "/workspace/models/DeepSeek-V3-0324-UD-Q2_K_XL/UD-Q2_K_XL/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf",
    )
    binary = os.environ.get("CAPTURE_BINARY", "/workspace/consciousness-experiment/capture_activations")
    llama_build_bin = os.environ.get("LLAMA_BUILD_BIN", "/workspace/src/llama.cpp-b8123/build-cuda/bin")
    ngl = int(os.environ.get("NGL", "999"))
    ctx = int(os.environ.get("CTX", "4096"))
    threads = int(os.environ.get("THREADS", "16"))

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = llama_build_bin + ":" + env.get("LD_LIBRARY_PATH", "")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        prompt_path = tmp_path / "verify.tsv"
        prompt_path.write_text(build_prompt(args.prompt_suite))

        cmd = [
            binary,
            "-m", model,
            "--prompt-file", str(prompt_path),
            "-o", str(tmp_path / "output"),
            "-n", "0",
            "-ngl", str(ngl),
            "-c", str(ctx),
            "-t", str(threads),
            "--routing-only",
            "--no-stream",
        ]
        print("Running verification prompt:", " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)

        router_dir = tmp_path / "output" / "P01A_verify" / "router"
        files = sorted(router_dir.glob("ffn_moe_logits-*.npy"))
        if not files:
            raise SystemExit("No ffn_moe_logits tensors were captured")

        lines = []
        for fp in files:
            shape = np.load(str(fp)).shape
            if len(shape) != 2:
                raise SystemExit(f"{fp.name}: expected rank-2 tensor, got {shape}")
            if shape[1] != N_ROUTED_EXPERTS:
                raise SystemExit(f"{fp.name}: expected shape[1] == {N_ROUTED_EXPERTS}, got {shape}")
            if shape[0] <= 0:
                raise SystemExit(f"{fp.name}: expected non-zero token rows, got {shape}")
            lines.append(f"{fp.name}\t{shape[0]}\t{shape[1]}")

    out_path = pathlib.Path(args.tensor_names_out)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Verified {len(lines)} tensors with shape[1] == {N_ROUTED_EXPERTS} and shape[0] > 0")
    print(f"Wrote tensor manifest to {out_path}")


if __name__ == "__main__":
    main()
