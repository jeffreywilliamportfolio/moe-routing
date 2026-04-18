#!/usr/bin/env python3
"""Render the think-mode single-prompt Expert 114 run family.

This is a prepare-only helper. It does not launch anything.
It prints a compact summary and concrete shell command templates for:
  - baseline think
  - Expert 114 bias: -3.0, -5.0, -8.0, +2.0, +3.0, +5.0
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts" / "single_prompt_emergent_intelligence_think_family.json"


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    prompt_tsv = manifest["prompt_tsv"]
    model_path = manifest["model_path"]
    binary_path = manifest["binary_path"]
    common_args = manifest["common_args"]

    print(f"Experiment: {manifest['experiment']}")
    print(f"Prompt TSV: {prompt_tsv}")
    print(f"Prompt ID: {manifest['prompt_id']}")
    print(f"Mode: {manifest['mode']}")
    print()
    print("Run matrix:")
    for run in manifest["runs"]:
        bias = run["expert_bias"] if run["expert_bias"] is not None else "none"
        print(f"  - {run['slug']}: {run['label']} | expert_bias={bias}")

    print()
    print("Launch templates:")
    for run in manifest["runs"]:
        out_dir = f'/workspace/consciousness-experiment/experiments/qwen-huahua-6cond-moe-manips/captures/${{RUN_ID}}'
        print(f"# {run['label']}")
        print("export LD_LIBRARY_PATH=/workspace/llama.cpp.new/build/bin:$LD_LIBRARY_PATH")
        print(f"NOW=$(date -u +%Y%m%dT%H%M%SZ)")
        print(f'RUN_ID="${{NOW}}_{run["slug"]}"')
        print(f"OUT_DIR={out_dir}")
        cmd_parts = [
            shell_quote(binary_path),
            "-m",
            shell_quote(model_path),
            "--prompt-file",
            shell_quote(f"/workspace/consciousness-experiment/experiments/qwen-huahua-6cond-moe-manips/{prompt_tsv}"),
            "-o",
            '"$OUT_DIR"',
        ]
        if run["expert_bias"] is not None:
            cmd_parts.extend(["--expert-bias", shell_quote(run["expert_bias"])])
        cmd_parts.extend(shell_quote(arg) for arg in common_args)
        print(" \\\n  ".join(cmd_parts))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
