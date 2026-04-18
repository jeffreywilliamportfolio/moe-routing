#!/usr/bin/env python3
"""Run the HauhauCS Expert-114 basin-steering experiment or its smoke matrix."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime

from analyze_generation import analyze_run

_HERE = pathlib.Path(__file__).parent
PROMPT_SUITE = _HERE / "prompt-suite-3band.json"
RUBRIC = _HERE / "rubric_markers.json"
SHAM_CONTROLS = _HERE / "sham_controls.json"
FULL_TSV = _HERE / "prompts_full.tsv"
SMOKE_TSV = _HERE / "prompts_smoke.tsv"
RUNS_DIR = _HERE / "runs"

MODEL = os.environ.get(
    "MODEL_PATH",
    "/workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf",
)
BINARY = os.environ.get(
    "CAPTURE_BINARY",
    "/workspace/consciousness-experiment/capture_activations",
)
LLAMA_BUILD_BIN = os.environ.get(
    "LLAMA_BUILD_BIN",
    "/workspace/llama.cpp.new/build/bin",
)
NGL = int(os.environ.get("NGL", "999"))
CTX = int(os.environ.get("CTX", "16384"))
THREADS = int(os.environ.get("THREADS", "16"))
FLASH_ATTN = os.environ.get("FLASH_ATTN", "on")
CACHE_TYPE_K = os.environ.get("CACHE_TYPE_K", "q8_0")
CACHE_TYPE_V = os.environ.get("CACHE_TYPE_V", "q8_0")
SEED = int(os.environ.get("SEED", "12345"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))


def ensure_inputs() -> dict:
    if (
        not FULL_TSV.exists() or
        not SMOKE_TSV.exists() or
        PROMPT_SUITE.stat().st_mtime > FULL_TSV.stat().st_mtime or
        PROMPT_SUITE.stat().st_mtime > SMOKE_TSV.stat().st_mtime
    ):
        subprocess.run([sys.executable, str(_HERE / "generate_tsv.py")], check=True)
    if not SHAM_CONTROLS.exists():
        subprocess.run([sys.executable, str(_HERE / "mine_sham_controls.py")], check=True)
    return json.loads(SHAM_CONTROLS.read_text())


def build_conditions(sham_controls: dict, smoke: bool) -> list[dict]:
    shams = sham_controls["selected"]
    if smoke:
        return [
            {"label": "baseline", "mode": "none", "expert": None, "bias": None},
            {"label": "expert_114_soft_bias_1.0", "mode": "soft-bias", "expert": 114, "bias": 1.0},
            {"label": "expert_114_forced_inclusion", "mode": "forced-inclusion", "expert": 114, "bias": None},
            {"label": f"expert_{shams[0]}_soft_bias_1.0", "mode": "soft-bias", "expert": shams[0], "bias": 1.0},
            {"label": f"expert_{shams[0]}_forced_inclusion", "mode": "forced-inclusion", "expert": shams[0], "bias": None},
            {"label": f"expert_{shams[1]}_soft_bias_1.0", "mode": "soft-bias", "expert": shams[1], "bias": 1.0},
            {"label": f"expert_{shams[1]}_forced_inclusion", "mode": "forced-inclusion", "expert": shams[1], "bias": None},
        ]

    levels = [0.25, 0.5, 1.0, 2.0, 3.0]
    conditions = [{"label": "baseline", "mode": "none", "expert": None, "bias": None}]
    for expert in [114, *shams]:
        for bias in levels:
            conditions.append({
                "label": f"expert_{expert}_soft_bias_{bias}",
                "mode": "soft-bias",
                "expert": expert,
                "bias": bias,
            })
        conditions.append({
            "label": f"expert_{expert}_forced_inclusion",
            "mode": "forced-inclusion",
            "expert": expert,
            "bias": None,
        })
    return conditions


def run_condition(tsv_path: pathlib.Path, run_dir: pathlib.Path, condition: dict) -> int:
    output_dir = run_dir / "capture" / condition["label"]
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_BUILD_BIN + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        BINARY,
        "-m", MODEL,
        "--prompt-file", str(tsv_path),
        "-o", str(output_dir),
        "-n", str(MAX_NEW_TOKENS),
        "-ngl", str(NGL),
        "-c", str(CTX),
        "-t", str(THREADS),
        "-fa", FLASH_ATTN,
        "--cache-type-k", CACHE_TYPE_K,
        "--cache-type-v", CACHE_TYPE_V,
        "--seed", str(SEED),
        "--temp", "0",
        "--top-k", "1",
        "--top-p", "1",
        "--min-p", "0",
        "--repeat-penalty", "1",
        "--mirostat", "0",
        "--routing-only",
        "--no-stream",
    ]

    if condition["mode"] != "none":
        cmd.extend([
            "--intervention-mode", condition["mode"],
            "--intervention-expert", str(condition["expert"]),
        ])
        if condition["mode"] == "soft-bias":
            cmd.extend(["--intervention-bias", str(condition["bias"])])

    manifest = {
        "label": condition["label"],
        "command": cmd,
        "mode": condition["mode"],
        "expert": condition["expert"],
        "bias": condition["bias"],
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    (output_dir / "condition_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Running:", " ".join(cmd))
    sys.stdout.flush()
    completed = subprocess.run(cmd, env=env, check=False)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run the 3-prompt smoke matrix instead of the full 24-prompt matrix.")
    parser.add_argument("--run-name", default=None, help="Optional run directory name.")
    parser.add_argument("--tsv-path", default=None, help="Optional explicit TSV prompt file override.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip post-run analysis generation.")
    args = parser.parse_args()

    sham_controls = ensure_inputs()
    tsv_path = pathlib.Path(args.tsv_path) if args.tsv_path else (SMOKE_TSV if args.smoke else FULL_TSV)
    conditions = build_conditions(sham_controls, smoke=args.smoke)

    RUNS_DIR.mkdir(exist_ok=True)
    run_name = args.run_name or datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "capture").mkdir(exist_ok=True)

    run_manifest = {
        "run_name": run_name,
        "smoke": args.smoke,
        "prompt_suite": str(PROMPT_SUITE),
        "rubric": str(RUBRIC),
        "sham_controls": sham_controls,
        "tsv_path": str(tsv_path),
        "skip_analysis": args.skip_analysis,
        "conditions": conditions,
        "model": MODEL,
        "binary": BINARY,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "ctx": CTX,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))

    for condition in conditions:
        rc = run_condition(tsv_path, run_dir, condition)
        if rc != 0:
            raise SystemExit(rc)

    if args.skip_analysis:
        print("Skipping analysis (--skip-analysis).")
        return

    analysis = analyze_run(
        run_dir=run_dir,
        prompt_suite_path=PROMPT_SUITE,
        rubric_path=RUBRIC,
        focus_expert=114,
        sham_controls_path=SHAM_CONTROLS,
    )
    (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(f"Wrote {run_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
