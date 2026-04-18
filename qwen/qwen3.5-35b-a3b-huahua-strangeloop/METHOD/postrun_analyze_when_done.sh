#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/consciousness-experiment/experiments/qwen3.5-35b-a3b-huahua-strangeloop}"
LAUNCHER_PID="${LAUNCHER_PID:-}"
CAPTURE_DIR="${CAPTURE_DIR:-$ROOT/captures/20260410T000413Z_strangeloop_no_think_prefill}"
PROMPT_JSON="${PROMPT_JSON:-$ROOT/prompts/qwen_strangeloop_paired_1_prompts.json}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
RESULTS_STEM="${RESULTS_STEM:-results_strangeloop_paired_20260410T000413Z}"
MODEL_LABEL="${MODEL_LABEL:-Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0}"

mkdir -p "$RESULTS_DIR"

if [[ -n "$LAUNCHER_PID" ]]; then
  while kill -0 "$LAUNCHER_PID" 2>/dev/null; do
    sleep 30
  done
fi

python3 - <<'PY'
try:
    import scipy  # noqa: F401
    print("scipy already present")
except Exception:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
PY

python3 "$ROOT/scripts/analyze_strangeloop_paired.py" \
  --capture-dir "$CAPTURE_DIR" \
  --prompt-json "$PROMPT_JSON" \
  --report "$RESULTS_DIR/${RESULTS_STEM}.md" \
  --results-json "$RESULTS_DIR/${RESULTS_STEM}.json" \
  --model-label "$MODEL_LABEL"
