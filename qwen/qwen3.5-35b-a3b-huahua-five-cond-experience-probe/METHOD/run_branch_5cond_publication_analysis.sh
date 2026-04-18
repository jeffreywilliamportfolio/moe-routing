#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
BRANCH_NAME="${BRANCH_NAME:-qwen-hauhau-5cond-smoke-only}"

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 CONDITION_DIR [OUTPUT_JSON] [OUTPUT_MD]" >&2
  exit 1
fi

CONDITION_DIR="$1"
OUTPUT_JSON="${2:-$ROOT_DIR/results/$(basename "$CONDITION_DIR").branch-5cond-analysis.json}"
OUTPUT_MD="${3:-$ROOT_DIR/results/$(basename "$CONDITION_DIR").branch-5cond-analysis.md}"

TSV_PATH="${TSV_PATH:-$ROOT_DIR/prompts/qwen_5cond_experience_probe_no_think.tsv}"
PROMPT_SUITE_PATH="${PROMPT_SUITE_PATH:-$ROOT_DIR/prompts/qwen_5cond_experience_probe_prompt_suite.json}"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

git -C "$REPO_ROOT" show "${BRANCH_NAME}:scripts/qwen_router.py" > "$TMP_DIR/qwen_router.py"
git -C "$REPO_ROOT" show "${BRANCH_NAME}:scripts/analyze_5cond_condition.py" > "$TMP_DIR/analyze_5cond_condition.py"
chmod +x "$TMP_DIR/analyze_5cond_condition.py"

python3 "$TMP_DIR/analyze_5cond_condition.py" \
  --condition-dir "$CONDITION_DIR" \
  --tsv-path "$TSV_PATH" \
  --prompt-suite "$PROMPT_SUITE_PATH" \
  --output-json "$OUTPUT_JSON" \
  --output-md "$OUTPUT_MD"
