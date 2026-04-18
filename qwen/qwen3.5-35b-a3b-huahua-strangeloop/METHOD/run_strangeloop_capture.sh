#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/consciousness-experiment/experiments/qwen3.5-35b-a3b-huahua-strangeloop}"
CAPTURE_BINARY="${CAPTURE_BINARY:-/workspace/consciousness-experiment/capture_activations}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf}"
PROMPT_TSV="${PROMPT_TSV:-$ROOT_DIR/prompts/qwen_strangeloop_paired_1_no_think.tsv}"
LLAMA_BUILD_BIN="${LLAMA_BUILD_BIN:-/workspace/llama.cpp.new/build/bin}"

RUN_LABEL="${RUN_LABEL:-strangeloop_no_think}"
CAPTURES_DIR="${CAPTURES_DIR:-$ROOT_DIR/captures}"
LOGS_DIR="${LOGS_DIR:-$ROOT_DIR/non_npy_remote_artifacts/logs}"
N_PREDICT="${N_PREDICT:-0}"
NGL="${NGL:-999}"
CTX="${CTX:-16384}"
THREADS="${THREADS:-16}"
FLASH_ATTN="${FLASH_ATTN:-on}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
SEED="${SEED:-42}"
EXPERT_BIAS="${EXPERT_BIAS:-}"

mkdir -p "$CAPTURES_DIR" "$LOGS_DIR"
export LD_LIBRARY_PATH="$LLAMA_BUILD_BIN:${LD_LIBRARY_PATH:-}"

now="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${now}_${RUN_LABEL}"
out_dir="$CAPTURES_DIR/$run_id"
log_path="$LOGS_DIR/${run_id}_capture.log"
cmd_path="$LOGS_DIR/${run_id}_command.sh"

mkdir -p "$out_dir"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "export LD_LIBRARY_PATH=$LLAMA_BUILD_BIN:\${LD_LIBRARY_PATH:-}"
  printf '%q' "$CAPTURE_BINARY"
  printf ' -m %q --prompt-file %q -o %q' "$MODEL_PATH" "$PROMPT_TSV" "$out_dir"
  if [[ -n "$EXPERT_BIAS" ]]; then
    printf ' --expert-bias %q' "$EXPERT_BIAS"
  fi
  printf ' -n %q -ngl %q -c %q -t %q -fa %q --cache-type-k %q --cache-type-v %q --seed %q --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 --mirostat 0 --routing-only --no-stream\n' \
    "$N_PREDICT" "$NGL" "$CTX" "$THREADS" "$FLASH_ATTN" "$CACHE_TYPE_K" "$CACHE_TYPE_V" "$SEED"
} > "$cmd_path"
chmod +x "$cmd_path"

echo "=== Running $run_id ==="
echo "Command: $cmd_path"
bash "$cmd_path" | tee "$log_path"
