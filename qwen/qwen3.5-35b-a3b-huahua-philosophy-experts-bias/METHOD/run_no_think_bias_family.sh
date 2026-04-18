#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/consciousness-experiment/experiments/qwen-huahua-philosophy-experts-bias}"
CAPTURE_BINARY="${CAPTURE_BINARY:-/workspace/consciousness-experiment/capture_activations}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf}"
PROMPT_TSV="${PROMPT_TSV:-$ROOT_DIR/domain_specialist_probe_60_no_think.tsv}"
LLAMA_BUILD_BIN="${LLAMA_BUILD_BIN:-/workspace/llama.cpp.new/build/bin}"

CAPTURES_DIR="${CAPTURES_DIR:-$ROOT_DIR/captures}"
LOGS_DIR="${LOGS_DIR:-$ROOT_DIR/non_npy_remote_artifacts/logs}"
N_PREDICT="${N_PREDICT:-1024}"
NGL="${NGL:-999}"
CTX="${CTX:-16384}"
THREADS="${THREADS:-16}"
FLASH_ATTN="${FLASH_ATTN:-on}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
SEED="${SEED:-42}"

declare -a RUN_SPECS=(
  "m8|-8.0"
  "m5|-5.0"
  "m3|-3.0"
  "m2|-2.0"
  "baseline|"
  "p2|2.0"
  "p3|3.0"
  "p5|5.0"
  "p8|8.0"
)

build_bias_spec() {
  local bias="$1"
  if [[ -z "$bias" ]]; then
    return 0
  fi
  printf '114:%s,87:%s,170:%s,68:%s' "$bias" "$bias" "$bias" "$bias"
}

mkdir -p "$CAPTURES_DIR" "$LOGS_DIR"
export LD_LIBRARY_PATH="$LLAMA_BUILD_BIN:${LD_LIBRARY_PATH:-}"

for spec in "${RUN_SPECS[@]}"; do
  slug="${spec%%|*}"
  bias="${spec#*|}"
  now="$(date -u +%Y%m%dT%H%M%SZ)"
  run_id="${now}_philosophy_core_cluster_no_think_${slug}"
  out_dir="$CAPTURES_DIR/$run_id"
  log_path="$LOGS_DIR/${run_id}_capture.log"
  cmd_path="$LOGS_DIR/${run_id}_command.sh"

  mkdir -p "$out_dir"
  bias_spec="$(build_bias_spec "$bias")"

  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "export LD_LIBRARY_PATH=$LLAMA_BUILD_BIN:\${LD_LIBRARY_PATH:-}"
    printf '%q' "$CAPTURE_BINARY"
    printf ' -m %q --prompt-file %q -o %q' "$MODEL_PATH" "$PROMPT_TSV" "$out_dir"
    if [[ -n "$bias_spec" ]]; then
      printf ' --expert-bias %q' "$bias_spec"
    fi
    printf ' -n %q -ngl %q -c %q -t %q -fa %q --cache-type-k %q --cache-type-v %q --seed %q --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 --mirostat 0 --routing-only --no-stream\n' \
      "$N_PREDICT" "$NGL" "$CTX" "$THREADS" "$FLASH_ATTN" "$CACHE_TYPE_K" "$CACHE_TYPE_V" "$SEED"
  } > "$cmd_path"
  chmod +x "$cmd_path"

  echo "=== Running $run_id ==="
  echo "Command: $cmd_path"
  bash "$cmd_path" | tee "$log_path"
done
