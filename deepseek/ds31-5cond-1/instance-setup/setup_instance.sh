#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$SCRIPT_DIR}"

LLAMA_REPO_URL="${LLAMA_REPO_URL:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_TAG="${LLAMA_TAG:-b8123}"
LLAMA_DIR="${LLAMA_DIR:-/workspace/src/llama.cpp-b8123}"
LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-$LLAMA_DIR/build-cuda}"

CAPTURE_SOURCE="${CAPTURE_SOURCE:-$EXPERIMENT_DIR/capture_activations.cpp}"
CAPTURE_TARGET_NAME="${CAPTURE_TARGET_NAME:-llama-capture-activations}"
CAPTURE_BINARY_DEST="${CAPTURE_BINARY_DEST:-/workspace/consciousness-experiment/capture_activations}"

MODEL_ROOT="${MODEL_ROOT:-/workspace/models/DeepSeek-V3-0324-UD-Q2_K_XL}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/UD-Q2_K_XL}"
MODEL_MANIFEST="${MODEL_MANIFEST:-download_manifest_deepseek_ud_q2_xl.txt}"
MODEL_WORKERS="${MODEL_WORKERS:-6}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-6}"

DOWNLOAD_LOG="${DOWNLOAD_LOG:-$EXPERIMENT_DIR/download.log}"
BUILD_LOG="${BUILD_LOG:-$EXPERIMENT_DIR/build.log}"
VERIFY_LOG="${VERIFY_LOG:-$EXPERIMENT_DIR/tensor_verify.log}"
GENERATE_LOG="${GENERATE_LOG:-$EXPERIMENT_DIR/generate_tsv.log}"
TOKEN_VERIFY_PASS1_LOG="${TOKEN_VERIFY_PASS1_LOG:-$EXPERIMENT_DIR/token_verify_pass1.log}"
TOKEN_VERIFY_FIX_LOG="${TOKEN_VERIFY_FIX_LOG:-$EXPERIMENT_DIR/token_verify_fix.log}"
TOKEN_VERIFY_PASS2_LOG="${TOKEN_VERIFY_PASS2_LOG:-$EXPERIMENT_DIR/token_verify_pass2.log}"
EXPERIMENT_LOG="${EXPERIMENT_LOG:-$EXPERIMENT_DIR/experiment.log}"
ANALYSIS_LOG="${ANALYSIS_LOG:-$EXPERIMENT_DIR/analysis.log}"

THREADS="${THREADS:-$(nproc)}"
NGL="${NGL:-999}"
CTX="${CTX:-4096}"

DOWNLOAD_SCREEN_NAME="${DOWNLOAD_SCREEN_NAME:-ds31_5cond_download}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

echo "== DS3.1 5cond setup =="
echo "experiment_dir: $EXPERIMENT_DIR"
echo "llama_dir: $LLAMA_DIR"
echo "llama_tag: $LLAMA_TAG"
echo "model_dir: $MODEL_DIR"
echo "ngl: $NGL"
echo "ctx: $CTX"
echo

mkdir -p /workspace/src /workspace/models /workspace/consciousness-experiment "$MODEL_DIR"
cd "$EXPERIMENT_DIR"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  git \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  libcurl4-openssl-dev \
  libcublas-dev-12-9 \
  screen \
  python3-pip

python3 -m pip install --upgrade numpy scipy

require_file "$CAPTURE_SOURCE"
require_file "$EXPERIMENT_DIR/$MODEL_MANIFEST"
require_file "$EXPERIMENT_DIR/run_wget_workers.sh"
require_file "$EXPERIMENT_DIR/wget_worker.sh"

if screen -list | grep -q "[.]$DOWNLOAD_SCREEN_NAME"; then
  echo "download screen session already exists: $DOWNLOAD_SCREEN_NAME"
else
  echo "== start model download in detached screen =="
  screen -dmS "$DOWNLOAD_SCREEN_NAME" bash -lc "
    set -euo pipefail
    cd '$EXPERIMENT_DIR'
    chmod +x ./run_wget_workers.sh ./wget_worker.sh
    ./run_wget_workers.sh '$MODEL_MANIFEST' '$MODEL_WORKERS' '$MODEL_DIR' 'download_logs' | tee '$DOWNLOAD_LOG'
    echo DOWNLOAD COMPLETE | tee -a '$DOWNLOAD_LOG'
  "
fi

echo "== llama.cpp checkout and build =="
{
  if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    git clone "$LLAMA_REPO_URL" "$LLAMA_DIR"
  fi
  git -C "$LLAMA_DIR" fetch --tags --force
  git -C "$LLAMA_DIR" checkout "$LLAMA_TAG"

  mkdir -p "$LLAMA_DIR/examples/capture_activations"
  cp "$CAPTURE_SOURCE" "$LLAMA_DIR/examples/capture_activations/capture_activations.cpp"
  cat > "$LLAMA_DIR/examples/capture_activations/CMakeLists.txt" <<'EOF'
set(TARGET llama-capture-activations)
add_executable(${TARGET} capture_activations.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
EOF

  if ! grep -q 'add_subdirectory(capture_activations)' "$LLAMA_DIR/examples/CMakeLists.txt"; then
    python3 - "$LLAMA_DIR/examples/CMakeLists.txt" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
needle = "    add_subdirectory(gen-docs)\n"
insert = needle + "    add_subdirectory(capture_activations)\n"
if "add_subdirectory(capture_activations)" not in text:
    if needle not in text:
        raise SystemExit(f"Could not find insertion point in {path}")
    text = text.replace(needle, insert, 1)
    path.write_text(text)
PY
  fi

  cmake -S "$LLAMA_DIR" -B "$LLAMA_BUILD_DIR" \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_EXAMPLES=ON

  cmake --build "$LLAMA_BUILD_DIR" --target "$CAPTURE_TARGET_NAME" -j"$THREADS"
  cp "$LLAMA_BUILD_DIR/bin/$CAPTURE_TARGET_NAME" "$CAPTURE_BINARY_DEST"
  chmod +x "$CAPTURE_BINARY_DEST"
  git -C "$LLAMA_DIR" rev-parse HEAD > "$EXPERIMENT_DIR/build_commit.txt"
  md5sum "$CAPTURE_BINARY_DEST" | awk '{print $1}' > "$EXPERIMENT_DIR/binary_md5.txt"
} | tee "$BUILD_LOG"

echo "== wait for model shards =="
while true; do
  shard_count="$(find "$MODEL_DIR" -maxdepth 1 -name 'DeepSeek-V3-0324-UD-Q2_K_XL-*.gguf' | wc -l | tr -d ' ')"
  part_count="$(find "$MODEL_DIR" -maxdepth 1 -name '*.part' | wc -l | tr -d ' ')"
  if [[ "$shard_count" == "$EXPECTED_SHARDS" && "$part_count" == "0" ]]; then
    break
  fi
  echo "waiting: shards=$shard_count/$EXPECTED_SHARDS part_files=$part_count"
  sleep 30
done

if ! tail -n 20 "$DOWNLOAD_LOG" | grep -q "DOWNLOAD COMPLETE"; then
  echo "download finished shard checks but missing DOWNLOAD COMPLETE marker" >&2
  exit 1
fi

export CAPTURE_BINARY="$CAPTURE_BINARY_DEST"
export MODEL_PATH="$MODEL_DIR/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf"
export LLAMA_BUILD_BIN="$LLAMA_BUILD_DIR/bin"
export LD_LIBRARY_PATH="$LLAMA_BUILD_BIN:${LD_LIBRARY_PATH:-}"
export NGL
export CTX
export THREADS

echo "== tensor verification gate =="
python3 verify_tensors.py --prompt-suite prompt_suite.json --tensor-names-out tensor_names.txt | tee "$VERIFY_LOG"

echo "== generate TSV =="
python3 generate_tsv.py | tee "$GENERATE_LOG"

echo "== token verify pass 1 =="
python3 token_verify.py | tee "$TOKEN_VERIFY_PASS1_LOG"

if grep -q "MISMATCH" "$TOKEN_VERIFY_PASS1_LOG"; then
  echo "== token verify fix =="
  python3 token_verify.py --fix | tee "$TOKEN_VERIFY_FIX_LOG"
fi

echo "== token verify pass 2 =="
python3 token_verify.py | tee "$TOKEN_VERIFY_PASS2_LOG"

if grep -q "MISMATCH" "$TOKEN_VERIFY_PASS2_LOG"; then
  echo "token verification still reports mismatches after fix" >&2
  exit 1
fi

echo "== run capture =="
python3 run_experiment.py | tee "$EXPERIMENT_LOG"

echo "== analyze results =="
python3 analyze_local.py --output-dir output/ --prompt-suite prompt_suite.json | tee "$ANALYSIS_LOG"

echo "== complete =="
