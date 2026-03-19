#!/bin/bash
set -euo pipefail

# Qwen3.5-397B-A17B Q8_0 instance setup
# 4x H200, llama.cpp b8123, capture_activations binary

echo "=== Setting up Qwen Q8_0 experiment instance ==="

# 1. Install dependencies
apt-get update && apt-get install -y cmake git build-essential wget

# 2. Clone and build llama.cpp b8123
mkdir -p /workspace/src
cd /workspace/src
if [ ! -d "llama.cpp-b8123" ]; then
    git clone --branch b8123 --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp-b8123
fi
cd llama.cpp-b8123

# Copy capture_activations source
mkdir -p examples/capture_activations
cp /workspace/experiment-qwen397b-selfref-5cond-q8_0-run1/capture_activations.cpp examples/capture_activations/capture_activations.cpp

# Create CMakeLists.txt for capture_activations
cat > examples/capture_activations/CMakeLists.txt << 'CEOF'
set(TARGET llama-capture-activations)
add_executable(${TARGET} capture_activations.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
CEOF

# Build with CUDA
mkdir -p build-cuda && cd build-cuda
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90" -DLLAMA_CURL=OFF
cmake --build . --target llama-capture-activations -j$(nproc)

# Install binary
mkdir -p /workspace/consciousness-experiment
cp bin/llama-capture-activations /workspace/consciousness-experiment/capture_activations
chmod +x /workspace/consciousness-experiment/capture_activations

echo "Binary ready: /workspace/consciousness-experiment/capture_activations"
echo "LD_LIBRARY_PATH=/workspace/src/llama.cpp-b8123/build-cuda/bin"

# 3. Download Q8_0 model (10 shards, ~422GB)
mkdir -p /workspace/models/Qwen3.5-397B-A17B-Q8_0
cd /workspace/models/Qwen3.5-397B-A17B-Q8_0

echo "=== Downloading Qwen3.5-397B-A17B Q8_0 (10 shards) ==="
for i in $(seq -w 1 10); do
    SHARD="Qwen3.5-397B-A17B-Q8_0-000${i}-of-00010.gguf"
    if [ ! -f "$SHARD" ]; then
        echo "Downloading $SHARD..."
        wget -q --show-progress "https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF/resolve/main/Q8_0/${SHARD}"
    else
        echo "Already have $SHARD"
    fi
done

echo ""
echo "=== Setup complete ==="
echo "Model shards: $(ls -1 /workspace/models/Qwen3.5-397B-A17B-Q8_0/*.gguf | wc -l)/10"
echo ""
echo "Next steps:"
echo "  cd /workspace/experiment-qwen397b-selfref-5cond-q8_0-run1"
echo "  export LD_LIBRARY_PATH=/workspace/src/llama.cpp-b8123/build-cuda/bin:\$LD_LIBRARY_PATH"
echo "  python3 derive_token_corrections.py"
echo "  python3 run_experiment.py --clean"
