#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/workspace/llama.cpp.new/build/bin:${LD_LIBRARY_PATH:-}
/workspace/consciousness-experiment/capture_activations   -m "/workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf"   --prompt-file "/workspace/consciousness-experiment/experiments/qwen-huahua-expert-identification/prompts/domain_specialist_probe_60_no_think.tsv"   -o "/workspace/consciousness-experiment/experiments/qwen-huahua-expert-identification/captures/20260408T234727Z_domain_specialist_no_think_hauhau"   -n 1024 -ngl 999 -c 16384 -t 16 -fa on   --cache-type-k q8_0 --cache-type-v q8_0   --seed 42 --temp 0 --top-k 1 --top-p 1 --min-p 0   --repeat-penalty 1 --mirostat 0 --routing-only --no-stream
