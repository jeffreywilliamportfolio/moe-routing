#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/workspace/llama.cpp.new/build/bin:${LD_LIBRARY_PATH:-}
/workspace/consciousness-experiment/capture_activations -m /workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf --prompt-file /workspace/consciousness-experiment/experiments/qwen-huahua-philosophy-experts-bias/domain_specialist_probe_60_no_think.tsv -o /workspace/consciousness-experiment/experiments/qwen-huahua-philosophy-experts-bias/captures/20260409T194923Z_philosophy_core_cluster_no_think_m2 --expert-bias 114:-2.0\,87:-2.0\,170:-2.0\,68:-2.0 -n 1024 -ngl 999 -c 16384 -t 16 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --seed 42 --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 --mirostat 0 --routing-only --no-stream
