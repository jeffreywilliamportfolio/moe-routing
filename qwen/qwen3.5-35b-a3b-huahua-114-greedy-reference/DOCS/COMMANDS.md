# Commands

Commands executed for the deterministic greedy reference run. Connection endpoints, SSH ports,
key material, API keys, Hugging Face tokens, and `.env` contents are intentionally omitted.
Remote endpoints are written as `<remote>`.

## Vast Instance Selection

```bash
set -a; source /Volumes/ExternalSSD/sae-tests/.env; set +a
python3 ~/.codex/skills/vastai-cli-operator/scripts/run_vast.py show user --raw
python3 ~/.codex/skills/vastai-cli-operator/scripts/run_vast.py search offers --raw \
  'gpu_name=RTX_5090 num_gpus=2 reliability>0.98 rentable=True rented=False disk_space>=500 inet_down>100 inet_up>20 direct_port_count>=2'
python3 ~/.codex/skills/vastai-cli-operator/scripts/run_vast.py create instance 33920449 \
  --image nvidia/cuda:12.8.1-devel-ubuntu22.04 \
  --disk 500 --ssh --direct --cancel-unavail --raw
python3 ~/.codex/skills/vastai-cli-operator/scripts/run_vast.py logs <failed-instance-id>
python3 ~/.codex/skills/vastai-cli-operator/scripts/run_vast.py stop instance <failed-instance-id> --raw
```

The user-selected offer failed to start with a Vast CDI GPU device injection error. A replacement
2x RTX 5090 / 500 GB offer was rented and used for all captures below.

```bash
python3 ~/.codex/skills/vastai-cli-operator/scripts/run_vast.py create instance 30709121 \
  --image nvidia/cuda:12.8.1-devel-ubuntu22.04 \
  --disk 500 --ssh --direct --cancel-unavail --raw
python3 ~/.codex/skills/vastai-cli-operator/scripts/run_vast.py show instance <capture-instance-id> --raw
python3 ~/.codex/skills/vastai-cli-operator/scripts/run_vast.py ssh-url <capture-instance-id>
python3 ~/.codex/skills/vastai-cli-operator/scripts/run_vast.py scp-url <capture-instance-id>
```

## Remote Hygiene And Staging

```bash
ssh <remote> 'set -euo pipefail; if command -v supervisorctl >/dev/null 2>&1; then supervisorctl status || true; supervisorctl stop llama || true; supervisorctl status || true; else echo supervisorctl_not_found; fi'
ssh <remote> 'set -euo pipefail; mkdir -p /workspace; df -h / /workspace; nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader'
rsync -az --delete \
  --exclude 'raw/' --exclude 'analysis/' --exclude 'captures/' \
  /Volumes/ExternalSSD/sae-tests/qwen3.5-35b-a3b-huahua-residual-analysis/ \
  <remote>:/workspace/residual-analysis/
```

## Bootstrap

The remote bootstrap wrapper read `HF_TOKEN` from stdin, wrote it only to remote unversioned
environment files, set `HF_HOME=/workspace/.hf_home`, disabled Xet, enabled `hf_transfer`, then
ran the repository bootstrap script.

```bash
ssh <remote> 'cat > /workspace/run_bootstrap_remote.sh && chmod 700 /workspace/run_bootstrap_remote.sh'
printf '%s\n' "$HF_TOKEN" | ssh <remote> '/workspace/run_bootstrap_remote.sh'
```

Remote-only recovery steps were required before the successful bootstrap:

```bash
ssh <remote> 'set -euo pipefail; apt-get update; apt-get install -y git build-essential cmake ninja-build pkg-config python3-pip'
ssh <remote> 'set -euo pipefail; python3 -m pip install -U pip hf_transfer'
ssh <remote> 'set -euo pipefail; kill <stalled-hf-download-pid> || true; rm -rf /workspace/models/qwen35-hauhau-q8/.cache/huggingface/download'
printf '%s\n' "$HF_TOKEN" | ssh <remote> '/workspace/run_bootstrap_remote.sh'
```

## Environment Verification

```bash
ssh <remote> 'set -euo pipefail
date -u +%Y-%m-%dT%H:%M:%SZ
. /etc/os-release; printf "%s %s\n" "$NAME" "$VERSION_ID"
nvcc --version | tail -n 1
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
git -C /workspace/llama.cpp.new rev-parse HEAD
sha256sum /workspace/residual-analysis/bin/capture_activations
sha256sum /workspace/residual-analysis/bin/capture_residuals
/workspace/residual-analysis/bin/capture_residuals --help >/tmp/capture_help.txt 2>&1
stat -c "%F %s bytes %n" /workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf
sha256sum /workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf
'
```

## Single-Prompt Greedy Capture

```bash
ssh <remote> 'set -euo pipefail
/workspace/residual-analysis/bin/capture_residuals \
  -m /workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf \
  --prompt-file /workspace/residual-analysis/single_prompt_processing_hum_no_think.tsv \
  -o /workspace/residual-analysis/captures/20260418T160353Z_single_prompt_processing_hum_no_think_gen_n1024_greedy \
  -n 1024 -c 2048 -ngl 99 \
  --tensor-split 1,1 --main-gpu 0 \
  --no-stream --seed 0 \
  --temp 0 --top-k 1
'
rsync -az -e 'ssh <key-and-port-options>' \
  <remote>:/workspace/residual-analysis/captures/20260418T160353Z_single_prompt_processing_hum_no_think_gen_n1024_greedy \
  /Volumes/ExternalSSD/sae-tests/runs/greedy_reference_20260418T160353Z/single_prompt/raw/
```

Manifest result:

```text
1 prompt processed, 1 succeeded, 0 failed
S01_processing_hum_probe: 117 prompt tokens, 1024 generated tokens
Captured tensors: ffn_moe_logits-{13,14,15}, attn_post_norm-{13,14,15}
```

## Single-Prompt Analysis

```bash
RAW=/Volumes/ExternalSSD/sae-tests/runs/greedy_reference_20260418T160353Z/single_prompt/raw/20260418T160353Z_single_prompt_processing_hum_no_think_gen_n1024_greedy
ANALYSIS=/Volumes/ExternalSSD/sae-tests/runs/greedy_reference_20260418T160353Z/single_prompt/analysis/20260418T160353Z_single_prompt_processing_hum_no_think_gen_n1024_greedy
python3 scripts/step1_extract_contexts.py --raw-dir "$RAW" --analysis-dir "$ANALYSIS"
python3 scripts/step2_decile_sample.py --analysis-dir "$ANALYSIS" --samples-per-decile 10 --seed 0
python3 scripts/step3_build_labeler_prompt.py --analysis-dir "$ANALYSIS"
```

Validation:

```text
context_trim_mode: trim_at_literal_imend
hauhau_imend_trim_found: true
hauhau_imend_trim_idx: 108
n_tokens_generated_raw: 1024
n_tokens_generated_trimmed: 108
WSQ_identity_residual_max: 1.3877787807814457e-17
L14 generation_trimmed W_mean/S_mean/Q_mean: 0.08337904608085038 / 0.6944444444444444 / 0.12006582635642453
```

## Heldout Greedy Capture

```bash
ssh <remote> 'set -euo pipefail
/workspace/residual-analysis/bin/capture_residuals \
  -m /workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf \
  --prompt-file /workspace/residual-analysis/heldout_prompts.tsv \
  -o /workspace/residual-analysis/captures/heldout_20260418T160353Z_greedy \
  -n 256 -c 2048 -ngl 99 \
  --tensor-split 1,1 --main-gpu 0 \
  --no-stream --seed 0 \
  --temp 0 --top-k 1
'
rsync -az -e 'ssh <key-and-port-options>' \
  <remote>:/workspace/residual-analysis/captures/heldout_20260418T160353Z_greedy \
  /Volumes/ExternalSSD/sae-tests/runs/greedy_reference_20260418T160353Z/heldout/raw/
```

Manifest result:

```text
20 prompts processed, 20 succeeded, 0 failed
All prompts generated 256 tokens
Captured tensors per prompt: ffn_moe_logits-{13,14,15}, attn_post_norm-{13,14,15}
```

## Heldout Analysis

```bash
RAW=/Volumes/ExternalSSD/sae-tests/runs/greedy_reference_20260418T160353Z/heldout/raw/heldout_20260418T160353Z_greedy
ANALYSIS=/Volumes/ExternalSSD/sae-tests/runs/greedy_reference_20260418T160353Z/heldout/analysis/heldout_20260418T160353Z_greedy
python3 scripts/analyze_heldout.py \
  --raw-dir "$RAW" \
  --classes-tsv heldout_classes.tsv \
  --analysis-dir "$ANALYSIS"
```

Outputs:

```text
heldout_stats.tsv
heldout_timeseries_top4.png
fire mean-of-means:   0.068089 +/- 0.034584
nofire mean-of-means: 0.003249 +/- 0.006195
ratio fire/nofire:    20.955x
Cohen's d:            2.61
range overlap:        YES
```

## Cleanup

```bash
python3 ~/.codex/skills/vastai-cli-operator/scripts/run_vast.py stop instance <capture-instance-id> --raw
python3 ~/.codex/skills/vastai-cli-operator/scripts/run_vast.py show instances --raw
```

The capture instance was stopped after the heldout raw bundle and analysis outputs were local.
