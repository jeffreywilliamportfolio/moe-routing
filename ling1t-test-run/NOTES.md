# Ling-1T 5-Condition System Run Notes

This folder is the clean rerun bundle for the Ling-1T 5-condition `system` experiment.

The five manipulated conditions are:
- `this system`
- `a system`
- `your system`
- `the system`
- `their system`

The current prompt suite contains `150` prompts total (`30` prompt groups x `5` conditions) and is configured for prefill-only capture.

It intentionally excludes archived results and logs. Only the files needed to generate prompts, verify token counts, run capture, and analyze a fresh run belong here.

## Included Files

- `prompt_suite.json`: 30 paired manipulation paragraphs with Ling-1T-correct architecture references
- `generate_tsv.py`: builds the wrapped Ling-1T TSV input
- `prompts_5cond_system.tsv`: generated prompt TSV for the current prompt suite
- `token_verify.py`: checks token-count equality across the five conditions
- `run_experiment.py`: batched prefill-only capture runner
- `analyze_local.py`: Ling-specific sigmoid top-8 routing entropy analysis
- `setup_instance.sh`: bootstrap script for a remote capture host
- `capture_activations.cpp`: capture binary source staged into llama.cpp

## Ling-1T Assumptions Used Here

- Architecture family: `BailingMoeV2`
- Hidden layers: `80`
- Routed MoE layers expected in capture: `76`
- Experts per routed layer: `256`
- Active experts per token: `8`
- Shared experts: `1`
- Hidden size: `8192`
- Gating function: sigmoid

## Q3_K_S VRAM Requirement

- The `unsloth/Ling-1T-GGUF` repo contains `Q3_K_S/` (9 shards) — there is no `UD-Q3_K_S` variant in that repo
- The setup script downloads `Q3_K_S/*` from `unsloth/Ling-1T-GGUF`
- GGUF size on disk: about `402 GB` across `9` shards
- Recorded full-offload device-memory fit: about `415,500 MiB` total GPU memory
- Recorded fit used `8x NVIDIA H200` with `n_gpu_layers=999`
- Per-GPU projected usage in that run:
  - `CUDA0`: `33,793 MiB`
  - `CUDA1`: `59,768 MiB`
  - `CUDA2`: `54,371 MiB`
  - `CUDA3`: `54,371 MiB`
  - `CUDA4`: `54,371 MiB`
  - `CUDA5`: `54,371 MiB`
  - `CUDA6`: `54,371 MiB`
  - `CUDA7`: `50,080 MiB`

Practical takeaway: plan for roughly `406 GiB` total VRAM for this quant with full GPU offload, and treat `8x H200 141GB` as the known-good configuration in this repo.

## Prompt Hygiene

- Legacy architecture text has been removed.
- The prompt suite now refers to `256` experts, `8` selected experts, `76` routed layers, and `8192`-dimensional representations where architecture-specific text is used.
- No old `RESULTS.md`, `experiment.log`, or results JSON is carried into this folder.

## Tensor Preflight

- Run `--list-tensors` before the real capture run.
- Treat this as a discovery-and-validation step, not just a confirmation step.
- The live binary on the target host is the source of truth.
- The live H200 preflight captured in this folder confirms routed layers `4..79` and these Ling tensor families:
  - `ffn_moe_logits-N`
  - `ffn_moe_probs_biased-N`
  - `ffn_moe_probs_masked-N`
  - `ffn_moe_topk-N`
  - `ffn_moe_weights_norm-N`
  - `ffn_moe_group_topk-N`
- Expected routed-layer tensor shape for the full-width families is `[N_TOKENS, 256]`.
- Expected routed-layer count is about `76`.
- Expected gating metadata in the load log is `expert_gating_func = sigmoid`.

Preflight must verify all of the following before the main run:
- router tensor names actually exposed by the live binary
- routed-layer tensor count and suffix range
- tensor shape `[N_TOKENS, 256]`
- model metadata indicates sigmoid gating
- captured tensors are raw gate logits, not already-normalized probabilities
- whether Ling exposes any additional bias or corrected-score tensor that affects which `8` experts are selected
- whether `ffn_moe_weights_norm-N` and `ffn_moe_topk-N` are present for exact top-8 reconstruction

The current capture path is:
- capture tensor families `ffn_moe_logits-N`, `ffn_moe_probs_biased-N`, `ffn_moe_probs_masked-N`, `ffn_moe_topk-N`, `ffn_moe_weights_norm-N`, and `ffn_moe_group_topk-N`
- save files under `output/<prompt>/router/`
- sort one file per routed layer by numeric suffix

The current analysis path is:
- prefer `ffn_moe_weights_norm-N.npy` when present
- compute entropy directly on the captured final normalized top-8 weights
- otherwise fall back to `ffn_moe_logits-N.npy` and analyze as `sigmoid -> top-8 mask -> renormalize -> entropy`

Important limitation:
- when `ffn_moe_weights_norm-N` is available, the headline RE metric reflects the exact captured final normalized top-8 weights rather than a logits-only proxy
- `ffn_moe_probs_biased-N`, `ffn_moe_probs_masked-N`, and `ffn_moe_group_topk-N` are preserved for fidelity checks and future deeper reconstruction work
- if the live binary or source inspection shows Ling uses any additional tensor beyond these saved families in final expert selection, capture and analysis must be updated again before treating the run as exact

## Reporting Scope

- the current Ling routing-entropy metric is a within-Ling metric
- `analyze_local.py` computes entropy after top-8 masking and normalizes by `log2(8)`
- this makes Ling prompt comparisons, condition contrasts, and pairwise effect sizes internally meaningful within Ling runs
- do not directly compare Ling absolute RE magnitudes to DeepSeek or Qwen RE magnitudes derived from full-distribution softmax analyses
- if cross-model reporting is needed later, add a second explicitly labeled Ling metric rather than silently treating the current top-8 metric as equivalent to the DeepSeek/Qwen scale

If the live binary exposes a different Ling router tensor name, update `is_router_tensor()` in [capture_activations.cpp](/Users/jeffreyshorthill/llama-eeg-tests/experiments/ling1t-5cond/capture_activations.cpp) before the main run.

If the live binary appears to expose post-sigmoid or already-normalized routing values instead of raw logits, stop and adjust [analyze_local.py](/Users/jeffreyshorthill/llama-eeg-tests/experiments/ling1t-5cond/analyze_local.py) before running the full experiment.

If the live binary or model code indicates that expert selection uses an additional tensor beyond `ffn_moe_weights_norm`, `ffn_moe_topk`, `ffn_moe_probs_biased`, `ffn_moe_probs_masked`, and `ffn_moe_group_topk`, stop and patch both [capture_activations.cpp](/Users/jeffreyshorthill/llama-eeg-tests/experiments/ling1t-5cond/capture_activations.cpp) and [analyze_local.py](/Users/jeffreyshorthill/llama-eeg-tests/experiments/ling1t-5cond/analyze_local.py) again before the main run.

## Expected Workflow

1. Run `python3 generate_tsv.py`
2. Run `--list-tensors` and discover the actual Ling router tensors and any bias/correction tensors used for expert selection
3. Run `python3 token_verify.py` on the target Ling-1T GGUF
4. Run `python3 run_experiment.py`
5. Run `python3 analyze_local.py --output-dir output/ --prompt-suite prompt_suite.json`
