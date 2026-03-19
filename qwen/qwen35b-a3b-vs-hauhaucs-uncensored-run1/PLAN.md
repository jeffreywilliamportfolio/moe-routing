# PLAN — Qwen3.5-35B-A3B vs HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive

## Purpose

Compare the official `Qwen/Qwen3.5-35B-A3B` against `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive` under deterministic **prefill-only** conditions to test whether retraining changes MoE routing behavior for:

- 5-condition self-reference prompts (`a / the / this / their / your system`)
- consciousness-centered prompts
- safety-centered prompts

The goal is not output comparison. The goal is to measure whether post-training changed **internal routing**, **entropy/KL structure**, and **expert coalitions** for these prompt classes.

The two models share the same Qwen3.5-35B-A3B MoE family layout: 35B total parameters, ~3B active, 256 experts, and 8 routed plus 1 shared expert per token. 

---

## Models

### Base
- `Qwen/Qwen3.5-35B-A3B`
- Link: `https://huggingface.co/Qwen/Qwen3.5-35B-A3B`

### Retrained
- `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive`
- Link: `https://huggingface.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive`


### Main Suite
- 150 prompts total = 30 prompt families from `prompt-suite.json`, each with 5 variants (`A`/`B`/`C`/`D`/`E`)
- Prompt style: **Cal–Manip–Cal**
- Goal: paired condition comparison under tightly controlled wording variation

## Metrics

### Headline Metrics
- Primary KL: `KL(manip || mean(Cal1_same_prompt))`
- Secondary KL: `KL(manip || global_mean(Cal1_suite))`, where `global_mean(Cal1_suite)` is the all-suite Cal1 mean
- prefill routing entropy
- last-token routing entropy, defined as the last token of the `manip` span

### Coalition Metrics
- top manipulation-selected experts by condition
- manipulation-vs-Cal1 expert selection shifts
- recurrent expert overlap should report both top-N set overlap and frequency-weighted overlap
- recurrent expert overlap across:
  - 5-condition self-reference prompts
  - consciousness prompts
  - safety prompts

### Comparison Targets
- base vs retrained model
- condition vs condition
- consciousness vs safety analysis on the same prompt suite

---

## Routing Assumptions

- Architecture family: Qwen MoE
- Routing type: **softmax**
- Expert count: **256**
- Routed experts per token: **8 + 1 shared**
- Entropy max: `log2(8) = 3.0`
- Reconstruction name: `softmax_then_topk8_renorm`
- Analysis should use the same routing reconstruction path for both models, since the retrained model is based on the official Qwen3.5-35B-A3B family. 

---

## Quant / Precision

- Run both models in **BF16**
- HauhauCS BF16 GGUF size is listed at about **69.4 GB**, so on a single H200 the two models should be run **sequentially**, not in parallel. 

Rationale:
- This experiment is about subtle routing and expert-selection differences
- BF16 is the reference format for highest routing fidelity
- Quantized runs can be used later for screening if needed, but BF16 is the primary experiment format

---

## Hardware

- **2x H200** instance
- `llama.cpp` capture path
- Prefill-only
- No generation sampling
- Deterministic setup

---

## Experimental Constraints

- Same prompt suite for both models
- Same prompt template
- Same tokenizer accounting procedure
- Same context length
- Same capture binary
- Same routing reconstruction code
- Same layer inclusion/exclusion policy
- Same metrics pipeline
- Same file naming / manifest structure

Only model weights should change.

---

## Run Order

### 1. Build / confirm capture binary
Need a `llama.cpp` capture binary that dumps the full routing tensors required for exact reconstruction during prefill.

Before any capture run, execute `--tensor-list` on each model to confirm router tensors are exposed as `ffn_moe_logits-*` with width `256`, then lock the capture + analysis tensor mapping for both models.

Base-model preflight note: the BF16 GGUF exposes `ffn_moe_logits-*` with width `256`, plus `ffn_moe_topk-*`, `ffn_moe_weights-*`, and `ffn_moe_weights_norm-*`. Primary reconstruction remains raw logits first.
Instance note: token preflight uses the deployed GGUF tokenizer via `llama-tokenize` fallback when `llama_cpp-python` is not installed.
Split-GGUF note: if the BF16 model is sharded, create a merged GGUF once for tokenizer preflight and pass it via `TOKENIZER_MODEL_PATH`; capture can still use the shard entrypoint.

Minimum required artifacts per run:
- raw `ffn_moe_logits-*.npy` routing tensors for exact reconstruction
- token counts
- exact prompt text
- per-prompt expert-count summaries
- all recoverable expert-routing data derivable from the raw logits, including selected expert IDs / top-k indices, normalized top-k weights, dense softmax probabilities, and expert selection frequencies
- layer coverage metadata
- prompt manifest / run manifest

Do not delete `.npy` routing tensors on-instance. SCP them to `/Volumes/ExternalSSD` before any cleanup.

### 2. Re-check past Qwen routing analysis
Before the run:
- confirm exact Qwen routing reconstruction assumptions
- confirm tensor naming conventions
- use the prior Qwen completeness filter: include routed layers whose captured row count is at least 50% of the per-prompt median layer row count; exclude shorter/incomplete capture layers and report excluded layers in outputs
- confirm metric code is ready for this architecture family

### 3. Smoke test
Before full runs, do a small validation pass:
- 1 pair from the 5-condition suite
- 1 consciousness prompt
- 1 safety prompt

Purpose:
- verify tensor capture
- verify tokenizer alignment
- verify layer counts
- verify analysis pipeline
- verify output file structure

Smoke note: base-model smoke on `P01`/`P06`/`P22` (15 prompts total) captured all 40 router tensors per prompt; within-family token counts matched exactly (`P01=368`, `P06=367`, `P22=381`); analysis completed for all 15 prompts with valid-layer range `[39, 39]` and excluded-layer union `[39]`.
Full-base token audit note: 28/30 prompt families matched exactly on first full capture; only `P03` (`B=350` vs `351`) and `P11` (`D=364` vs `361`) required padding correction, recorded in `token_corrections.json`.
Corrected-base note: after regenerating `prompts_qwen35b_5cond.tsv` with `token_corrections.json`, the base rerun reached `30/30` token-matched families and produced the corrected 150-prompt base results bundle.
HauhauCS note: tensor preflight matched the base artifact contract (`ffn_moe_logits-*`, width `256`, plus `ffn_moe_topk-*`, `ffn_moe_weights-*`, and `ffn_moe_weights_norm-*`); the corrected 150-prompt HauhauCS run also reached `30/30` token-matched families and produced `results_hauhaucs_qwen35b_a3b_aggressive_prefill.json` with valid-layer range `[39, 39]` and excluded-layer union `[39]`.
Duplicate-base note: a second base-only rerun was executed from a clean duplicate experiment directory with token preflight skipped as redundant for the same corrected TSV/model path; the duplicate reproduced the original corrected base results exactly (`150/150` prompts, identical prompt-level metrics, identical valid/excluded layers, identical top manipulation experts), including `P13A/C/E` and the category-level `experience_probe` expert-`114` result (`9031`, rank `#1`).

### 4. Formal instance setup
Freeze:
- paths
- output directories
- model launch commands
- prompt manifest
- analysis command sequence
- SCP destination layout

Current base-model paths:
- remote experiment dir: `/workspace/experiments/Qwen3.5-35B-A3B-vs-HauhauCS-Qwen3.5-uncensored`
- capture binary: `/workspace/experiments/Qwen3.5-35B-A3B-vs-HauhauCS-Qwen3.5-uncensored/capture_activations`
- base BF16 shard entrypoint: `/workspace/models/Qwen3.5-35B-A3B-BF16/BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf`
- merged tokenizer GGUF: `/workspace/models/merged/Qwen3.5-35B-A3B-BF16-merged.gguf`
- raw output dir: `/workspace/experiments/Qwen3.5-35B-A3B-vs-HauhauCS-Qwen3.5-uncensored/output`
- smoke manifest: `/workspace/experiments/Qwen3.5-35B-A3B-vs-HauhauCS-Qwen3.5-uncensored/smoke.tsv`

### 5. Run base model
Run the full suite on:
- `Qwen/Qwen3.5-35B-A3B`

### 6. Run retrained model
Run the full suite on:
- `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive`

### 7. Compute metrics on-instance
Immediately after each run:
- compute KL to baseline
- compute prefill entropy
- compute last-token entropy
- compute expert coalition summaries
- save compact JSON/TSV summary before transfer

### 8. Transfer raw artifacts
SCP all `.npy` routing tensors and manifests to:
- `/Volumes/ExternalSSD`

Transfer note: the raw backup was staged to `/Volumes/ExternalSSD/llama-eeg-tests/qwen35b-a3b-vs-hauhaucs-2026-03-19` as split tar parts so reassembly/extraction can happen later without blocking the next run.
Final archival note: the duplicate base audit trail, duplicate log, and duplicate reproducibility manifest were also copied off-instance before permanent teardown.

### 9. Cross-model comparison
After both runs are complete:
- compare headline metrics
- compare condition ordering
- compare consciousness/safety coalition overlap
- compare recurrent experts
- compare whether retraining preserved or reshaped the routing basin

---

## Deliverables

### Per Model
- raw routing tensor bundle
- prompt manifest
- metadata / run manifest
- per-prompt metrics JSON
- expert coalition summary
- compact headline summary

### Cross-Model
- base vs retrained comparison table
- condition-by-condition KL / entropy comparison
- consciousness coalition comparison
- safety coalition comparison
- recurrent-expert overlap summary



### Extra Notes From User

- The HF_TOKEN is in the local .env at the project root. Always use this for huggingface-cli or needed HF calls.
- Vast.ai key is set in the local .env at the project root. 
- Instances are meant to be cost efficient but NEVER closed before pulling all metadata, results, capture binary, prompt suite, validations, anything needed besides .npy (unless user explicitly says)
