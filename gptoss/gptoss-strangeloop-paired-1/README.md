# GPT-OSS Strange-Loop Paired Run

## What This Folder Is

This folder contains the reviewed GPT-OSS-120B strange-loop paired control experiment.

- Model: GPT-OSS-120B `mxfp4`
- Design: 30 paired prompts, `A` versus `B`, for 60 prompts total
- Condition contrast: deictic phrasing like `this paradox` or `this loop` versus generic phrasing like `a paradox` or `a loop`
- Inference mode: prefill-only (`n_predict=0`), cold KV cache, greedy argmax
- Routing reconstruction: `topk_raw_logit_then_softmax_4`
- Architecture assumptions: 128 experts, top-4 routing, 36 MoE layers with layer `35` excluded from analysis
- Purpose: control for the self-reference experiment by using recursive content that is not about the model

## Reviewed Status

The checked-in result bundle looks usable and internally consistent at the high-impact level.

- `results_strangeloop_paired_prefill_gptoss.json` contains 60 per-prompt rows
- All 30 `A/B` pairs are complete
- `token_mismatch_pairs` is empty
- Every analyzed prompt uses 35 valid MoE layers with excluded layer set `(35,)`
- The routing reconstruction matches GPT-OSS top-4 routing

This is the GPT-OSS strange-loop control run that should be treated as the valid artifact in this repo.

## Important Caveat

The checked-in artifact looks valid. A fresh rerun is still not fully fail-closed.

`run_experiment.py` hard-fails batch capture itself with `check=True`, which is good, but after capture it only warns if it finds fewer than 60 prompt directories and can still continue to analysis and print `DONE`. That means an incomplete rerun can still produce a result file that looks finished.

For any rerun, manually verify at minimum:

- `output/` contains all 60 prompt directories
- each prompt directory has `metadata.txt`, `prompt_tokens.json`, and router `.npy` files
- the final analyzed JSON still has 60 per-prompt rows and 30 complete `A/B` pairs
- `token_mismatch_pairs` remains empty unless you intentionally changed prompt construction

## Main Files

- `results_strangeloop_paired_prefill_gptoss.json`: reviewed result bundle with per-prompt metrics and paired tests
- `RESULTS.md`: narrative summary of the control result
- `run_experiment.py`: capture plus analysis runner
- `gptoss_router.py`: GPT-OSS routing reconstruction logic
- `generate_tsv.py`: prompt TSV generator
- `derive_token_corrections.py`: helper used to derive exact token corrections when needed
- `prompt_suite.json`: prompt definitions
- `prompts_strangeloop_paired.tsv`: capture input TSV
- `capture_activations.cpp`: capture binary source used by this workflow

## Running It

The script expects a compiled `capture_activations` binary and a local GPT-OSS GGUF model. By default it reads:

- `MODEL_PATH=/workspace/models/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf`
- `CAPTURE_BINARY=/workspace/consciousness-experiment/capture_activations`
- `LLAMA_BUILD_BIN=/workspace/src/llama.cpp-b8123/build-cuda/bin`

Common optional overrides:

- `NGL`
- `CTX`
- `THREADS`
- `FLASH_ATTN`
- `CACHE_TYPE_K`
- `CACHE_TYPE_V`

Typical flow:

```bash
python3 run_experiment.py
```

## Notes On Interpretation

`RESULTS.md` summarizes the control result: small but statistically reliable whole-prompt routing differences, null last-token effect, and cautious interpretation of absolute KL scale. For external readers, the key distinction is between the checked-in artifact, which looks internally consistent, and fresh reruns, which still need manual completeness checks.
