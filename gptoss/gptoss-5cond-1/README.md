# GPT-OSS 5-Condition Self-Referential Run

## What This Folder Is

This folder contains the reviewed GPT-OSS-120B five-condition self-referential routing experiment.

- Model: GPT-OSS-120B `mxfp4`
- Design: 30 prompt groups x 5 determiner conditions = 150 prompts
- Conditions: `A=this system`, `B=a system`, `C=your system`, `D=the system`, `E=their system`
- Inference mode: prefill-only (`n_predict=0`), cold KV cache, greedy argmax
- Routing reconstruction: `topk_raw_logit_then_softmax_4`
- Architecture assumptions: 128 experts, top-4 routing, 36 MoE layers with layer `35` excluded from analysis

## Reviewed Status

The checked-in result bundle looks usable and internally consistent at the high-impact level.

- `results_5cond_prefill_gptoss.json` contains 150 per-prompt rows
- All 30 five-condition prompt groups are complete
- Every analyzed prompt uses 35 valid MoE layers with excluded layer set `(35,)`
- The routing reconstruction matches GPT-OSS top-4 routing rather than an older dense-128 reconstruction variant

This is the GPT-OSS five-condition run that should be treated as the valid artifact in this repo.

## Important Caveat

The checked-in artifact looks valid. The rerun path is not fail-closed.

`run_experiment.py` launches capture with `subprocess.run(..., check=False)` and only prints how many prompt directories were found at the end. If capture fails partway through, the script can still exit without hard-failing the run. Do not treat a fresh rerun as valid just because the script finished.

For any rerun, manually verify at minimum:

- `output/` contains all 150 prompt directories
- each prompt directory has `metadata.txt`, `prompt_tokens.json`, and router `.npy` files
- the final analyzed JSON still has 150 per-prompt rows and 30 complete `ABCDE` groups

## Main Files

- `results_5cond_prefill_gptoss.json`: reviewed result bundle with per-prompt metrics and pairwise tests
- `RESULTS.md`: narrative summary of the outcome and statistical tests
- `run_experiment.py`: capture runner
- `analyze_5cond.py`: local paired analysis across all five conditions
- `analyze_final.py`: final summary helper
- `gptoss_router.py`: GPT-OSS routing reconstruction logic
- `prompt_suite.json`: prompt definitions
- `prompts_5cond_gptoss.tsv`: capture input TSV
- `capture_activations.cpp`: capture binary source used by this workflow

## Running It

The capture script expects a compiled `capture_activations` binary and a local GPT-OSS GGUF model. By default it reads:

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
python3 analyze_5cond.py
python3 analyze_final.py
```

## Notes On Interpretation

The checked-in `RESULTS.md` is the best concise summary of the observed effects. The important distinction for readers is between the checked-in artifact, which looks internally consistent, and fresh reruns, which still require manual completeness checks because the capture path is not fail-closed.
