# Reproduce

This reviewer bundle is reproducible primarily by inspection and local reanalysis, not by one-command environment bootstrap.

Realistic guarantee:

- Yes: reproducible local reanalysis of the included `5cond` and `smoke-test` raw `.npy` files.
- No: a self-contained one-command rerun from scratch.

The goal is simple: a reviewer should be able to read the method, inspect the prompts and analysis code, and reproduce the local reanalysis claims from the included raw `.npy` files.

## Scope

This folder is the researcher-facing subset. The raw `5cond` and `smoke-test` `.npy` files are tracked in this branch with Git LFS under:

- `experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/nothink-5cond-boost-1024-20260323`
- `experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/smoke-20260323b`

Fetch them after cloning with:

```bash
git lfs pull
```

## Method Lock

The canonical routing implementation is [qwen_router.py](/Users/jeffreyshorthill/llama-eeg-tests/experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/qwen_router.py).

Use these rules consistently:

- Dense probabilities: `softmax(logits)` over all `256` experts.
- Sparse routed probabilities: dense softmax, then top-8 selection, then renormalization.
- Normalized entropy: sparse routed entropy divided by `log2(8)`.
- Sparse routed probabilities are used for entropy, expert selection, routed weights, and routing summaries.
- Dense probabilities are used only for explicitly dense metrics such as current `kl_manip_*` and `kl_cal2_*` fields.
- `soft-bias` and `forced-inclusion` are separate intervention regimes and must never be pooled.

## Inspection Path

For documentation-first reproducibility, read these files in order:

1. [PLAN.md](/Users/jeffreyshorthill/llama-eeg-tests/qwen3.5-35b-a3b-huahua-agressive-experts/DOCS/PLAN.md)
2. [RESULTS.md](/Users/jeffreyshorthill/llama-eeg-tests/qwen3.5-35b-a3b-huahua-agressive-experts/DOCS/RESULTS.md)
3. [run_experiment.py](/Users/jeffreyshorthill/llama-eeg-tests/qwen3.5-35b-a3b-huahua-agressive-experts/METHOD/run_experiment.py)
4. [analyze_5cond_condition.py](/Users/jeffreyshorthill/llama-eeg-tests/qwen3.5-35b-a3b-huahua-agressive-experts/METHOD/analyze_5cond_condition.py)
5. [analyze_generation.py](/Users/jeffreyshorthill/llama-eeg-tests/qwen3.5-35b-a3b-huahua-agressive-experts/METHOD/analyze_generation.py)
6. [qwen_router.py](/Users/jeffreyshorthill/llama-eeg-tests/qwen3.5-35b-a3b-huahua-agressive-experts/METHOD/qwen_router.py)
7. [prompt-suite-3band.json](/Users/jeffreyshorthill/llama-eeg-tests/qwen3.5-35b-a3b-huahua-agressive-experts/PROMPTS/prompt-suite-3band.json), [prompt_suite.json](/Users/jeffreyshorthill/llama-eeg-tests/qwen3.5-35b-a3b-huahua-agressive-experts/PROMPTS/prompt_suite.json), and [rubric_markers.json](/Users/jeffreyshorthill/llama-eeg-tests/qwen3.5-35b-a3b-huahua-agressive-experts/PROMPTS/rubric_markers.json)

Then compare the bundled outputs against the claims:

- [RESULTS-NOTHINK-COMPARISON.md](/Users/jeffreyshorthill/llama-eeg-tests/qwen3.5-35b-a3b-huahua-agressive-experts/DOCS/RESULTS-NOTHINK-COMPARISON.md)
- [RESULTS-SHAM-CONTROLS.md](/Users/jeffreyshorthill/llama-eeg-tests/qwen3.5-35b-a3b-huahua-agressive-experts/DOCS/RESULTS-SHAM-CONTROLS.md)
- [20260325-raw-npy-rerun](/Users/jeffreyshorthill/llama-eeg-tests/qwen3.5-35b-a3b-huahua-agressive-experts/DOCS/20260325-raw-npy-rerun)

## Expected Comparison Targets

These are the main outputs a reviewer should compare:

- `DOCS/20260325-raw-npy-rerun/5cond/analysis-*.json`
- `DOCS/20260325-raw-npy-rerun/smoke/analysis.json`

Expected high-level matches:

- Smoke reanalysis should be stable except for path text such as `run_dir`.
- The reviewable 5-condition subset should match overlapping prompt records up to floating-point noise only.

## What Is Required For A True Rerun

A true rerun is not fully self-contained in git. It requires:

- the HauhauCS GGUF model file
- the correct capture binary or llama.cpp build
- matching runtime flags
- matching seed and decode settings
- a host with sufficient GPU memory

At minimum, preserve these facts for rerun attempts:

- model path or model file hash
- binary path or binary hash
- seed
- context size
- max new tokens
- decode flags
- intervention mode, expert, and bias
- prompt TSV hash

This bundle is sufficient for local review and local reanalysis. It is not sufficient for a full environment rebuild from scratch.
