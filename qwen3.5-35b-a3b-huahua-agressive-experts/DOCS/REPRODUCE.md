# Reproduce

This reviewer bundle is reproducible primarily by inspection and local reanalysis, not by one-command environment bootstrap.

Realistic guarantee:

- Yes: reproducible local reanalysis of the bundled rerun summaries, analysis JSON, and focus reports under `DOCS/`.
- No: a self-contained one-command rerun from scratch.

The goal is simple: a reviewer should be able to read the method, inspect the prompts and analysis code, and reproduce the local reanalysis claims from the bundled artifacts in this folder.

## Scope

This folder is the reviewer-facing subset. The branch bundles reanalysis outputs and focus reports under:

- [DOCS/20260325-raw-npy-rerun/](20260325-raw-npy-rerun/)
- [DOCS/20260327-focus-layers/](20260327-focus-layers/)
- [DOCS/20260327-focus-coalition/](20260327-focus-coalition/)
- [DOCS/20260327-focus-tokens/](20260327-focus-tokens/)

Historical full raw run directories referenced in earlier notes are not included in this repo snapshot.

## Method Lock

The canonical routing implementation is [qwen_router.py](../METHOD/qwen_router.py).

Use these rules consistently:

- Dense probabilities: `softmax(logits)` over all `256` experts.
- Sparse routed probabilities: dense softmax, then top-8 selection, then renormalization.
- Normalized entropy: sparse routed entropy divided by `log2(8)`.
- Sparse routed probabilities are used for entropy, expert selection, routed weights, and routing summaries.
- Dense probabilities are used only for explicitly dense metrics such as current `kl_manip_*` and `kl_cal2_*` fields.
- `soft-bias` and `forced-inclusion` are separate intervention regimes and must never be pooled.

## Inspection Path

For documentation-first reproducibility, read these files in order:

1. [PLAN.md](PLAN.md)
2. [RESULTS.md](RESULTS.md)
3. [run_experiment.py](../METHOD/run_experiment.py)
4. [analyze_5cond_condition.py](../METHOD/analyze_5cond_condition.py)
5. [analyze_generation.py](../METHOD/analyze_generation.py)
6. [qwen_router.py](../METHOD/qwen_router.py)
7. [prompt-suite-3band.json](../PROMPTS/prompt-suite-3band.json), [prompt_suite.json](../PROMPTS/prompt_suite.json), and [rubric_markers.json](../PROMPTS/rubric_markers.json)

Then compare the bundled outputs against the claims:

- [RESULTS-NOTHINK-COMPARISON.md](RESULTS-NOTHINK-COMPARISON.md)
- [RESULTS-SHAM-CONTROLS.md](RESULTS-SHAM-CONTROLS.md)
- [20260325-raw-npy-rerun](20260325-raw-npy-rerun)

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
