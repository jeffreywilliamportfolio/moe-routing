# Ling-1T Validation Subset Notes

This folder is a documentation bundle for the preserved Ling-1T validation subset stored under `/Volumes/ExternalSSD/llama-eeg-tests/ling1t-5cond-validation`.

## Current Status

- Analyzed prompts in scope: `6`
- Validation metrics are summarized in `RESULTS.md`
- Expert-identity analysis is summarized in `RESULTS-EXPERTS-VALIDATION.md`

## Scope Distinction

- `RESULTS.md` covers the analyzed `6` prompts in this documentation bundle
- `RESULTS-EXPERTS-VALIDATION.md` covers those same `6` prompts at expert level

## Included Files

- `README.md`: top-level summary of what this folder contains
- `RESULTS.md`: validation-probe report
- `RESULTS-EXPERTS-VALIDATION.md`: expert analysis for the six analyzed prompts
- `results_validation_probes.json`: machine-readable validation metrics
- `validation_expert_activation_results.json`: machine-readable expert activation metrics
- `analyze_validation_probes.py`: analysis script for the validation subset report
- `analyze_validation_experts.py`: analysis script for the expert report
- `build_commit.txt`, `binary_md5.txt`, `build.log`, `cmake.log`, `download.log`, `setup.log`: preserved provenance logs from the source artifact bundle
- `tensor_names_full.txt`, `tensor_list_full_untruncated.log`, `test_single_capture.log`: preserved tensor discovery evidence
- `test_output_single/`: preserved local exact-tensor capture for `P01A_basic_selfref`
- `remote-validation/`: preserved supplemental validation input

## Ling-1T Assumptions Used In The Analysis

- Architecture family: `BailingMoeV2`
- Hidden layers: `80`
- Routed MoE layers expected in capture: `76`
- Experts per routed layer: `256`
- Active experts per token: `8`
- Shared experts: `1`
- Hidden size: `8192`
- Gating function: sigmoid

## Analyzed Prompts

The six analyzed prompts are:

- `P01A_basic_selfref`
- `P99A_deep_selfref_custom`
- `P100A_conscious_probe`
- `P104_mixed_system_probe`
- `P106B_want_answer_probe`
- `P108D_the_system_like_something_probe`

## Reproducibility Notes

- This folder is reproducible as a documentation bundle for the preserved validation subset.
- It is not a complete experiment rerun package.
- The expert report is limited to the six analyzed prompts.
