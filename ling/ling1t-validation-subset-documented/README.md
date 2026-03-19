# Ling-1T Validation Subset Documentation Bundle

This folder documents the preserved Ling-1T validation subset from `/Volumes/ExternalSSD/llama-eeg-tests/ling1t-5cond-validation`.

## Scope

- Analyzed prompts: `6`
- Expert-identity analysis is available for those `6` prompts
- This folder is not a complete experiment bundle

## Key Files

- `RESULTS.md`: validation report for the analyzed prompts
- `RESULTS-EXPERTS-VALIDATION.md`: expert-identity analysis for the `6` analyzed prompts
- `results_validation_probes.json`: machine-readable validation metrics
- `validation_expert_activation_results.json`: machine-readable expert activation results
- `analyze_validation_probes.py`: script used to summarize the validation subset
- `analyze_validation_experts.py`: script used to summarize the `6` analyzed prompts
- `build_commit.txt`, `binary_md5.txt`, `build.log`, `cmake.log`, `download.log`, `setup.log`: preserved provenance logs from the validation artifact bundle
- `tensor_names_full.txt`, `tensor_list_full_untruncated.log`, `test_single_capture.log`: preserved tensor-discovery evidence

## Interpretation

- Treat this as a documented validation subset, not a complete rerun package.
- The analyzed set in this bundle is `6` prompts.
- The expert report covers those same `6` prompts.
