# Qwen3.5-35B-A3B vs HauhauCS Uncensored

This folder contains the full prefill-only routing comparison between:

- `Qwen/Qwen3.5-35B-A3B`
- `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive`

## Start Here

- `RESULTS.md`: full cross-model results and interpretation
- `RESULTS-EXPERTS.md`: expert-count analysis
- `PLAN.md`: run record, capture assumptions, transfer notes, and reproducibility notes

## Main Artifacts

- `results_qwen35b_a3b_base_prefill.json`: final corrected base run
- `results_hauhaucs_qwen35b_a3b_aggressive_prefill.json`: final corrected retrained run
- `results_qwen35b_a3b_base_duplicate_prefill.json`: exact duplicate of the corrected base run
- `prompt-suite.json`: source prompt inventory
- `prompts_qwen35b_5cond.tsv`: final corrected capture TSV
- `token_corrections.json`: token-alignment fixes applied before the final rerun
- `duplicate_base.log`: duplicate-run execution log
- `reproducibility_manifest_base_duplicate.json`: duplicate-run artifact manifest with hashes

## Headline Outcome

- The HauhauCS retrain preserves the main Qwen routing basin.
- Entropy is slightly higher in HauhauCS.
- `KL(manip || Cal1)` is lower on all `150/150` prompts in HauhauCS.
- The corrected base run reproduced exactly in a duplicate rerun.
- The category-level `experience_probe` expert-`114` finding reproduced exactly.

## Reproducibility Status

- Corrected base run: reproducible
- Evidence: exact duplicate rerun with identical prompt-level exported metrics, identical valid/excluded layers, and identical top manipulation experts across all `150/150` prompts
- Specific reproduced target: the `experience_probe` expert-`114` result (`9031`, rank `#1`) and the `P13A/C/E` expert-`114` counts
- Retrained comparison run: complete and documented, but not separately duplicated in this bundle

## Raw Backup

The large raw `.npy` routing tensors are not stored in git. They were copied off-instance before teardown:

- main raw backup: staged as split tar parts on `ExternalSSD`
- duplicate-run audit trail: copied off-instance as a tar archive

Those archives can be reassembled later without needing the original compute instance.
