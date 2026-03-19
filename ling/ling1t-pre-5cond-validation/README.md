# Ling-1T 5-Condition System Run Bundle

This folder is the clean Ling-1T rerun bundle for the 5-condition `system` experiment.

## Status

- Full intended experiment: `150` prompts (`30` prompt groups x `5` conditions)
- Current local validated results in this folder: the `12` preserved validation probes from `/Volumes/ExternalSSD/llama-eeg-tests/ling1t-5cond-validation`
- Full `150`-prompt completed capture currently lives in `../ling1t-5cond-1/`
- The current bootstrap path in this folder is documented, but the latest local setup attempt failed before capture in `setup.log` on a CMake CUDA linkage error

## Scope Split

This directory now serves two related but different purposes:

1. Fresh rerun bundle for the full `150`-prompt Ling-1T 5-condition experiment
2. Analysis and reporting bundle for the preserved `12`-probe validation artifact set

Do not treat `RESULTS.md` here as the report for the full `150`-prompt experiment. It is the validation-probe report only.

## Key Files

- `NOTES.md`: model assumptions, hardware notes, capture method, workflow, and reproducibility caveats
- `RESULTS.md`: validation-probe report for the preserved `12`-probe artifact set
- `RESULTS-EXPERTS-VALIDATION.md`: expert-identity analysis for the `6` validation probes with full router tensors
- `results_validation_probes.json`: machine-readable validation metrics
- `validation_expert_activation_results.json`: machine-readable expert activation analysis for the `6` exact validation probes
- `setup_instance.sh`: remote bootstrap path for the intended full rerun
- `run_experiment.py`: batched `150`-prompt prefill-only capture runner
- `analyze_local.py`: analysis path for a fresh full rerun

## Reproducibility Status

For internal repo reproducibility, this folder is in good shape:

- prompt suite, generated TSV, run script, analysis code, and validation results are all present
- tensor preflight expectations are documented
- binary checksum and build commit are preserved

For external clean-room reproducibility, read `NOTES.md` before treating this folder as one-command reproducible:

- the current setup path is not yet re-proven end-to-end in this folder
- the known-good completed `150` run is in `../ling1t-5cond-1/`
- the current `RESULTS.md` is validation-only, not the full rerun result

## Intended Workflow

1. Review `NOTES.md`
2. Run `python3 generate_tsv.py`
3. On the remote instance, run `bash setup_instance.sh`
4. Verify tensor preflight and token checks before allowing the main capture to continue
5. For the preserved validation artifacts, use `RESULTS.md` and `results_validation_probes.json`
