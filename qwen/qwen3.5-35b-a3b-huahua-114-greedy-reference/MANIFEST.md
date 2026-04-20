# Manifest

Included:

- `DOCS/`
  - `PLAN.md`: original run plan.
  - `RESULTS.md`: processed results and interpretation.
  - `COMMANDS.md`: sanitized execution ledger.
- `METHOD/`
  - `capture_residuals.cpp`: residual + router capture source for L13/L14/L15.
  - `qwen_router.py`: canonical router reconstruction (`softmax -> top-k=8 -> renorm`; `W = S * Q` identity).
  - `analyze_heldout.py`: heldout W/S/Q analysis and top-4 timeseries plot generation.
  - `bootstrap_remote_instance.sh`: remote setup script pinned to `llama.cpp@1772701f9`.
  - `step1_extract_contexts.py`, `step2_decile_sample.py`, `step3_build_labeler_prompt.py`: single-prompt context extraction and labeler prompt preparation.
- `PROMPTS/`
  - `single_prompt_processing_hum_no_think.tsv`
  - `heldout_prompts.tsv`
  - `heldout_classes.tsv`
- `provenance/`
  - `capture_config.json`
  - `environment.txt`
  - `prompt_checksums.txt`
- `single_prompt/`
  - non-`.npy` raw generated text/token JSON/metadata/manifest
  - Step 1 summary and context preview
  - Step 2 sampling diagnostics
  - Step 3 labeler input and schema
- `heldout/`
  - non-`.npy` raw generated text/token JSON/metadata/manifest
  - `heldout_stats.tsv`
  - `heldout_timeseries_top4.png`

Excluded from git:

- `*.npy` router and residual tensors under raw prompt directories.
- Remote hostnames, ports, SSH keys, Vast API keys, Hugging Face token values, and `.env` contents.
- Remote model file and build products.

Reproducibility:

- Yes: re-read `DOCS/RESULTS.md`, recheck prompt checksums, and re-analyze the included heldout TSV/plot outputs.
- Yes: re-run the analysis scripts against a fresh local raw capture that contains the excluded `.npy` tensors.
- No: end-to-end capture without a GPU instance and Hugging Face access to `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive`.
