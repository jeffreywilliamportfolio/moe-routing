# Manifest

Included:

- `METHOD/`: `analyze_single_prompt_family.py`, `capture_activations_expert_bias.cpp`, prompt-suite JSON files
- `PROMPTS/`: think and no-think TSVs for both target prompts
- `DOCS/`: `PLAN.md`, `RESULTS.md` (W/S/Q tables for all bias levels, generated-text links)
- `results/`: generated text files per run, per-family analysis outputs (`.md`), baseline capture directory with `.npy` files (baseline no-think only)

Raw data:

- `.npy` files: baseline no-think capture only is present locally (`results/single_prompt_baseline/capture_20260408T200535Z/`)
- All other single-prompt runs: raw npy archive was created remotely but SCP failed mid-transfer — not recoverable from this repo

Reproducibility:

- Yes: reanalysis of generated text and analysis outputs in `results/`
- No: end-to-end rerun from scratch
