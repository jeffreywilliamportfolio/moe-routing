# Manifest

Included:

- `METHOD/`: `analyze_huahua_5cond_diectics.py`, `build_single_joke_5cond_diectics.py`, `run_huahua_5cond_diectics.sh`, `bootstrap_remote_instance.sh`, `qwen_router.py`, `capture_activations.cpp`
- `PROMPTS/`: `single_joke_5cond_diectics.tsv`, `single_joke_5cond_diectics_prompt_suite.json`
- `DOCS/`: `PLAN.md`, `RESULTS.md`, `PER-TOKEN-RESULTS.md`, `PROVENANCE.md`, `SOURCE-README.md`
- `results/`: base summary JSON/Markdown plus Expert 114 JSON/Markdown extracts for run `20260410T184005Z`

Excluded:

- `raw/`: reserved, but no raw capture directory is included here
- `.npy` router tensors: not present in git

Reproducibility:

- Yes: local reanalysis of included prompts and saved result artifacts
- Partial: prompt regeneration and environment bootstrap scripting
- No: full capture rerun without the external model artifact, capture binary build, and raw router tensors
