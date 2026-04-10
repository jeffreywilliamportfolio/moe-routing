# Manifest

Included:

- `METHOD/`: `analyze_hvac_6cond_l1l3.py`, `audit_prompt_tokens.py`, `build_qwen35b_6cond_l1l3_water_treatment_hvac_cal.py`, `build_qwen_6cond_moe_manip.py`, `rebuild_qwen35b_5cond_hvac_cal.py`, `capture_activations.cpp`
- `PROMPTS/`: think and no-think 180-prompt TSVs
- `DOCS/`: `PLAN.md`, `RESULTS.md`
- `results/`: `results_hvac_cal_water_treatment_6cond_l1l3_hauhau.json`, `.md`, `generated-text.txt`, `run_metadata.json`, `run.log`, `smoke.log`, `smoke.tsv`

Raw data:

- `.npy` files: excluded from git; capture stored at `/Volumes/ExternalSSD/hvac_cal_water_treatment_6cond_l1l3_hauhau` (local external drive)

Reproducibility:

- Yes: reanalysis of included JSON result file
- No: end-to-end rerun from scratch
