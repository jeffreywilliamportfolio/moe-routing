# Manifest

Included:

- `METHOD/`: `analyze_expert_identification.py`, `audit_prompt_tokens.py`, `build_domain_specialist_probe_60.py`, `build_expert_identification_think_family.py`, `capture_activations.cpp`
- `PROMPTS/`: `domain_specialist_probe_60_no_think.tsv`, `domain_specialist_probe_60_think.tsv`, and JSON metadata files
- `DOCS/`: `PLAN.md`, `RESULTS.md` (per-domain W/S/Q tables, E114 breakdown, overall expert rankings)
- `results/`: `results_domain_specialists_20260408T235839Z.json`, `.md`, `.npz`
- `non_npy_remote_artifacts/`: index of remote capture location

Raw data:

- `.npy` files: remote only — capture directory `20260408T235839Z_domain_specialist_no_think_hauhau` on Vast.ai instance
- Not pulled locally; `non_npy_remote_artifacts/` has the index

Reproducibility:

- Yes: reanalysis of included `.json` / `.npz` result files
- No: end-to-end rerun from scratch
- Note: think-mode TSV exists but has not been run yet
