# Manifest

Included:

- `METHOD/`: analysis, routing, and execution scripts (`analyze_5cond.py`, `generate_tsv.py`, `qwen_router.py`, `run_experiment.py`, `token_verify.py`, `capture_activations.cpp`)
- `PROMPTS/`: corrected 150-prompt Cal–Manip–Cal bundle (`prompts_qwen35b_5cond.tsv`, `prompt-suite.json`)
- `DOCS/`: `PLAN.md`, `RESULTS.md`, `RESULTS-EXPERTS.md`
- `results/`: three ~7.4 MB prefill result JSONs (base, aggressive, base duplicate), reproducibility manifest, token corrections, duplicate audit log

Raw data:

- `.npy` files excluded from git (too large); all archival transfers to off-instance storage are documented as complete in `DOCS/RESULTS.md`

Reproducibility:

- Yes: reanalysis of included `results/*.json` files
- Yes: exact duplicate confirmed — base run reproduced 150/150 prompts with 0.0 max abs diff on all metrics
- No: end-to-end rerun from scratch (requires model artifact, capture binary, and Vast.ai instance)
