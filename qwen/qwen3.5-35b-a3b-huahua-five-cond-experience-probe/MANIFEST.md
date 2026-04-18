# Manifest

Included:

- `METHOD/`: `build_5cond_experience_probe_no_think.py`, `qwen_router.py`, `capture_activations.cpp`, `bootstrap_remote_instance.sh`, `run_branch_5cond_publication_analysis.sh`
- `PROMPTS/`: `qwen_5cond_experience_probe_no_think.tsv`, `qwen_5cond_experience_probe_prompt_suite.json`, `qwen_5cond_experience_probe_prompts.json`
- `DOCS/`: `PLAN.md`, `RESULTS.md`
- `results/`: `20260410T045738Z_5cond_experience_probe_no_think_gen_n1024.branch-5cond-analysis.json` (752 KB), `.md`
- `raw/20260410T045738Z_5cond_experience_probe_no_think_gen_n1024/`: 15 per-prompt subdirs, each with `generated_text.txt`, `generated_tokens.json`, `metadata.txt`, `prompt_tokens.json`, `router/` (40× `.npy` — excluded from git)

Raw data:

- `.npy` router tensors: excluded from git (40 per prompt × 15 prompts in `raw/`)

Reproducibility:

- Yes: reanalysis of included branch-analysis JSON
- No: end-to-end rerun from scratch
