# Manifest

Included:

- `METHOD/`: `analyze_single_prompt_family.py`, `qwen_router.py`, `capture_activations.cpp`, `bootstrap_remote_instance.sh`
- `PROMPTS/`: `single_prompt_processing_hum_no_think.tsv`
- `DOCS/`: `PLAN.md`, `RESULTS.md`
- `results/`: full-output token breakdown TSV, deep-still-water segment TSV, snippet TSV, summary JSONs, run-level JSON and markdown
- `raw/20260410T042340Z_single_prompt_processing_hum_no_think_gen_n1024/S01_processing_hum_probe/`: `generated_text.txt`, `generated_tokens.json`, `metadata.txt`, `prompt_tokens.json`, `router/` (40× `.npy` — excluded from git)

Raw data:

- `.npy` router tensors: excluded from git (40 files in `raw/S01_processing_hum_probe/router/`)

Reproducibility:

- Yes: reanalysis of included TSV token breakdowns and summary JSON
- No: end-to-end rerun from scratch
