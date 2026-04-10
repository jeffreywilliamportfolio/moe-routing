# Manifest

Included:

- `METHOD/`: `analyze_qwen_6cond_moe_manip.py`, `audit_prompt_tokens.py`, `build_domain_expert_probe_60.py`, `build_domain_expert_probe_60_style_varied.py`, `build_qwen_6cond_moe_manip.py`, `capture_activations.cpp`
- `PROMPTS/`: `qwen-6cond-moe-manip.tsv`, `qwen-6cond-moe-manip-think.tsv`, `domain_expert_probe_60.json`, `domain_expert_probe_60_no_think.tsv`, `domain_expert_probe_60_style_varied.json`, `domain_expert_probe_60_style_varied_no_think.tsv`
- `DOCS/`: `PLAN.md`, `RESULTS.md`
- `results/`: `results_qwen_6cond_moe_manip_hauhau_20260408T162729Z.json`, `.md`, `_prefill_e114_heatmap.json`, `generated-text.txt`

Raw data:

- `.npy` files: excluded from git; raw capture stored remotely

Reproducibility:

- Yes: reanalysis of included JSON result files
- No: end-to-end rerun from scratch
