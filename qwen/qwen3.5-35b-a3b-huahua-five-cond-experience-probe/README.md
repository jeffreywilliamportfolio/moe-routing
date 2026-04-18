# Qwen3.5-35B-A3B HauhauCS — 5-Condition Experience Probe

5-condition experience probe on HauhauCS Qwen3.5-35B-A3B Q8_0.

## Scope

15 prompts (3 prompt pairs × 5 deictic conditions: A/B/C/D/E) using experience-probe content in no-think mode. Captures full routing with 40-layer `.npy` tensors per prompt. Uses the Cal–Manip–Cal structure with routing entropy and KL-to-baseline metrics.

Run: `20260410T045738Z` | 15 prompts × ~1010 generated tokens | no-think, greedy

**Headline result**: E114 is the top manipulation expert on all 15 prompts without exception. KL-to-manip mean 0.274 (Wilcoxon p = 6.3e-05 vs. baseline). Last-token RE is significantly elevated vs. all-token RE (Wilcoxon p = 7.6e-03).

- `METHOD/`: capture binary, analysis and builder scripts
- `PROMPTS/`: TSV prompt suite (no-think) and JSON suite files
- `DOCS/`: experiment plan and full results with per-prompt E114 counts
- `results/`: branch-analysis JSON and markdown
- `raw/`: timestamped capture directory with per-prompt router `.npy` files (excluded from git)

## Reproducibility

- Yes: reanalysis of included `results/*.json` file
- No: end-to-end rerun (requires model artifact and instance)
- Note: `.npy` files in `raw/` are excluded from git

## Reading Order

1. [DOCS/PLAN.md](DOCS/PLAN.md)
2. [DOCS/RESULTS.md](DOCS/RESULTS.md)
3. [METHOD/build_5cond_experience_probe_no_think.py](METHOD/build_5cond_experience_probe_no_think.py)
