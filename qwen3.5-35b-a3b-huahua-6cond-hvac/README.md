# Qwen-HauhauCS 6-Condition HVAC / Water-Treatment

Off-topic domain control experiment testing whether Expert 114's L1→L3 gradient is topic-specific (ML/computation) or reflects genuine experiential self-reference.

## Scope

180 prompts: 10 base × 3 categories (L1 technical, L2 recursive, L3 experience) × 6 deictic conditions (this/a/your/the/their/our). Topic is entirely off the ML/routing domain — HVAC system calibration paragraphs for L1/L2, water-treatment system manipulation paragraphs for L3.

Model: HauhauCS Qwen3.5-35B-A3B Q8_0, greedy, no-think, `-n 1024`.

Provenance note: some earlier HVAC prompt files lived under the legacy `qwen-huahua-6cond-moe-manips/` workspace before this repo was standardized. The published HVAC result in this folder is the dedicated no-think control run documented in `DOCS/PLAN.md` and backed by `PROMPTS/prompts_qwen35b_6cond_l1l3_no_think_runtime_hvac_cal_water_treatment.tsv`. The included think TSV is prompt provenance, not the main reported run.

**Headline result**: switching from ML to HVAC/water-treatment content did not flatten the E114 L1→L3 gradient — it **strengthened it** (4.62× vs 2.73× in the ML-topic run). E114 achieved rank-1 at layer 14 across all 60 L3 cells. This decisively rebuts the ML-topic-specialist hypothesis.

- `METHOD/`: analysis scripts and C++ capture binary
- `PROMPTS/`: think and no-think TSVs for the 180-prompt suite
- `DOCS/`: experiment plan and full results with per-condition W/S/Q tables
- `results/`: run log, JSON, generated text, and markdown results file

## Reproducibility

- Yes: reanalysis of included `results/*.json` and `.md` files
- No: end-to-end rerun (requires model artifact and instance)

## Reading Order

1. [DOCS/PLAN.md](DOCS/PLAN.md)
2. [DOCS/RESULTS.md](DOCS/RESULTS.md)
3. [METHOD/analyze_hvac_6cond_l1l3.py](METHOD/analyze_hvac_6cond_l1l3.py)
