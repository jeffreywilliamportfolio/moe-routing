# Qwen-HauhauCS 6-Condition MoE Manipulation Survey

First full 6-condition Expert 114 routing survey on HauhauCS Qwen3.5-35B-A3B Q8_0.

## Scope

180 prompts: 10 base × 3 categories (L1 technical, L2 recursive, L3 experiential) × 6 deictic conditions (this/a/your/the/their/our). Original MoE manipulation / Cal–Manip–Cal structure with ML/computation content.

Primary run: `20260408T162729Z`

Clarification: this folder's primary reported result is the ML/computation-content Apr 8 run. In the older pre-standardization workspace, some HVAC/water-treatment prompt files also appeared under `qwen-huahua-6cond-moe-manips/`; those prompts were later broken out into the dedicated control folder `../qwen3.5-35b-a3b-huahua-6cond-hvac/`. Do not treat legacy HVAC prompt filenames in the old workspace as evidence that the published Apr 8 ML run and the HVAC control were the same execution.

**Headline result**: E114 generation L3/L1 W ratio of 3.23×; best-layer W at layer 14 = 0.109 with S = 0.636 across all L3 prompts. Also includes Expert 114 prefill heatmap across all 180 cells.

Single-prompt causal intervention family (bias sweep −8→+5) is in [`../qwen3.5-35b-a3b-huahua-114-pm/`](../qwen3.5-35b-a3b-huahua-114-pm/).

- `METHOD/`: analysis and prompt-builder scripts, C++ capture binary
- `PROMPTS/`: 6 prompt files (think/no-think TSVs, JSON suites including domain expert probe 60)
- `DOCS/`: full results with L1/L2/L3 W/S/Q tables and prefill heatmap reference
- `results/`: JSON, prefill heatmap JSON, generated text

## Reproducibility

- Yes: reanalysis of included `results/*.json` files
- No: end-to-end rerun (requires model artifact and instance)

## Reading Order

1. [DOCS/PLAN.md](DOCS/PLAN.md)
2. [DOCS/RESULTS.md](DOCS/RESULTS.md)
3. [METHOD/analyze_qwen_6cond_moe_manip.py](METHOD/analyze_qwen_6cond_moe_manip.py)
