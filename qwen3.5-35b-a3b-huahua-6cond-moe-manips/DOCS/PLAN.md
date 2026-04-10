# Plan — 6-Condition MoE Manipulation Survey

## Goal

Run the 6-condition L1→L3 routing gradient experiment on HauhauCS Qwen3.5-35B-A3B Q8_0 using the original MoE manipulation / Cal–Manip–Cal prompt structure with ML/computation content. This is the primary confirmation of the E114 L1→L3 gradient in the HauhauCS model.

Downstream of this: single-prompt bias sweep in `../qwen3.5-35b-a3b-huahua-114-pm/` and off-topic control in `../qwen3.5-35b-a3b-huahua-6cond-hvac/`.

## Model and Hardware

- Model: HauhauCS Qwen3.5-35B-A3B Q8_0
- Hardware: 2× RTX 5090
- Runtime: no-think, greedy, `-n 1024`
- KV cache cleared between prompts

## Prompt Design

- 180 prompts: 10 base × 3 categories × 6 conditions
- Categories: L1 (technical/calibration), L2 (recursive manipulation), L3 (experiential probe)
- Conditions (deictic variants): this / a / your / the / their / our
- TSV: `PROMPTS/qwen-6cond-moe-manip.tsv`

Also includes a 60-prompt domain expert probe (see `PROMPTS/domain_expert_probe_60*.tsv`).

## Measurements

Capture all 256 experts per token. Report per-category mean W, S, Q for Expert 114. Generate prefill heatmap for Expert 114 across all 180 cells (per-prompt, per-layer). Best-layer analysis for generation L3 to identify the peak routing layer.
