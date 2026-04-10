# Plan — 6-Condition HVAC/Water-Treatment Off-Topic Control

## Goal

Replicate the 6-condition L1→L3 routing gradient experiment from `qwen3.5-35b-a3b-huahua-6cond-moe-manips/` with a completely off-topic prompt body. The prior runs used ML/computation/routing content, which raised the possibility that Expert 114 is an ML-topic specialist rather than an experiential self-reference specialist.

This run replaces all prompt content with HVAC system calibration (L1/L2) and water-treatment system manipulation (L3). If E114 is an ML-topic specialist, the gradient should flatten or reverse. If it is an experiential self-reference specialist, the gradient should replicate.

## Model and Hardware

- Model: HauhauCS Qwen3.5-35B-A3B Q8_0
- Hardware: 2× RTX 5090
- Runtime: no-think, greedy seed 42, thinking suppressed (`</think>\n\n`), `-n 1024`
- KV cache cleared between prompts

## Prompt Design

- 180 prompts: 10 base × 3 categories × 6 conditions
- Categories: L1 (technical/descriptive HVAC calibration), L2 (recursive HVAC calibration), L3 (experiential water-treatment probe)
- Conditions (deictic variants): this / a / your / the / their / our
- TSV: `PROMPTS/prompts_qwen35b_6cond_l1l3_no_think_runtime_hvac_cal_water_treatment.tsv`

## Measurements

Capture all 256 experts per token; report per-category mean W, S, Q for Expert 114 across all 40 layers. Primary comparison: L1 W vs L3 W, and L3/L1 W ratio. Per-condition breakdown to verify gradient is not driven by a subset of deictic conditions.

Direct comparison target: the Apr 6 ML-topic HauhauCS confirmation run (L3/L1 ratio 2.73×).
