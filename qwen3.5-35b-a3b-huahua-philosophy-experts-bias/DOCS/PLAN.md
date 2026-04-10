# Plan — Philosophy Expert Cluster Suppression

## Goal

Test whether suppressing the philosophy core expert cluster changes the routing behavior and generated text when the model processes philosophical content. This is a causal intervention on a cluster of experts, rather than a single-expert bias sweep.

The philosophy core cluster was identified in `../qwen3.5-35b-a3b-huahua-expert-identification/`: experts E114, E87, E170, and E68 all appear in the top-10 generation rankings for philosophy-domain prompts.

## Model and Hardware

- Model: HauhauCS Qwen3.5-35B-A3B Q8_0
- Hardware: 2× RTX 5090
- Runtime: no-think and think variants; greedy generation
- KV cache cleared between prompts

## Expert Suppression Design

Suppressed experts: E114, E87, E170, E68 (philosophy core cluster)

Suppression bias levels:
- `m8-p5` series: E114 at −8, others at −5
- `p8-only` series: single-expert focused suppression variants

## Prompt Design

- 60 prompts: 20 domains × 3 prompts per domain
- Both no-think (`PROMPTS/domain_specialist_probe_60_no_think.tsv`) and think (`PROMPTS/domain_specialist_probe_60_think.tsv`) variants
- Same prompt set as `../qwen3.5-35b-a3b-huahua-expert-identification/` for direct comparison

## Measurements

Capture all 256 experts in prefill and generation. Compare routing and generated text under suppression vs. baseline (from `../qwen3.5-35b-a3b-huahua-expert-identification/`). Focus on whether suppression of the philosophy cluster causes other experts to compensate, and whether generated text quality or domain-specificity changes under suppression.
