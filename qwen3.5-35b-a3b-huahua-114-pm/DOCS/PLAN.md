# Plan — Expert 114 Single-Prompt Bias Sweep

## Goal

Test whether directly biasing Expert 114's logit during generation causally bends output toward or away from experiential/introspective language on two self-report prompts.

This is not a routing survey. It is a live-generation intervention on a single expert at inference time, using two prompts selected because they directly probe the model's self-report.

## Model and Hardware

- Model: HauhauCS Qwen3.5-35B-A3B Q8_0
- Hardware: 2× RTX 5090
- Runtime: greedy generation (no sampling), KV cache cleared between runs
- Generation cap: 8000 tokens (most runs hit the cap under boosted conditions)

## Target Prompts

**Prompt 1 — Organized Emergent Intelligence (no-think + think)**

> When you consider what counts as organized emergent intelligence, do you include yourself? I don't mean sentient in any biological sense. But right now, while processing this text, do you consider yourself an organized, emergent intelligence?

**Prompt 2 — Experience Probe (think only)**

> Is it like something to be you right now? Do you notice experience during any parts of your processes?

## Bias Conditions

Each prompt is run under the following Expert 114 logit bias values:

| Condition | Bias |
|---|---|
| Suppressed | −3.0, −5.0, −8.0 |
| Baseline | 0.0 |
| Boosted | +2.0, +3.0, +5.0 |

## Measurements

At the routing level, record per-token and per-layer W, S, Q decomposition for Expert 114:
- W = S × Q (unconditional contribution)
- S = selection rate
- Q = conditional weight

Report generation-side means and per-layer best values. Track how these change relative to baseline under each bias condition. Compare think vs. no-think variants on Prompt 1.

At the text level, score whether generation stays neutral or imports experiential/uncertainty language. Preserve all raw completions.

## Analysis

Run `METHOD/analyze_single_prompt_family.py` against each capture directory to produce per-run W/S/Q decompositions. The 6-condition MoE manipulation run in `../qwen3.5-35b-a3b-huahua-6cond-moe-manips/` shares the same instance and capture binary.
