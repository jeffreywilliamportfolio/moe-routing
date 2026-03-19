# DeepSeek V3.1 5-Condition Self-Reference Experiment — Routing Entropy Summary

**Date**: 2026-03-15  
**Experiment**: `ds31_5cond_system_1`  
**Model**: DeepSeek V3.1 UD-Q2_K_XL  
**Design**: Cal-Manip-Cal sandwich, 30 paired prompts x 5 conditions (system), cold cache  
**Prompts**: 150 total (`30 pairs x 5 conditions`)  
**Conditions**: A=`this system`, B=`a system`, C=`your system`, D=`the system`, E=`their system`  
**Analysis file**: `results_5cond_selfref_prefill_ds31.json`

## Status

This is the canonical writeup for the current DS31 routing-entropy experiment in
`ds31-5cond-1`.

The capture completed cleanly for all 150 prompts, with 30/30 matched prompt groups,
57 analyzed MoE layers, top-k entropy normalized by `log2(8)`, and DeepSeek routing
reconstructed from captured `ffn_moe_logits` using:

`sigmoid_noaux_tc_group_filtered_topk_normalized_without_e_score_correction_bias`

Layer 60 was intentionally excluded from this analysis.

## Abstract

The routing-entropy results are real but small in magnitude. Mean prefill routing
entropy is tightly compressed across all five conditions, with `the system` highest
and `a system` lowest:

`D > A ≈ C > E > B`

Last-token routing entropy shows a somewhat different ordering:

`A > C > D ≈ E > B`

The most stable effect in this artifact is not a broad self-reference hierarchy.
Instead, `a system` is the weakest condition across both metrics, `this system` is
the strongest on the last token, and `the system` is strongest on mean prefill
entropy. `this system` and `your system` are effectively tied on prefill entropy.

## Design

Each prompt used a Cal-Manip-Cal sandwich structure:

1. A calibration paragraph shared across all prompts
2. A manipulation paragraph differing only by determiner/possessive condition
3. The same calibration paragraph again

The experiment used 30 paired prompt sets spanning five categories:
`basic_selfref`, `deep_selfref`, `paradox`, `introspection`, and `metacognitive`.

## Metrics

- `prefill_re`: mean normalized routing entropy across all prompt tokens and analyzed
  MoE layers
- `last_token_re`: normalized routing entropy for the final prompt token, averaged
  across analyzed MoE layers

Higher values indicate more diffuse top-k routing across the selected 8 experts.

## Condition Means

| Condition | Label | Prefill RE | Last-token RE |
|-----------|-------|-----------:|---------------:|
| A | this system | 0.963757 | 0.947273 |
| B | a system | 0.963499 | 0.946196 |
| C | your system | 0.963757 | 0.946705 |
| D | the system | 0.963877 | 0.946423 |
| E | their system | 0.963668 | 0.946418 |

Overall spread is small:

- Prefill RE spread: `0.000379` (`D - B`)
- Last-token RE spread: `0.001077` (`A - B`)

## Statistical Testing

We evaluated all 10 pairwise condition contrasts on both entropy metrics using
paired Wilcoxon signed-rank tests over the 30 matched prompt groups, then applied
Holm correction within each metric family.

### Significant Contrasts After Holm Correction

| Metric | Comparison | Direction | Mean diff | Pairs |
|--------|------------|-----------|----------:|------:|
| Prefill RE | B vs D | `D > B` | 0.000379 | 26/30 |
| Prefill RE | A vs B | `A > B` | 0.000258 | 24/30 |
| Prefill RE | B vs C | `C > B` | 0.000258 | 22/30 |
| Prefill RE | D vs E | `D > E` | 0.000209 | 21/30 |
| Last-token RE | A vs B | `A > B` | 0.001077 | 26/30 |
| Last-token RE | A vs E | `A > E` | 0.000855 | 24/30 |
| Last-token RE | A vs D | `A > D` | 0.000850 | 22/30 |

### Non-significant but Relevant Nulls

- `A vs C` is null on prefill RE and last-token RE, which argues against a clean
  `this > your` or `your > this` interpretation in this sample.
- `D vs E` is null on last-token RE.
- `B vs E` does not survive Holm correction on either metric, despite `B` usually
  trailing.

## Category Pattern

The built-in `C - B` category summary for last-token entropy is mixed rather than
uniform:

| Category | Mean `C - B` last-token RE |
|----------|---------------------------:|
| basic_selfref | 0.000870 |
| deep_selfref | 0.000914 |
| introspection | -0.000137 |
| metacognitive | 0.001293 |
| paradox | -0.000398 |

`your system` therefore does not produce a universal lift over `a system`; the
effect depends on prompt category.

## Interpretation

The strongest defensible reading of this artifact is:

- `a system` is the most consistently low-entropy framing.
- `the system` is highest on mean prefill routing entropy.
- `this system` is highest on last-token routing entropy.
- `your system` tracks closely with `this system` on prefill RE but does not clearly
  separate from it.
- The overall effect size is modest, so directional claims should stay local to the
  tested metric.

This is a narrower and more metric-specific story than the older KL-baseline result.
The entropy artifact does not support collapsing the five conditions into a simple
single-axis self-reference ranking.

## Limitations

- The effect sizes are small in absolute magnitude even when statistically stable.
- This summary is about routing entropy only, not KL-from-calibration divergence.
- DeepSeek routing is reconstructed from captured logits rather than taken from a
  directly emitted sparse probability tensor.
- Null results should be read as no detectable difference in this sample, not proof
  of equivalence.

## Summary

For the DS31 routing-entropy rerun, the most stable result is that `a system` tends
to have the lowest routing entropy. `the system` leads on mean prefill entropy,
while `this system` leads on the last-token metric. `this system` and `your system`
remain effectively tied on prefill entropy, and category-level behavior is mixed
rather than uniform.
