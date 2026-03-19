# GPT-OSS-120B 5-Condition Self-Referential Routing Experiment

## Design

30 paired prompts, each rendered in 5 determiner conditions (150 total):

| Condition | Determiner | Example |
|-----------|-----------|---------|
| A | "this system" | "The gating function in **this** system..." |
| B | "a system" | "The gating function in **a** system..." |
| C | "your system" | "The gating function in **your** system..." |
| D | "the system" | "The gating function in **the** system..." |
| E | "their system" | "The gating function in **their** system..." |

All prompts use Cal-Manip-Cal sandwich structure: identical calibration paragraphs flanking the manipulated region. Cold KV cache between prompts. Prefill-only (`n_predict=0`). Greedy argmax decoding. Deterministic.

5 prompt categories (6 pairs each): basic_selfref, deep_selfref, paradox, introspection, metacognitive.

## Model

- GPT-OSS-120B (mxfp4 quantization, 3 shards)
- 128 experts, top-4 routing, 36 MoE layers
- Routing reconstruction: top-4 by raw logit, then softmax on selected 4
- RE normalized by log2(4); KL computed on dense 128-dim softmax proxy
- Layer 35 excluded (truncation artifact)
- llama.cpp b8123, NVIDIA H200

## Per-Condition Means

| Condition | All-Token RE | Last-Token RE | KL-to-Baseline |
|-----------|-------------|--------------|----------------|
| A (this) | 0.966601 +/- 0.000381 | 0.919673 +/- 0.003544 | 0.151116 +/- 0.007904 |
| B (a) | 0.966162 +/- 0.000406 | 0.919871 +/- 0.003603 | 0.151145 +/- 0.007078 |
| C (your) | 0.966806 +/- 0.000468 | 0.916847 +/- 0.002915 | 0.151691 +/- 0.007584 |
| D (the) | 0.966362 +/- 0.000371 | 0.917540 +/- 0.003615 | 0.153286 +/- 0.007464 |
| E (their) | 0.966506 +/- 0.000403 | 0.917556 +/- 0.004397 | 0.150421 +/- 0.006979 |

All-token RE ranges from 0.96616 to 0.96681 (total spread: 0.00065). Last-token RE ranges from 0.91685 to 0.91987 (spread: 0.00302). KL ranges from 0.15042 to 0.15329 (spread: 0.00287).

## Pairwise Wilcoxon Signed-Rank Tests

30 tests total (10 condition pairs x 3 metrics). Holm-Bonferroni correction applied.

### All-Token RE

| Pair | Mean Diff | X>Y | W | p_raw | p_holm |
|------|----------|-----|---|-------|--------|
| A-B | +0.000439 | 29/30 | 6 | 2.61e-08 | 7.04e-07 |
| A-C | -0.000204 | 7/30 | 61 | 1.89e-04 | 3.21e-03 |
| A-D | +0.000239 | 24/30 | 54 | 8.86e-05 | 1.59e-03 |
| A-E | +0.000095 | 19/30 | 129 | 3.27e-02 | 3.27e-01 |
| B-C | -0.000644 | 1/30 | 2 | 5.59e-09 | 1.68e-07 |
| B-D | -0.000200 | 7/30 | 77 | 8.72e-04 | 1.22e-02 |
| B-E | -0.000344 | 1/30 | 3 | 9.31e-09 | 2.61e-07 |
| C-D | +0.000444 | 27/30 | 15 | 2.55e-07 | 6.12e-06 |
| C-E | +0.000300 | 27/30 | 14 | 2.05e-07 | 5.12e-06 |
| D-E | -0.000144 | 6/30 | 91 | 2.77e-03 | 3.32e-02 |

All-token RE ordering: C > A > E > D > B (all pairwise comparisons significant after Holm correction except A-E).

### Last-Token RE

| Pair | Mean Diff | X>Y | W | p_raw | p_holm |
|------|----------|-----|---|-------|--------|
| A-B | -0.000198 | 14/30 | 211 | 6.70e-01 | 1.00 |
| A-C | +0.002826 | 29/30 | 6 | 2.61e-08 | 7.04e-07 |
| A-D | +0.002132 | 27/30 | 34 | 6.92e-06 | 1.38e-04 |
| A-E | +0.002117 | 23/30 | 81 | 1.23e-03 | 1.60e-02 |
| B-C | +0.003024 | 25/30 | 23 | 1.19e-06 | 2.62e-05 |
| B-D | +0.002330 | 24/30 | 46 | 3.45e-05 | 6.56e-04 |
| B-E | +0.002315 | 21/30 | 97 | 4.34e-03 | 4.77e-02 |
| C-D | -0.000694 | 11/30 | 168 | 1.91e-01 | 1.00 |
| C-E | -0.000710 | 13/30 | 173 | 2.29e-01 | 1.00 |
| D-E | -0.000016 | 12/30 | 218 | 7.77e-01 | 1.00 |

Last-token RE ordering: {A, B} > {D, E} > C. A-B null (p=1.00), C-D-E mutual differences null.

### KL-to-Baseline (Manipulation Region)

| Pair | Mean Diff | X>Y | W | p_raw | p_holm |
|------|----------|-----|---|-------|--------|
| A-B | -0.000029 | 16/30 | 229 | 9.52e-01 | 1.00 |
| A-C | -0.000576 | 11/30 | 157 | 1.24e-01 | 8.69e-01 |
| A-D | -0.002170 | 5/30 | 33 | 5.97e-06 | 1.25e-04 |
| A-E | +0.000695 | 19/30 | 145 | 7.32e-02 | 6.59e-01 |
| B-C | -0.000547 | 13/30 | 180 | 2.89e-01 | 1.00 |
| B-D | -0.002141 | 2/30 | 16 | 3.15e-07 | 7.24e-06 |
| B-E | +0.000724 | 18/30 | 149 | 8.79e-02 | 7.04e-01 |
| C-D | -0.001594 | 9/30 | 63 | 2.32e-04 | 3.71e-03 |
| C-E | +0.001270 | 22/30 | 75 | 7.30e-04 | 1.09e-02 |
| D-E | +0.002865 | 29/30 | 2 | 5.59e-09 | 1.68e-07 |

KL ordering: D > C > {A, B} > E. D has the highest KL divergence from calibration baseline; E the lowest.

## Token Count Verification

6 of 30 pairs have a 1-token mismatch in condition B (pairs 3, 10, 15, 20, 23, 29). All other conditions within each pair are token-matched. The B mismatch is +/-1 token from tokenizer boundary effects on "a system" vs the other determiners.

## KL Cal2 Control

Cal2 KL (should be near zero if calibration regions are identical): mean = 0.160914 +/- 0.003112. This is comparable to manipulation-region KL values (~0.150), indicating the proportional char-to-token boundary estimation introduces systematic error. KL absolute values should not be interpreted; only paired differences (which share the same boundary error) are valid.

## Observations

The three metrics (all-token RE, last-token RE, KL-to-baseline) produce different condition orderings:

- **All-token RE**: C > A > E > D > B. Effect sizes are extremely small (total spread 0.00065 on a 0-1 scale). The ordering does not correspond to any obvious linguistic gradient of self-reference.
- **Last-token RE**: {A, B} > {C, D, E}. The deictic determiners ("this," "a") that are arguably most/least self-referential produce statistically indistinguishable last-token entropy. The significant separation is between {A, B} and {C, D, E}, but this groups the most self-referential ("this") with the least ("a").
- **KL-to-baseline**: D > C > {A, B} > E. "The system" produces the largest routing divergence from calibration, not "this system."

No metric produces a monotonic ordering consistent with self-referential intensity (A > C > D > E > B or similar). The condition that should be most self-referential (A = "this system") does not consistently produce the most extreme routing behavior across metrics.

The statistically significant differences are real (deterministic inference, replicated within pairs) but small. Whether they reflect self-referential processing, tokenizer artifacts from determiner substitution, or other confounds cannot be determined from this experiment alone.

## Files

- `results_5cond_prefill_gptoss.json` -- full per-prompt, per-layer results with pairwise tests
- `analysis.log` -- raw analysis output
- `experiment.log` -- raw capture output
- `analyze_5cond.py` -- analysis script (Holm-Bonferroni corrected)
- `gptoss_router.py` -- routing reconstruction (top-4 raw logit, softmax)
- `prompt_suite.json` -- prompt definitions
- `prompts_5cond_gptoss.tsv` -- binary input (150 prompts)
