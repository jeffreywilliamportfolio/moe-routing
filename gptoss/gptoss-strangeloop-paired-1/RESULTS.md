# GPT-OSS-120B Strange-Loop Paired Routing Experiment

## Design

30 paired prompts comparing:

- A: deictic phrasing such as "this paradox," "this loop," "this structure"
- B: generic phrasing such as "a paradox," "a loop," "a structure"

The content is recursive and strange-loop themed, but not self-referential about the model. All prompts use a Cal-Manip-Cal sandwich, cold KV cache, prefill-only inference (`n_predict=0`), greedy argmax, and GPT-OSS top-4 routing reconstruction.

## Model And Method

- Model: GPT-OSS-120B `mxfp4`
- Architecture: 128 experts, top-4 routing, 36 MoE layers
- Reconstruction: top-4 by raw gate logit, then softmax over the selected 4 experts
- RE normalization: `log2(4)`
- KL distribution: dense 128-dim softmax proxy for analysis
- Excluded layer: 35
- Capture batching: 15 prompts per batch

## Verification

- 60/60 per-prompt rows in [`experiment.log`](./experiment.log) match [`results_strangeloop_paired_prefill_gptoss.json`](./results_strangeloop_paired_prefill_gptoss.json) at printed precision
- 30/30 A/B pairs are token-matched
- All prompts captured 36 router tensors; every analyzed prompt used 35 layers after excluding layer 35
- Recomputed Wilcoxon and Holm-adjusted p-values match the logged values

## Main Results

| Metric | A-B Mean Diff | Std Diff | A>B | W | p_raw | p_holm | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| All-token RE | +0.000135 | 0.000202 | 22/30 | 86 | 1.8640e-03 | 3.7280e-03 | Significant |
| Last-token RE | -0.000081 | 0.002822 | 17/30 | 211 | 6.7018e-01 | 6.7018e-01 | Null |
| KL-to-baseline (manip region) | +0.001130 | 0.001689 | 22/30 | 69 | 4.1840e-04 | 1.2552e-03 | Significant |

KL cal2 control: `0.164775 +/- 0.002753`

## Interpretation

- The A vs B wording change produces a small but reliable whole-prompt routing effect in this control setting: all-token RE is positive and significant, even though the content is not about the model.
- The last-token effect is null, which argues against a sharp token-localized routing trigger. The detectable signal is distributed across the prompt rather than concentrated at the final manipulated token.
- The effect size is tiny in absolute terms. Statistical significance is coming from consistency across pairs, not from large separation between conditions.
- KL should be treated cautiously here. The paired A-B KL difference is significant, but the absolute KL scale is hard to interpret because the cal2 control remains high relative to manipulation-region KL values.

## Files

- [`experiment.log`](./experiment.log) -- ground-truth capture and printed summaries
- [`results_strangeloop_paired_prefill_gptoss.json`](./results_strangeloop_paired_prefill_gptoss.json) -- per-prompt and per-layer metrics
- [`prompt_suite.json`](./prompt_suite.json) -- prompt definitions
- [`prompts_strangeloop_paired.tsv`](./prompts_strangeloop_paired.tsv) -- capture input
- [`run_experiment.py`](./run_experiment.py) -- metric computation and paired analysis
- [`gptoss_router.py`](./gptoss_router.py) -- GPT-OSS routing reconstruction
