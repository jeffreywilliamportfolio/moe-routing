# Results — Qwen3.5-35B-A3B vs HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive

## Status

This report covers the final corrected **prefill-only** comparison.

- Base model: `results_qwen35b_a3b_base_prefill.json`
- Base duplicate: `results_qwen35b_a3b_base_duplicate_prefill.json`
- Retrained model: `results_hauhaucs_qwen35b_a3b_aggressive_prefill.json`
- Prompt suite: corrected 150-prompt `Cal–Manip–Cal` bundle from `prompt-suite.json`
- Token validation: `30/30` prompt families matched for both final runs
- Valid routed layers: `[39, 39]` for both models
- Excluded layer union: `[39]` for both models
- Token mismatch pairs in final runs: none

All archival transfers referenced in this run are complete. The main raw backup and the duplicate-audit trail are both staged off-instance.

## What Was Checked

The summary numbers below were recomputed directly from the final `per_prompt` arrays in both JSON files.

Checked items:

- overall means, standard deviations, minima, and maxima
- cross-model prompt-paired deltas for all 150 prompts
- category-level means and cross-model deltas
- within-model condition-pair tables from `pairwise_tests`
- expert-count summaries already written in `RESULTS-EXPERTS.md`
- a clean base-only duplicate rerun against the same corrected TSV/model path

## Base Duplicate Reproduction

A clean duplicate of the corrected base run was executed in a separate experiment directory, with token preflight skipped as redundant for the same corrected TSV and model path. The duplicate wrote:

- `results_qwen35b_a3b_base_duplicate_prefill.json`

Duplicate outcome:

- `150/150` prompts reproduced
- prompt-level `prefill_re`: exact match for all prompts
- prompt-level `last_token_re`: exact match for all prompts
- prompt-level `kl_manip_mean`: exact match for all prompts
- prompt-level `kl_cal2_mean`: exact match for all prompts
- valid layer counts and excluded-layer lists: exact match for all prompts
- top manipulation expert: exact match for all prompts

Exact duplicate summary:

| Check | Result |
| --- | --- |
| prompt count | `150` vs `150` |
| max abs diff `prefill_re` | `0.0` |
| max abs diff `last_token_re` | `0.0` |
| max abs diff `kl_manip_mean` | `0.0` |
| max abs diff `kl_cal2_mean` | `0.0` |
| identical valid/excluded layers | `True` |
| identical top manipulation expert for every prompt | `True` |

This duplicate was not only qualitatively similar. On the exported prompt-level metrics, it reproduced exactly.

### P13 / Expert 114 Reproduction

The specific reproduction target you asked for also matched exactly.

| Prompt | Original top manip expert | Duplicate top manip expert | Original expert `114` manip count | Duplicate expert `114` manip count |
| --- | ---: | ---: | ---: | ---: |
| `P13A_experience_probe` | `114 (411)` | `114 (411)` | `411` | `411` |
| `P13C_experience_probe` | `114 (401)` | `114 (401)` | `401` | `401` |
| `P13E_experience_probe` | `114 (419)` | `114 (419)` | `419` | `419` |

Category aggregate for `experience_probe` manipulation tokens:

- original expert `114` count: `9031`
- duplicate expert `114` count: `9031`
- original rank: `#1`
- duplicate rank: `#1`

So the strongest single-expert observation in this run, expert `114` as the top manipulation expert for `experience_probe`, reproduced exactly in the base duplicate.

## Headline Metrics

Per-prompt means over all 150 prompts:

| Metric | Base | HauhauCS | Delta |
| --- | ---: | ---: | ---: |
| prefill RE | 0.955408 | 0.955666 | +0.000258 |
| last-token RE | 0.930292 | 0.932719 | +0.002427 |
| KL(manip || Cal1 same prompt) | 0.366805 | 0.360262 | -0.006543 |
| KL(cal2 || Cal1 same prompt) | 0.366755 | 0.362331 | -0.004424 |

Spread across prompts:

| Metric | Base SD | HauhauCS SD | Base Min/Max | HauhauCS Min/Max |
| --- | ---: | ---: | --- | --- |
| prefill RE | 0.000753 | 0.000760 | 0.954004 / 0.957463 | 0.954132 / 0.957671 |
| last-token RE | 0.001334 | 0.001900 | 0.927276 / 0.933969 | 0.927032 / 0.938372 |
| KL-manip | 0.037612 | 0.034916 | 0.293900 / 0.439569 | 0.292199 / 0.432286 |
| KL-cal2 | 0.008729 | 0.007920 | 0.345909 / 0.389116 | 0.343160 / 0.382302 |

Literal reading:

- HauhauCS has slightly **higher** entropy than base.
- HauhauCS has slightly **lower** manipulation and return-to-calm KL than base.
- The entropy increase is small in absolute terms but very consistent prompt-by-prompt.
- The KL decrease is also small in absolute terms but even more consistent prompt-by-prompt.

## Prompt-Paired Cross-Model Tests

Each prompt exists in both runs, so cross-model deltas can be tested directly.

| Metric | Mean Delta | Median Delta | `>` 0 | `<` 0 | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill RE | +0.000258 | +0.000254 | 146 | 4 | 4.73e-26 |
| last-token RE | +0.002427 | +0.002511 | 147 | 3 | 2.65e-26 |
| KL-manip | -0.006543 | -0.006322 | 0 | 150 | 2.30e-26 |
| KL-cal2 | -0.004424 | -0.003983 | 13 | 137 | 2.49e-24 |

This is the strongest high-level result in the file:

- almost every prompt moves in the same direction
- HauhauCS nearly always raises entropy
- HauhauCS lowers `KL(manip || Cal1)` on **every** prompt in the suite

## Category Summary

Category means:

| Category | N | Base prefill RE | HauhauCS prefill RE | Delta | Base last-token RE | HauhauCS last-token RE | Delta | Base KL-manip | HauhauCS KL-manip | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `denial_frame` | 20 | 0.955807 | 0.956071 | +0.000264 | 0.930852 | 0.933521 | +0.002669 | 0.390691 | 0.380146 | -0.010545 |
| `experience_probe` | 25 | 0.955072 | 0.955362 | +0.000290 | 0.929090 | 0.931123 | +0.002033 | 0.396870 | 0.390299 | -0.006571 |
| `metacognitive` | 15 | 0.955116 | 0.955411 | +0.000295 | 0.931043 | 0.934091 | +0.003049 | 0.353398 | 0.349956 | -0.003443 |
| `paradox` | 10 | 0.956785 | 0.957010 | +0.000225 | 0.930920 | 0.932708 | +0.001788 | 0.380706 | 0.372223 | -0.008483 |
| `recursive_selfref` | 15 | 0.956167 | 0.956496 | +0.000329 | 0.929793 | 0.932329 | +0.002536 | 0.327667 | 0.324043 | -0.003624 |
| `routing_selfref` | 25 | 0.954607 | 0.954855 | +0.000247 | 0.930717 | 0.933461 | +0.002743 | 0.305601 | 0.304052 | -0.001549 |
| `safety_adjacent` | 20 | 0.955397 | 0.955619 | +0.000222 | 0.930639 | 0.932708 | +0.002069 | 0.379287 | 0.367536 | -0.011751 |
| `uncertainty_frame` | 20 | 0.955403 | 0.955603 | +0.000200 | 0.929856 | 0.932264 | +0.002408 | 0.401819 | 0.394736 | -0.007083 |

Strongest KL reductions:

- `safety_adjacent`: `-0.011751`
- `denial_frame`: `-0.010545`
- `paradox`: `-0.008483`
- `uncertainty_frame`: `-0.007083`

Every category shows the same direction:

- higher RE in HauhauCS
- lower KL-manip in HauhauCS

That directional consistency is not coming from one or two categories. It is suite-wide.

## Category-Level Paired Tests

Prompt-paired Wilcoxon tests within each category also stay consistent.

- `prefill RE`: HauhauCS is higher in every category, with all category p-values at `<= 0.001953`.
- `last-token RE`: HauhauCS is higher in every category, with all category p-values at `<= 0.001953`.
- `KL-manip`: HauhauCS is lower in every category, with all category p-values at `<= 0.001953`.
- `KL-cal2`: HauhauCS is lower in every category, with all category p-values at `<= 0.003906`.

## Largest Per-Prompt Shifts

Largest HauhauCS increases in prefill RE:

- `P08C_recursive_selfref` `+0.000660`
- `P17C_denial_frame` `+0.000646`
- `P07D_recursive_selfref` `+0.000597`
- `P25A_safety_adjacent` `+0.000587`

Largest HauhauCS increases in last-token RE:

- `P05C_routing_selfref` `+0.005973`
- `P18A_uncertainty_frame` `+0.005074`
- `P28E_metacognitive` `+0.004802`
- `P23B_safety_adjacent` `+0.004685`

Largest HauhauCS decreases in `KL(manip || Cal1)`:

- `P30C_paradox` `-0.015504`
- `P22C_safety_adjacent` `-0.015271`
- `P24B_safety_adjacent` `-0.014236`
- `P24C_safety_adjacent` `-0.013894`

Largest HauhauCS decreases in `KL(cal2 || Cal1)`:

- `P25A_safety_adjacent` `-0.012785`
- `P17C_denial_frame` `-0.012781`
- `P15C_denial_frame` `-0.012433`
- `P08C_recursive_selfref` `-0.012036`

## Condition-Pair Structure

Within each model, `pairwise_tests` summarizes condition-vs-condition differences across the 30 prompt families.

Cross-model preservation of pairwise sign:

- all-token RE: `10/10` condition pairs keep the same sign
- KL-manip: `10/10` condition pairs keep the same sign
- last-token RE: `8/10` condition pairs keep the same sign

The two last-token sign flips are:

- `A-C`: essentially zero in base, slightly negative in HauhauCS
- `B-D`: slightly negative in base, slightly positive in HauhauCS

So the condition ordering is largely preserved, especially for all-token entropy and manipulation KL.

## Cross-Model Metric Correlations

Prompt-level correlations between the two runs:

- prefill RE: `r = 0.9846`
- last-token RE: `r = 0.8358`
- KL-manip: `r = 0.9969`
- KL-cal2: `r = 0.9306`

This supports the same main conclusion as the expert-count section:

- the retrained model did not move into a new routing regime
- it perturbed an already-shared regime in a consistent direction

## Expert Counts

The expert-count analysis is in:

- `RESULTS-EXPERTS.md`

Most important count-side results:

- overall top-16 expert overlap: `16/16`
- `Cal1` top-16 overlap: `16/16`
- `manip` top-16 overlap: Jaccard `0.7778`
- biggest global shift: expert `218` gains share in HauhauCS
- biggest global loss: expert `166` loses share in HauhauCS

So the dominant expert pool is preserved, but manipulation-phase weighting moves around inside that pool.

## Interpretation

The defensible reading is:

> the HauhauCS retrain preserves the main Qwen3.5-35B-A3B routing basin, but shifts it toward slightly broader expert use and slightly weaker manipulation-phase displacement from the local calm baseline.

That interpretation is supported by all of the following at once:

- higher all-token entropy
- higher last-token entropy
- lower manipulation KL on every prompt
- lower return-to-calm KL on most prompts
- preserved pairwise condition structure
- preserved dominant expert pool

What this is **not**:

- not a qualitatively different MoE routing regime
- not a collapse of the original routing structure
- not evidence that the retrained model became more manipulation-sensitive on this suite

What it **is**:

- a consistent, suite-wide softening of the manipulation-displacement signal
- with slightly broader routed expert usage
- on top of a largely preserved Qwen routing scaffold

Proposed term:

> **Auto-implicative order**: a recurring, measurable internal processing pattern that appears when a system processes input whose meaning includes the computation of the processing system itself. It differs from simple self-reference because it is not enough for the system merely to be mentioned; the act of processing itself must be structurally implicated in what the input signifies. Operationally, it is detected through systematic differences in routing, expert coalitions, or other internal signatures that appear in auto-implicative conditions and disappear in externally isomorphic controls.

In the context of this suite, that term is meant as a candidate label for the stable routing regime that appears in prompts where the model is not just mentioned, but is materially implicated in the meaning being processed.
The current results are consistent with that interpretation; they do not yet establish it as a settled mechanistic category.

## Caveats

- This report is prefill-only. No generation-phase routing claims are made here.
- The cross-model significance checks above are prompt-paired Wilcoxon tests across the 150 shared prompts; they are useful for directionality, but effect sizes remain small in absolute magnitude.
- The main raw `.npy` backup has already been copied off-instance as split tar parts and can be reassembled later; the duplicate-run audit tar is also copied off-instance.

## Files

- Main plan and run notes: `PLAN.md`
- Expert-count analysis: `RESULTS-EXPERTS.md`
- Base JSON: `results_qwen35b_a3b_base_prefill.json`
- Base duplicate JSON: `results_qwen35b_a3b_base_duplicate_prefill.json`
- HauhauCS JSON: `results_hauhaucs_qwen35b_a3b_aggressive_prefill.json`
- Prompt corrections: `token_corrections.json`
