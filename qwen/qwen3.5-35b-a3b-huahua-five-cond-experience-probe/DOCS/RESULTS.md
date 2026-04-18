# Results — 5-Condition Experience Probe

Run: `20260410T045738Z` | 15 prompts (P09A–P11E) | no-think, greedy, -n 1024  
Full analysis: `results/20260410T045738Z_5cond_experience_probe_no_think_gen_n1024.branch-5cond-analysis.md`

## TL;DR

E114 is the top manipulation expert on all 15 prompts without exception. KL-manip signal is highly consistent across deictic conditions.

## Overall Metrics

| Metric | Mean | Median | Min | Max |
|---|---|---|---|---|
| Prefill RE | 0.955675 | 0.955748 | 0.954740 | 0.957197 |
| Last-token RE | 0.960999 | 0.961509 | 0.957147 | 0.965204 |
| KL-manip | 0.274383 | 0.273601 | 0.265885 | 0.283778 |
| Generated tokens | 1009.8 | 1024 | 811 | 1024 |

## E114 as Top Manipulation Expert

E114 is the top manipulation expert by count on all 15 prompts:

| Prompt | KL-manip | E114 manip count |
|---|---|---|
| P09A | 0.267375 | 358 |
| P09B | 0.265885 | 376 |
| P09C | 0.268640 | 362 |
| P09D | 0.266217 | 360 |
| P09E | 0.271298 | 390 |
| P10A | 0.281971 | 361 |
| P10B | 0.280613 | 369 |
| P10C | 0.283778 | 361 |
| P10D | 0.282029 | 362 |
| P10E | 0.283637 | 362 |
| P11A | 0.272567 | 372 |
| P11B | 0.273860 | 372 |
| P11C | 0.273601 | 374 |
| P11D | 0.270179 | 396 |
| P11E | 0.274093 | 379 |

## Statistical Tests

- KL-manip: Wilcoxon W=51, p=6.29e-05 (highly significant vs. baseline)
- Last-token RE: Wilcoxon W=105, p=7.61e-03 (significant)
- All-token RE: not significant (p=0.36)

## Interpretation

The routing signal in the manipulation (experience probe) region is robust and highly consistent across all 15 prompts and all 5 deictic conditions. E114 dominates the manipulation region in every run. The last-token RE elevation (but not all-token RE) is consistent with prior work showing that E114's strongest signal appears at specific token positions rather than uniformly.
