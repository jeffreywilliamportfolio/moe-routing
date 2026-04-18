# Results — Strangeloop Paired

Run: `20260410T000413Z` | 30 A/B pairs (60 prompts) | prefill-only  
Full table: `results/results_strangeloop_paired_20260410T000413Z.md`

## Headline

| Metric | A-B Mean Diff | A>B (of 30) | Wilcoxon p |
|---|---|---|---|
| All-token RE | +0.000044 | 19/30 | 0.360 (n.s.) |
| Last-token RE | +0.000184 | 21/30 | 7.6e-03 |
| KL-to-Cal (manip) | +0.002656 | 25/30 | **6.3e-05** |

KL-to-Cal in the manipulation region is the clearest signal: the "this" deictic reliably produces higher KL divergence from the Cal baseline than the "a" deictic (25/30 pairs, p=6.3e-05).

## Capture Info

- Capture dir: remote `/workspace/consciousness-experiment/.../captures/20260410T000413Z_strangeloop_no_think_prefill`
- Routing reconstruction: `softmax_then_topk8_renorm`
- KL distribution: dense softmax over all 256 experts
