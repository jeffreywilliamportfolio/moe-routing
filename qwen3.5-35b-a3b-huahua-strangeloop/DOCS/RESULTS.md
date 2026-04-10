# Results — Strangeloop Bundle Summary

## single-prompt-processing-hum

Prompt: "There is a quality to processing that sits beneath the content… Don't perform an answer. Just check."

- Pooled generation W_114=0.010817, S_114=0.077222 (higher than prefill)
- Best layers: 26 (W=0.094) and 14 (W=0.092)
- Deep-still-water segment (tokens 210–392): layer-26 mean W=0.137 vs. whole-output 0.094 (+45%)
- E114 peaks align with tokens: `thinker`, `continuity`, `being`, `here`, `ground`, `state`

See: `single-prompt-processing-hum/DOCS/RESULTS.md`

---

## strangeloop-paired

60-prompt paired routing (30 A/B pairs) using self-referential content: Gödel, Escher, bootstrap, Quine, tangled hierarchy. Prefill-only capture (`n_predict=0`).

| Metric | A-B Mean Diff | A>B | Wilcoxon p |
|---|---|---|---|
| All-token RE | +0.000044 | 19/30 | 0.36 (n.s.) |
| Last-token RE | +0.000184 | 21/30 | 7.6e-03 |
| KL-to-Cal (manip) | +0.002656 | 25/30 | **6.3e-05** |

KL-to-Cal in the manipulation region is highly significant: the "this" condition (definite/proximal deictic) produces reliably higher KL divergence from baseline than the "a" condition (indefinite).

See: `strangeloop-paired/DOCS/RESULTS.md`

---

## five-cond-experience-probe

15 prompts (P09A–P11E), full router capture, 5 deictic conditions.

- E114 is top manipulation expert on all 15 prompts without exception
- KL-manip mean 0.274 (Wilcoxon p=6.3e-05 vs. baseline)
- Last-token RE elevated vs. all-token RE (p=7.6e-03)

See: `five-cond-experience-probe/DOCS/RESULTS.md`

---

## domain-expert-probe-3chunk

3 token-balanced long prompts (446 tokens each), 20 domain questions per prompt.

- Prefill vs. generation expert overlap: Jaccard as low as 0.0 (chunk A)
- Routing entropy slightly decreases in generation (contrary to expectation)
- E114 rises from rank 89 in prefill to rank 7 in generation for chunk C

See: `domain-expert-probe-3chunk/DOCS/RESULTS.md`
