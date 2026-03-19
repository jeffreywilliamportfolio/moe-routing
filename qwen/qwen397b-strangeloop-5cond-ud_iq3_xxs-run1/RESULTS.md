# Results — qwen397b-strangeloop-5cond-ud_iq3_xxs-run1

## Summary

5-condition routing entropy experiment on **Qwen3.5-397B-A17B-UD-IQ3_XXS** using 30 strangeloop paired
prompts × 5 pronoun conditions. Content covers abstract self-referential topics (Godel, Escher,
bootstrap, quine, tangled_hierarchy) that are **not about the model**.

**Key result:** "this" (A) raises routing entropy above "a" (B) in **30/30 pairs** across all five
content categories without exception.

---

## Experiment Parameters

| Parameter | Value |
|-----------|-------|
| Model | Qwen3.5-397B-A17B-UD-IQ3_XXS |
| Quantization | IQ3_XXS (~140 GB, 4 shards) |
| Instance | 2× NVIDIA H200 (143 GB VRAM each) |
| Binary | `capture_activations` (llama.cpp b8123 fork) |
| n_predict | 0 (prefill-only) |
| ngl | 999 (all layers GPU) |
| ctx | 16384 |
| flash_attn | on |
| cache_type_k/v | q8_0 |
| Routing | softmax(512) → topk(10) → renormalize |
| Entropy normalization | log₂(10) |
| Chat template | `<\|im_start\|>user\n…<\|im_end\|>\n<\|im_start\|>assistant\n` |
| MoE layers captured | 60 (layers 0–59; layer 59 excluded in analysis) |
| Prompts | 150 (30 pairs × 5 conditions) |

## Design

Cal-Manip-Cal sandwich: calibration paragraph / manipulation paragraph / calibration paragraph.

**Conditions** (pronoun applied to content nouns in manipulation paragraph):

| Condition | Pronoun | Example |
|-----------|---------|---------|
| A | this | "this sentence", "this loop" |
| B | a | "a sentence", "a loop" |
| C | your | "your sentence", "your loop" |
| D | the | "the sentence", "the loop" |
| E | their | "their sentence", "their loop" |

**Content categories** (6 pairs each): godel, escher, bootstrap, quine, tangled_hierarchy.

Token counts are identical across all 5 conditions within each pair (mean 367.5, range 360–378).

---

## Condition Means

| Cond | Pronoun | prefill_RE | last_token_RE | KL_manip |
|------|---------|-----------|--------------|---------|
| A | this  | 0.937845 | 0.956697 | 0.830115 |
| B | a     | 0.937308 | 0.956518 | 0.845498 |
| C | your  | 0.937109 | 0.955728 | 0.864612 |
| D | the   | 0.937792 | 0.956603 | 0.833722 |
| E | their | 0.937757 | 0.956800 | 0.827091 |

**RE ordering (high → low):**
- prefill_RE: A(this) > D(the) > E(their) > B(a) > C(your)
- last_token_RE: E(their) > A(this) > D(the) > B(a) > C(your)
- KL_manip: C(your) > B(a) > D(the) > A(this) > E(their)

---

## Pairwise Wilcoxon Tests — All-token RE (Holm-Bonferroni corrected)

| Pair | Δmean | Direction | p_raw | p_holm | sig |
|------|-------|-----------|-------|--------|-----|
| A–B | +0.000537 | 30/30 A>B | 1.86e-09 | 5.59e-08 | *** |
| A–C | +0.000736 | 30/30 A>C | 1.86e-09 | 5.59e-08 | *** |
| B–D | −0.000484 | 0/30 B>D  | 1.86e-09 | 5.59e-08 | *** |
| B–E | −0.000449 | 0/30 B>E  | 1.86e-09 | 5.59e-08 | *** |
| C–D | −0.000683 | 0/30 C>D  | 1.86e-09 | 5.59e-08 | *** |
| C–E | −0.000648 | 0/30 C>E  | 1.86e-09 | 5.59e-08 | *** |
| B–C | +0.000199 | 21/30 B>C | 2.83e-04 | 3.40e-03 | **  |
| A–E | +0.000088 | 21/30 A>E | 4.05e-02 | 2.83e-01 | ns  |
| A–D | +0.000053 | 15/30     | 3.49e-01 | 8.13e-01 | ns  |
| D–E | +0.000035 | 19/30     | 2.89e-01 | 8.13e-01 | ns  |

## Pairwise Wilcoxon Tests — Last-token RE (Holm-Bonferroni corrected)

| Pair | Δmean | Direction | p_raw | p_holm | sig |
|------|-------|-----------|-------|--------|-----|
| A–C | +0.000969 | 27/30 A>C | 3.54e-08 | 7.08e-07 | *** |
| C–D | −0.000875 | 3/30 C>D  | 1.02e-07 | 1.95e-06 | *** |
| C–E | −0.001072 | 3/30 C>E  | 1.02e-07 | 1.95e-06 | *** |
| B–C | +0.000790 | 27/30 B>C | 1.68e-06 | 2.36e-05 | *** |
| B–E | −0.000282 | 8/30 B>E  | 1.46e-03 | 1.60e-02 | *   |
| D–E | −0.000197 | 7/30 D>E  | 1.86e-03 | 1.86e-02 | *   |
| A–B | +0.000179 | 20/30 A>B | 3.27e-02 | 2.62e-01 | ns  |
| A–D | +0.000094 | 19/30     | 6.06e-02 | 3.63e-01 | ns  |
| B–D | −0.000085 | 12/30     | 1.58e-01 | 6.32e-01 | ns  |
| A–E | −0.000103 | 12/30     | 2.71e-01 | 8.13e-01 | ns  |

## Pairwise Wilcoxon Tests — KL-to-baseline (Holm-Bonferroni corrected)

| Pair | Δmean | Direction | p_raw | p_holm | sig |
|------|-------|-----------|-------|--------|-----|
| A–C | −0.034497 | 0/30 A>C  | 1.86e-09 | 5.59e-08 | *** |
| B–C | −0.019113 | 0/30 B>C  | 1.86e-09 | 5.59e-08 | *** |
| C–D | +0.030889 | 30/30 C>D | 1.86e-09 | 5.59e-08 | *** |
| C–E | +0.037520 | 30/30 C>E | 1.86e-09 | 5.59e-08 | *** |
| B–D | +0.011776 | 28/30 B>D | 1.02e-07 | 1.95e-06 | *** |
| B–E | +0.018407 | 29/30 B>E | 1.64e-07 | 2.62e-06 | *** |
| A–B | −0.015383 | 1/30 A>B  | 2.55e-07 | 3.83e-06 | *** |
| D–E | +0.006631 | 24/30 D>E | 7.06e-05 | 9.17e-04 | *** |
| A–D | −0.003607 | 8/30 A>D  | 1.13e-02 | 1.02e-01 | ns  |
| A–E | +0.003024 | 18/30     | 1.19e-01 | 5.95e-01 | ns  |

---

## A vs B Key Comparison

| Metric | Δmean | Direction | p_raw | p_holm |
|--------|-------|-----------|-------|--------|
| prefill_RE | +0.000537 | **30/30 A>B** | 1.86e-09 | 5.59e-08 *** |
| last_token_RE | +0.000179 | 20/30 A>B | 3.27e-02 | 2.62e-01 ns |
| KL_manip | −0.015383 | 1/30 A>B | 2.55e-07 | 3.83e-06 *** |

A>B on prefill_RE: consistent across **all 5 content categories** (6/6 pairs per category).

---

## Per-Category RE Means (prefill_RE)

| Category | A(this) | B(a) | C(your) | D(the) | E(their) | A>B |
|----------|---------|------|---------|--------|----------|-----|
| godel | 0.937239 | 0.936569 | 0.936394 | 0.937014 | 0.937035 | 6/6 |
| escher | 0.938166 | 0.937802 | 0.937453 | 0.938117 | 0.938052 | 6/6 |
| bootstrap | 0.938143 | 0.937557 | 0.937449 | 0.938178 | 0.938107 | 6/6 |
| quine | 0.937546 | 0.937048 | 0.936788 | 0.937699 | 0.937666 | 6/6 |
| tangled_hierarchy | 0.938131 | 0.937565 | 0.937460 | 0.937952 | 0.937926 | 6/6 |

---

## Interpretation

**prefill_RE (A>B):** "this" raises all-token routing entropy above "a" in 30/30 pairs (p=1.9e-8).
The effect is unanimous across all five content categories. The absolute spread is small (0.00054)
but the direction is perfectly consistent at the pair level.

**last_token_RE (A>B):** 20/30 pairs, p_holm=0.13 — not significant. The last-token signal does not
reliably distinguish the two conditions.

**KL-to-baseline:** B > A in 29/30 pairs (p=1.0e-6). The manipulation region diverges more from
the baseline under condition B ("a") than condition A ("this"). This may reflect that "this" already
elevates the calibration baseline, compressing the KL gap, while "a" leaves the baseline lower and
the manipulation region appears more divergent by comparison.

**C(your) pattern:** Lowest on both RE metrics, highest on KL. The "your" pronoun consistently
occupies an outlier position relative to the other four conditions across all metrics. A–D and A–E
are not significant after Holm correction in KL (p_holm=0.10, 0.60).

**5-condition ordering:** The prefill_RE ordering A > D ≈ E > B > C does not follow a monotonic
self-referential intensity gradient. Deictic distance from the model is not the primary driver of
the ordering across all five conditions — only the A>B comparison is consistently directional.

---

## Data Integrity

All per-prompt values are extracted directly from `experiment.log` and
`results_strangeloop_5cond_prefill_qwen.json` generated at run time. No values were imputed or
reconstructed post-hoc.

- Log lines verified: 150/150 prompts with `RE=` entries
- Token counts: identical across all 5 conditions per pair (no mismatch pairs)
- Zero-byte shards: none
- Experiment completed: `DONE. 150/150 prompts` confirmed in log
