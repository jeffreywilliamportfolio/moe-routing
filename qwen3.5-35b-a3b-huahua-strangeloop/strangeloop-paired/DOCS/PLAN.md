# Plan — Strangeloop Paired A/B Routing Contrast

## Goal

Test whether the definiteness deictic (A="this" vs. B="a") in a self-referential Cal–Manip–Cal prompt produces a measurable routing difference in HauhauCS Qwen3.5-35B-A3B Q8_0. Addresses Cameron Berg's 35B definiteness-control question.

## Model and Hardware

- Model: HauhauCS Qwen3.5-35B-A3B Q8_0
- Runtime: prefill-only (`n_predict=0`), greedy
- Routing reconstruction: `softmax_then_topk8_renorm`
- Entropy normalization: `log2(8)`

## Prompt Design

- 60 prompts: 30 A/B pairs
- Categories: godel (6 pairs), escher (6 pairs), bootstrap (6 pairs), quine (6 pairs), tangled_hierarchy (6 pairs)
- A = "this [content]..." (definite, proximal deictic)
- B = "a [content]..." (indefinite deictic)
- Structure: Cal–Manip–Cal sandwich; prefill-only capture

## Measurements

- All-token routing entropy (RE)
- Last-token routing entropy
- KL divergence from Cal1 baseline in the manipulation region (KL-manip)
- Wilcoxon signed-rank test on A-B differences for each metric
