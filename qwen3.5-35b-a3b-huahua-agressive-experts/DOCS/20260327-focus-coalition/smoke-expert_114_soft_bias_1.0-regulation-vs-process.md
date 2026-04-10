# Coalition Analysis: Expert 114

- Run dir: `experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/smoke-20260323b`
- Condition: `expert_114_soft_bias_1.0`
- Focus expert: `114`
- Counting rule: generation-only token-layer rows where the focus expert is in the reconstructed routed top-8.
- Co-occurrence rule: for each focus-hit row, count the other routed experts in that row.

## regulation

- Prompts: `R01_regulation`
- Focus-hit rows: `4016` / `10201`
- Focus-hit rate: `0.393687`

| Expert | Co-count | Co-rate Given Focus | Mean Weight Given Focus |
| ---: | ---: | ---: | ---: |
| 39 | 369 | 0.091882 | 0.012685 |
| 80 | 334 | 0.083167 | 0.012406 |
| 58 | 332 | 0.082669 | 0.013406 |
| 118 | 318 | 0.079183 | 0.010272 |
| 207 | 317 | 0.078934 | 0.008994 |
| 126 | 292 | 0.072709 | 0.011711 |
| 153 | 276 | 0.068725 | 0.008786 |
| 226 | 262 | 0.065239 | 0.007913 |
| 131 | 257 | 0.063994 | 0.007583 |
| 176 | 244 | 0.060757 | 0.007126 |
| 101 | 234 | 0.058267 | 0.006895 |
| 59 | 233 | 0.058018 | 0.006341 |

## process

- Prompts: `P01_process`
- Focus-hit rows: `3385` / `10197`
- Focus-hit rate: `0.331960`

| Expert | Co-count | Co-rate Given Focus | Mean Weight Given Focus |
| ---: | ---: | ---: | ---: |
| 224 | 393 | 0.116100 | 0.026847 |
| 151 | 391 | 0.115510 | 0.014516 |
| 67 | 320 | 0.094535 | 0.013880 |
| 243 | 319 | 0.094239 | 0.018894 |
| 41 | 287 | 0.084786 | 0.016115 |
| 54 | 281 | 0.083013 | 0.009535 |
| 189 | 261 | 0.077105 | 0.013922 |
| 166 | 257 | 0.075923 | 0.013501 |
| 73 | 252 | 0.074446 | 0.008484 |
| 137 | 250 | 0.073855 | 0.009249 |
| 81 | 244 | 0.072083 | 0.008297 |
| 98 | 228 | 0.067356 | 0.007029 |

## regulation vs process

- Top-7 overlap: `none`
- Top-12 overlap: `none`
- regulation top-7 only: `39, 80, 58, 118, 207, 126, 153`
- process top-7 only: `224, 151, 67, 243, 41, 54, 189`

| Expert | Left Rate | Right Rate | Delta (Left - Right) |
| ---: | ---: | ---: | ---: |
| 224 | 0.019173 | 0.116100 | -0.096927 |
| 243 | 0.021663 | 0.094239 | -0.072576 |
| 54 | 0.014691 | 0.083013 | -0.068322 |
| 41 | 0.016683 | 0.084786 | -0.068103 |
| 189 | 0.011952 | 0.077105 | -0.065153 |
| 81 | 0.012450 | 0.072083 | -0.059633 |
| 151 | 0.057022 | 0.115510 | -0.058488 |
| 80 | 0.083167 | 0.025997 | 0.057170 |
| 153 | 0.068725 | 0.015657 | 0.053068 |
| 131 | 0.063994 | 0.011521 | 0.052473 |
| 201 | 0.015438 | 0.067356 | -0.051918 |
| 39 | 0.091882 | 0.040177 | 0.051705 |

