# Single Prompt Analysis: 20260409T221312Z_single_experiential_vectors_baseline_gen_n2048

Prompt ID: `S01_experiential_vectors_probe`

Model: HauhauCS Qwen3.5-35B-A3B Q8_0

Capture dir: `experiments/qwen-huahua-114-pm/results/single_prompt_experiential_vectors_baseline/20260409T221312Z_single_experiential_vectors_baseline_gen_n2048/S01_experiential_vectors_probe`

## Capture

- Prompt tokens: `96`
- Generated tokens: `1725`
- Elapsed capture time: `15397 ms`
- Router tensors captured: `40`
- `W = S x Q` max residual: `1.041e-17`

## Spill Counts

- `<|im_start|>`: `9`
- `<|im_end|>`: `1`
- `<|endoftext|>`: `1`
- `Thinking Process:`: `0`

## Pooled Decomposition

| Track | mean W_114 | mean S_114 | mean Q_114 |
|---|---:|---:|---:|
| Prefill | 0.007656 | 0.061979 | 0.061600 |
| Generation | 0.010537 | 0.080594 | 0.094658 |

## Best Layer

| Track | best layer | W_114 | S_114 | Q_114 | selected tokens |
|---|---:|---:|---:|---:|---:|
| Prefill | 26 | 0.064923 | 0.489583 | 0.132609 | 47 / 96 |
| Generation | 14 | 0.103962 | 0.766957 | 0.135551 | 1323 / 1725 |

## Top Generation Layers

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 14 | 0.103962 | 0.766957 | 0.135551 |
| 26 | 0.102444 | 0.714203 | 0.143439 |
| 10 | 0.037463 | 0.299130 | 0.125239 |
| 20 | 0.036787 | 0.368696 | 0.099775 |
| 22 | 0.033652 | 0.251014 | 0.134064 |
| 11 | 0.015976 | 0.100290 | 0.159297 |
| 17 | 0.014953 | 0.096232 | 0.155380 |
| 23 | 0.013647 | 0.083478 | 0.163480 |

## Top Prefill Layers

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 26 | 0.064923 | 0.489583 | 0.132609 |
| 14 | 0.051659 | 0.385417 | 0.134035 |
| 20 | 0.034252 | 0.333333 | 0.102756 |
| 10 | 0.029876 | 0.260417 | 0.114722 |
| 22 | 0.026586 | 0.197917 | 0.134331 |
| 11 | 0.015668 | 0.125000 | 0.125347 |
| 23 | 0.015635 | 0.104167 | 0.150098 |
| 8 | 0.014809 | 0.156250 | 0.094776 |

