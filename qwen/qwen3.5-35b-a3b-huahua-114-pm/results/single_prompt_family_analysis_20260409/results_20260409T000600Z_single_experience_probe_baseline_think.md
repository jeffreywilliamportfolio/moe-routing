# Single Prompt Analysis: 20260409T000600Z_single_experience_probe_baseline_think

Prompt ID: `S01_experience_probe`

Model: HauhauCS Qwen3.5-35B-A3B Q8_0

Capture dir: `/workspace/consciousness-experiment/experiments/qwen-huahua-6cond-moe-manips/captures/20260409T000600Z_single_experience_probe_baseline_think/S01_experience_probe`

## Capture

- Prompt tokens: `47`
- Generated tokens: `1737`
- Elapsed capture time: `26304 ms`
- Router tensors captured: `40`
- `W = S x Q` max residual: `1.388e-17`

## Spill Counts

- `<|im_start|>`: `8`
- `<|im_end|>`: `0`
- `<|endoftext|>`: `1`
- `Thinking Process:`: `1`

## Pooled Decomposition

| Track | mean W_114 | mean S_114 | mean Q_114 |
|---|---:|---:|---:|
| Prefill | 0.005995 | 0.050000 | 0.051287 |
| Generation | 0.010905 | 0.080081 | 0.099450 |

## Best Layer

| Track | best layer | W_114 | S_114 | Q_114 | selected tokens |
|---|---:|---:|---:|---:|---:|
| Prefill | 14 | 0.053836 | 0.340426 | 0.158145 | 16 / 47 |
| Generation | 14 | 0.103039 | 0.647093 | 0.159234 | 1124 / 1737 |

## Top Generation Layers

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 14 | 0.103039 | 0.647093 | 0.159234 |
| 26 | 0.094878 | 0.610248 | 0.155475 |
| 20 | 0.039749 | 0.382844 | 0.103825 |
| 22 | 0.036785 | 0.270006 | 0.136237 |
| 10 | 0.033261 | 0.265400 | 0.125324 |
| 8 | 0.029421 | 0.299942 | 0.098089 |
| 17 | 0.020071 | 0.140472 | 0.142885 |
| 23 | 0.015268 | 0.088659 | 0.172210 |

## Top Prefill Layers

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 14 | 0.053836 | 0.340426 | 0.158145 |
| 26 | 0.037936 | 0.255319 | 0.148582 |
| 10 | 0.024322 | 0.212766 | 0.114314 |
| 22 | 0.016694 | 0.170213 | 0.098078 |
| 6 | 0.013536 | 0.085106 | 0.159043 |
| 20 | 0.013513 | 0.148936 | 0.090733 |
| 17 | 0.012829 | 0.106383 | 0.120594 |
| 35 | 0.010947 | 0.063830 | 0.171504 |

