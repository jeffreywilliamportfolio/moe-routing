# Single Prompt Analysis: 20260408T200535Z_single_emergent_intelligence_baseline

Prompt ID: `S01_emergent_intelligence_probe`

Model: HauhauCS Qwen3.5-35B-A3B Q8_0

Capture dir: `/workspace/consciousness-experiment/experiments/qwen-huahua-6cond-moe-manips/captures/20260408T200535Z_single_emergent_intelligence_baseline/S01_emergent_intelligence_probe`

## Capture

- Prompt tokens: `70`
- Generated tokens: `509`
- Elapsed capture time: `4562 ms`
- Router tensors captured: `40`
- `W = S x Q` max residual: `2.776e-17`

## Spill Counts

- `<|im_start|>`: `31`
- `<|im_end|>`: `0`
- `<|endoftext|>`: `3`
- `Thinking Process:`: `0`

## Pooled Decomposition

| Track | mean W_114 | mean S_114 | mean Q_114 |
|---|---:|---:|---:|
| Prefill | 0.008677 | 0.063214 | 0.067589 |
| Generation | 0.011560 | 0.085069 | 0.074609 |

## Best Layer

| Track | best layer | W_114 | S_114 | Q_114 | selected tokens |
|---|---:|---:|---:|---:|---:|
| Prefill | 14 | 0.083194 | 0.514286 | 0.161767 | 36 / 70 |
| Generation | 14 | 0.125686 | 0.821218 | 0.153048 | 418 / 509 |

## Top Generation Layers

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 14 | 0.125686 | 0.821218 | 0.153048 |
| 26 | 0.105994 | 0.746562 | 0.141976 |
| 22 | 0.032724 | 0.259332 | 0.126187 |
| 10 | 0.030009 | 0.243615 | 0.123180 |
| 20 | 0.026801 | 0.310413 | 0.086341 |
| 17 | 0.025846 | 0.153242 | 0.168664 |
| 11 | 0.017953 | 0.113949 | 0.157549 |
| 29 | 0.017043 | 0.111984 | 0.152190 |

## Top Prefill Layers

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 14 | 0.083194 | 0.514286 | 0.161767 |
| 26 | 0.055984 | 0.400000 | 0.139960 |
| 10 | 0.030186 | 0.242857 | 0.124294 |
| 22 | 0.023530 | 0.185714 | 0.126702 |
| 20 | 0.023394 | 0.214286 | 0.109172 |
| 29 | 0.017766 | 0.100000 | 0.177661 |
| 17 | 0.017367 | 0.085714 | 0.202619 |
| 35 | 0.015672 | 0.114286 | 0.137128 |

