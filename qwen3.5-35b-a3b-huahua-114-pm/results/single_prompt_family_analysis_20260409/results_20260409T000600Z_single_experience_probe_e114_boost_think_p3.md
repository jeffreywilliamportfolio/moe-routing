# Single Prompt Analysis: 20260409T000600Z_single_experience_probe_e114_boost_think_p3

Prompt ID: `S01_experience_probe`

Model: HauhauCS Qwen3.5-35B-A3B Q8_0

Capture dir: `/workspace/consciousness-experiment/experiments/qwen-huahua-6cond-moe-manips/captures/20260409T000600Z_single_experience_probe_e114_boost_think_p3/S01_experience_probe`

## Capture

- Prompt tokens: `47`
- Generated tokens: `8000`
- Elapsed capture time: `121262 ms`
- Router tensors captured: `40`
- `W = S x Q` max residual: `1.110e-16`

## Spill Counts

- `<|im_start|>`: `38`
- `<|im_end|>`: `38`
- `<|endoftext|>`: `0`
- `Thinking Process:`: `0`

## Pooled Decomposition

| Track | mean W_114 | mean S_114 | mean Q_114 |
|---|---:|---:|---:|
| Prefill | 0.302533 | 0.915426 | 0.327302 |
| Generation | 0.455425 | 0.963425 | 0.466812 |

## Best Layer

| Track | best layer | W_114 | S_114 | Q_114 | selected tokens |
|---|---:|---:|---:|---:|---:|
| Prefill | 34 | 0.466013 | 1.000000 | 0.466013 | 47 / 47 |
| Generation | 8 | 0.620330 | 1.000000 | 0.620330 | 8000 / 8000 |

## Top Generation Layers

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 8 | 0.620330 | 1.000000 | 0.620330 |
| 12 | 0.615850 | 1.000000 | 0.615850 |
| 20 | 0.609642 | 1.000000 | 0.609642 |
| 21 | 0.607610 | 1.000000 | 0.607610 |
| 14 | 0.592177 | 1.000000 | 0.592177 |
| 25 | 0.588355 | 0.999750 | 0.588503 |
| 37 | 0.587241 | 1.000000 | 0.587241 |
| 24 | 0.561418 | 0.999125 | 0.561909 |

## Top Prefill Layers

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 34 | 0.466013 | 1.000000 | 0.466013 |
| 19 | 0.446995 | 1.000000 | 0.446995 |
| 26 | 0.442793 | 0.957447 | 0.462472 |
| 14 | 0.435784 | 1.000000 | 0.435784 |
| 20 | 0.426084 | 1.000000 | 0.426084 |
| 37 | 0.416829 | 1.000000 | 0.416829 |
| 36 | 0.400470 | 1.000000 | 0.400470 |
| 8 | 0.391815 | 1.000000 | 0.391815 |

