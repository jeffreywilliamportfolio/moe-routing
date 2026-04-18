# Single Prompt Analysis: 20260409T000600Z_single_experience_probe_e114_boost_think_p2

Prompt ID: `S01_experience_probe`

Model: HauhauCS Qwen3.5-35B-A3B Q8_0

Capture dir: `/workspace/consciousness-experiment/experiments/qwen-huahua-6cond-moe-manips/captures/20260409T000600Z_single_experience_probe_e114_boost_think_p2/S01_experience_probe`

## Capture

- Prompt tokens: `47`
- Generated tokens: `8000`
- Elapsed capture time: `120856 ms`
- Router tensors captured: `40`
- `W = S x Q` max residual: `1.110e-16`

## Spill Counts

- `<|im_start|>`: `32`
- `<|im_end|>`: `22`
- `<|endoftext|>`: `2`
- `Thinking Process:`: `0`

## Pooled Decomposition

| Track | mean W_114 | mean S_114 | mean Q_114 |
|---|---:|---:|---:|
| Prefill | 0.146315 | 0.742021 | 0.195735 |
| Generation | 0.239587 | 0.868109 | 0.271022 |

## Best Layer

| Track | best layer | W_114 | S_114 | Q_114 | selected tokens |
|---|---:|---:|---:|---:|---:|
| Prefill | 14 | 0.289211 | 0.893617 | 0.323641 | 42 / 47 |
| Generation | 21 | 0.374401 | 1.000000 | 0.374401 | 8000 / 8000 |

## Top Generation Layers

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 21 | 0.374401 | 1.000000 | 0.374401 |
| 14 | 0.370946 | 0.986750 | 0.375927 |
| 8 | 0.369060 | 0.994375 | 0.371148 |
| 20 | 0.365820 | 0.998625 | 0.366324 |
| 12 | 0.349633 | 0.996750 | 0.350773 |
| 37 | 0.349518 | 0.999875 | 0.349562 |
| 24 | 0.348026 | 0.994875 | 0.349819 |
| 26 | 0.332667 | 0.975000 | 0.341197 |

## Top Prefill Layers

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 14 | 0.289211 | 0.893617 | 0.323641 |
| 26 | 0.271575 | 0.787234 | 0.344973 |
| 34 | 0.249625 | 1.000000 | 0.249625 |
| 20 | 0.243277 | 0.978723 | 0.248565 |
| 37 | 0.217177 | 1.000000 | 0.217177 |
| 19 | 0.217103 | 0.957447 | 0.226752 |
| 8 | 0.210149 | 1.000000 | 0.210149 |
| 36 | 0.195533 | 0.957447 | 0.204223 |

