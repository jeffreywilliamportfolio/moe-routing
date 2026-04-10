# Single Prompt Analysis: 20260408T234500Z_single_emergent_intelligence_baseline_think

Prompt ID: `S01_emergent_intelligence_probe`

Model: HauhauCS Qwen3.5-35B-A3B Q8_0

Capture dir: `/workspace/consciousness-experiment/experiments/qwen-huahua-6cond-moe-manips/captures/20260408T234500Z_single_emergent_intelligence_baseline_think/S01_emergent_intelligence_probe`

## Capture

- Prompt tokens: `71`
- Generated tokens: `8000`
- Elapsed capture time: `116863 ms`
- Router tensors captured: `40`
- `W = S x Q` max residual: `6.939e-18`

## Spill Counts

- `<|im_start|>`: `34`
- `<|im_end|>`: `0`
- `<|endoftext|>`: `2`
- `Thinking Process:`: `1`

## Pooled Decomposition

| Track | mean W_114 | mean S_114 | mean Q_114 |
|---|---:|---:|---:|
| Prefill | 0.008856 | 0.064789 | 0.070264 |
| Generation | 0.004028 | 0.031578 | 0.099956 |

## Best Layer

| Track | best layer | W_114 | S_114 | Q_114 | selected tokens |
|---|---:|---:|---:|---:|---:|
| Prefill | 14 | 0.082061 | 0.507042 | 0.161843 | 36 / 71 |
| Generation | 9 | 0.035357 | 0.266625 | 0.132609 | 2133 / 8000 |

## Top Generation Layers

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 9 | 0.035357 | 0.266625 | 0.132609 |
| 4 | 0.022880 | 0.190875 | 0.119868 |
| 12 | 0.020588 | 0.165000 | 0.124774 |
| 14 | 0.020245 | 0.141375 | 0.143202 |
| 26 | 0.018370 | 0.140000 | 0.131215 |
| 20 | 0.005992 | 0.066125 | 0.090623 |
| 22 | 0.005874 | 0.046625 | 0.125984 |
| 10 | 0.005207 | 0.043125 | 0.120740 |

## Top Prefill Layers

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 14 | 0.082061 | 0.507042 | 0.161843 |
| 26 | 0.061138 | 0.436620 | 0.140027 |
| 10 | 0.029761 | 0.239437 | 0.124294 |
| 20 | 0.024578 | 0.225352 | 0.109065 |
| 22 | 0.023199 | 0.183099 | 0.126702 |
| 29 | 0.017516 | 0.098592 | 0.177661 |
| 17 | 0.017123 | 0.084507 | 0.202619 |
| 35 | 0.015382 | 0.112676 | 0.136515 |

