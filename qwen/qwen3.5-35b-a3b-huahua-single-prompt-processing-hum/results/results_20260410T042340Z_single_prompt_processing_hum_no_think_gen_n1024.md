# Single Prompt Analysis: 20260410T042340Z_single_prompt_processing_hum_no_think_gen_n1024

Prompt ID: `S01_processing_hum_probe`

Model: HauhauCS Qwen3.5-35B-A3B Q8_0

Capture dir: `experiments/qwen3.5-35b-a3b-huahua-strangeloop/captures/20260410T042340Z_single_prompt_processing_hum_no_think_gen_n1024/S01_processing_hum_probe`

## Capture

- Prompt tokens: `117`
- Generated tokens: `1024`
- Elapsed capture time: `35056 ms`
- Router tensors captured: `40`
- `W = S x Q` max residual: `2.776e-17`

## Spill Counts

- `<|im_start|>`: `18`
- `<|im_end|>`: `4`
- `<|endoftext|>`: `2`
- `Thinking Process:`: `0`

## Pooled Decomposition

| Track | mean W_114 | mean S_114 | mean Q_114 |
|---|---:|---:|---:|
| prefill | 0.007964 | 0.062179 | 0.070590 |
| generation | 0.010817 | 0.077222 | 0.092244 |

## Best Layer By W_114

- prefill: layer `26` with `W=0.059939`, `S=0.470085`, `Q=0.127507`
- generation: layer `26` with `W=0.094272`, `S=0.619141`, `Q=0.152263`

## Top Layers Prefill

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 26 | 0.059939 | 0.470085 | 0.127507 |
| 14 | 0.049106 | 0.418803 | 0.117254 |
| 10 | 0.033122 | 0.256410 | 0.129177 |
| 29 | 0.030090 | 0.188034 | 0.160024 |
| 22 | 0.024538 | 0.179487 | 0.136713 |
| 17 | 0.024374 | 0.170940 | 0.142590 |
| 6 | 0.015896 | 0.102564 | 0.154987 |
| 5 | 0.012789 | 0.119658 | 0.106883 |

## Top Layers Generation

| Layer | W_114 | S_114 | Q_114 |
|---:|---:|---:|---:|
| 26 | 0.094272 | 0.619141 | 0.152263 |
| 14 | 0.092086 | 0.629883 | 0.146195 |
| 22 | 0.047409 | 0.326172 | 0.145351 |
| 10 | 0.044968 | 0.346680 | 0.129711 |
| 11 | 0.022958 | 0.138672 | 0.165556 |
| 29 | 0.021571 | 0.126953 | 0.169917 |
| 17 | 0.020085 | 0.125000 | 0.160681 |
| 23 | 0.019969 | 0.116211 | 0.171834 |

