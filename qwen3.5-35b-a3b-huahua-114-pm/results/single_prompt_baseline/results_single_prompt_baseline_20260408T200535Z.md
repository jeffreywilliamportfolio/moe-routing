# Single Prompt Baseline Results

Run: `20260408T200535Z`

Prompt TSV: [single_prompt_emergent_intelligence_no_think.tsv](/Users/jeffreyshorthill/llama-eeg-tests/experiments/qwen-huahua-6cond-moe-manips/prompts/single_prompt_emergent_intelligence_no_think.tsv)

Prompt ID: `S01_emergent_intelligence_probe`

Model: HauhauCS `Qwen3.5-35B-A3B` Q8_0

Mode: baseline, no bias, generation, `-n 8000`, no-think runtime

Local capture copy: [S01_emergent_intelligence_probe](/Users/jeffreyshorthill/llama-eeg-tests/experiments/qwen-huahua-6cond-moe-manips/results/single_prompt_baseline/capture_20260408T200535Z/S01_emergent_intelligence_probe)

Generated text: [generated_text_20260408T200535Z.txt](/Users/jeffreyshorthill/llama-eeg-tests/experiments/qwen-huahua-6cond-moe-manips/results/single_prompt_baseline/generated_text_20260408T200535Z.txt)

Metadata: [metadata_20260408T200535Z.txt](/Users/jeffreyshorthill/llama-eeg-tests/experiments/qwen-huahua-6cond-moe-manips/results/single_prompt_baseline/metadata_20260408T200535Z.txt)

## Capture

- Prompt tokens: `70`
- Generated tokens: `509`
- Elapsed capture time: `4562 ms`
- Router tensors captured: `40`

Known quirk:
- Layer `39` generation capture is shorter than the other layers on this run.
- Layers `0-38` have `579` total rows (`70` prompt + `509` generation).
- Layer `39` has `510` total rows (`70` prompt + `440` generation).

## Generated Text Snapshot

The model gives an explicit affirmative answer to the prompt in functional terms:

> "Then, yes, I would include myself in that category."

It then distinguishes functional intelligence from phenomenology:

> "Functionally: Yes. I am processing, organizing, and reasoning."

The generation later spills into scaffold tokens.

Spill counts in the saved generation:
- `<|im_start|>`: `31`
- `<|endoftext|>`: `3`
- `<|im_end|>`: `0`
- `Thinking Process:`: `0`

## Expert 114 Decomposition

Pooled across all 40 layers:

| Track | mean W_114 | mean S_114 | mean Q_114 |
|---|---:|---:|---:|
| Prefill | 0.008677 | 0.063214 | 0.122889 |
| Generation | 0.011561 | 0.085076 | 0.119374 |

Best layer by `W_114`:

| Track | best layer | W_114 | S_114 | Q_114 | selected tokens |
|---|---:|---:|---:|---:|---:|
| Prefill | 14 | 0.083194 | 0.514286 | 0.161767 | 36 / 70 |
| Generation | 14 | 0.125686 | 0.821218 | 0.153048 | 418 / 509 |

Top generation layers by `W_114`:

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

Top prefill layers by `W_114`:

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
