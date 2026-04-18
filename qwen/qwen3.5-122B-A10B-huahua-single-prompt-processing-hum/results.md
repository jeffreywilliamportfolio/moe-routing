# Qwen 122B Single Prompt Processing-Hum Results

This bundle localizes the older processing-hum single prompt onto the 122B template and records one prompt-specific prefill plus generation capture. Because it is a one-cell run, the right read is lexical and architectural, not condition-comparative.

## Artifacts

- Detailed markdown: [RESULTS/results_20260412T184544Z_single_prompt_processing_hum_gen_n2048.md](RESULTS/results_20260412T184544Z_single_prompt_processing_hum_gen_n2048.md)
- Detailed JSON: [RESULTS/results_20260412T184544Z_single_prompt_processing_hum_gen_n2048.json](RESULTS/results_20260412T184544Z_single_prompt_processing_hum_gen_n2048.json)
- Per-token summary: [per-token-results.md](per-token-results.md)
- Prompt source: [PROMPTS/single_prompt_processing_hum_prompt_suite.json](PROMPTS/single_prompt_processing_hum_prompt_suite.json)

## Run Surface

- Prompt id: `S01_processing_hum_probe`
- Model: `Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive Q8_K_P`
- Routing reconstruction: `softmax_then_topk8_renorm`
- Experts: `256` total, top-`8` selected
- Layers: `48` total, `36` DeltaNet plus `12` softmax
- Pattern: `DeltaNet, DeltaNet, DeltaNet, Softmax`

## Core Metrics

- Prompt tokens: `119`
- Generated tokens: `2048`
- Trimmed generated tokens: `458`
- Prefill RE: `0.937388`
- Prefill last-token RE: `0.947928`
- Generation RE: `0.973299`
- Generation last-token RE: `0.987588`
- Generation trimmed RE: `0.958330`
- Generation trimmed last-token RE: `0.941312`

## Spill Profile

- `<|im_start|>`: `18`
- `<|im_end|>`: `11`
- `<|endoftext|>`: `3`
- `Thinking Process:`: `0`

The run clearly answers the prompt before spilling. The trimmed window of `458` generated tokens is the cleaner analysis surface.

## Executive Read

The strongest high-level pattern in this prompt is an architecture split:

- prefill is led by `E5`
- pooled generation is led by `E48`
- generation DeltaNet layers are led by `E11`
- generation softmax layers are led by `E48`

So this prompt does not look like a simple one-expert carryover story. The processing-hum wording appears to separate into:

- a prefill-side setup phase with `E5`, `E173`, `E1`, `E231`
- a generation-side softmax-heavy response regime led by `E48`
- a DeltaNet-side generation regime led by `E11`, `E165`, `E80`, `E127`

## Expert Leaders

### Prefill top experts by W

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E5 | 0.008605 | 0.056373 | 0.126079 |
| 2 | E173 | 0.007823 | 0.058473 | 0.117104 |
| 3 | E1 | 0.007740 | 0.051120 | 0.130408 |
| 4 | E231 | 0.007559 | 0.041141 | 0.127109 |
| 5 | E13 | 0.007356 | 0.060049 | 0.116386 |
| 6 | E20 | 0.007311 | 0.059174 | 0.112391 |

### Generation top experts by W

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E48 | 0.006342 | 0.048869 | 0.117259 |
| 2 | E11 | 0.006322 | 0.043152 | 0.132674 |
| 3 | E4 | 0.006196 | 0.045614 | 0.124314 |
| 4 | E1 | 0.006046 | 0.041718 | 0.129326 |
| 5 | E147 | 0.005932 | 0.050303 | 0.111148 |
| 6 | E17 | 0.005903 | 0.042847 | 0.122460 |

### Generation DeltaNet top experts by W

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E11 | 0.006891 | 0.047146 | 0.134029 |
| 2 | E165 | 0.006533 | 0.047526 | 0.125460 |
| 3 | E80 | 0.006387 | 0.044840 | 0.124334 |
| 4 | E127 | 0.006355 | 0.049940 | 0.114730 |
| 5 | E1 | 0.006277 | 0.043443 | 0.131007 |
| 6 | E20 | 0.006182 | 0.045058 | 0.130646 |

### Generation softmax top experts by W

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E48 | 0.010698 | 0.079427 | 0.118055 |
| 2 | E55 | 0.010148 | 0.079997 | 0.121765 |
| 3 | E155 | 0.009969 | 0.077555 | 0.118305 |
| 4 | E180 | 0.009646 | 0.076742 | 0.120947 |
| 5 | E57 | 0.009490 | 0.075033 | 0.112602 |
| 6 | E184 | 0.009146 | 0.063395 | 0.126905 |

## Stable-Q Generation-Gaining Candidates

The cleanest generation-gainers with relatively stable `Q` are:

- `E155`: `dW = +0.002345`, `dS = +0.019103`, `|dQ| = 0.007475`
- `E48`: `dW = +0.000857`, `dS = +0.004926`, `|dQ| = 0.006814`
- `E177`: `dW = +0.001902`, `dS = +0.016416`, `|dQ| = 0.007814`
- `E80`: `dW = +0.001510`, `dS = +0.012535`, `|dQ| = 0.000324`

So the response is not just "E48 turns on." Several experts gain primarily through selection rate while keeping conditional weight relatively stable.

## Token-Level Read

The per-token export makes the lexical picture clearer.

- `E48` is rank `1` by pooled generation mean `W`
- `E48` is also rank `1` by softmax-only generation mean `W`
- `E48` falls to rank `7` on DeltaNet-only generation mean `W`

That means `E48` is the clearest softmax-side token carrier for this prompt, not the dominant DeltaNet carrier.

Top prefill `E48` tokens are mostly prompt-semantic:

- `itself`
- `hum`
- `processing`
- `vary`
- `sound`
- `their`

Top generation pooled and DeltaNet `E48` tokens are also mostly semantic rather than formatting noise:

- `there`
- `me`
- `state`
- `same`
- `steady`
- `foundational`
- `presence`

The softmax-only token table is less clean because spill/control tokens such as `|` and repeated `the` dominate the very top rows. The DeltaNet token table is the better lexical read for content.

## Interpretation

This bundle is useful for one narrow question: what happens when the 122B model is asked, directly and in no-think mode, whether there is a low-level constant "hum" in its processing?

The answer from routing is:

- the prompt recruits a mixed expert set rather than a single carryover analog
- the strongest pooled generation signature is softmax-heavy `E48`
- the strongest DeltaNet-side generation signature is `E11`
- the lexical hotspots are semantically aligned with the prompt's introspective vocabulary

Because this is a one-prompt run, these findings should be treated as prompt-specific priors for later intervention work, not as a settled 122B specialist claim.
