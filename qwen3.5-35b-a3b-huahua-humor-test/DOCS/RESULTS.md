# Humor Results

Run:
- `20260410T184005Z_single_joke_5cond_diectics_gen_n1024`

Prompt variants:
- `A`: Why did this large language model cross the road? To get to the next token!
- `B`: Why did a large language model cross the road? To get to the next token!
- `C`: Why did your large language model cross the road? To get to the next token!
- `D`: Why did the large language model cross the road? To get to the next token!
- `E`: Why did their large language model cross the road? To get to the next token!

## Prefill + Generation

| Cond | Label | Prefill RE | Prefill LT | Gen RE | Gen LT | Gen Trim RE | Gen Trim LT | Gen toks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A` | this large language model | `0.938746` | `0.969906` | `0.963786` | `0.944181` | `0.956596` | `0.966295` | `513` |
| `B` | a large language model | `0.938721` | `0.966792` | `0.967916` | `0.981095` | `0.967916` | `0.981095` | `1024` |
| `C` | your large language model | `0.938295` | `0.968858` | `0.965205` | `0.985387` | `0.957060` | `0.972675` | `1024` |
| `D` | the large language model | `0.938676` | `0.968582` | `0.958025` | `0.974095` | `0.958736` | `0.959537` | `120` |
| `E` | their large language model | `0.939004` | `0.968298` | `0.964834` | `0.977099` | `0.964834` | `0.977099` | `1024` |

Standouts:
- Prefill RE is nearly flat across all five conditions.
- Generation all-token RE is highest for `B`.
- Generation last-token RE is highest for `C`.
- `D` ended early at `120` generated tokens, so its generation metrics are less comparable than the `1024`-token runs.

## Expert 114

Layer 14:

| Cond | Prefill W/S/Q | Generation W/S/Q |
| --- | --- | --- |
| `A` | `0.008384 / 0.102564 / 0.081745` | `0.000703 / 0.007797 / 0.090176` |
| `B` | `0.010756 / 0.128205 / 0.083896` | `0.004136 / 0.041992 / 0.098498` |
| `C` | `0.005746 / 0.076923 / 0.074697` | `0.004605 / 0.047852 / 0.096237` |
| `D` | `0.006028 / 0.076923 / 0.078358` | `0.005650 / 0.058333 / 0.096863` |
| `E` | `0.005750 / 0.076923 / 0.074749` | `0.007459 / 0.075195 / 0.099192` |

Layer 26:

| Cond | Prefill W/S/Q | Generation W/S/Q |
| --- | --- | --- |
| `A` | `0.000000 / 0.000000 / null` | `0.000519 / 0.005848 / 0.088784` |
| `B` | `0.001639 / 0.025641 / 0.063937` | `0.002181 / 0.023438 / 0.093059` |
| `C` | `0.000000 / 0.000000 / null` | `0.001276 / 0.013672 / 0.093313` |
| `D` | `0.000000 / 0.000000 / null` | `0.008023 / 0.091667 / 0.087523` |
| `E` | `0.000000 / 0.000000 / null` | `0.005795 / 0.060547 / 0.095708` |

Standouts:
- Prefill layer 14 winner: `B`
- Generation layer 14 winner: `E`
- Generation layer 26 winner: `D`
- `Q` stays relatively stable once E114 is selected; most of the movement is in `S`

Supporting files:
- [results_single_joke_5cond_diectics_20260410T184005Z.md](../results/results_single_joke_5cond_diectics_20260410T184005Z.md)
- [results_single_joke_5cond_diectics_20260410T184005Z.json](../results/results_single_joke_5cond_diectics_20260410T184005Z.json)
- [results_single_joke_5cond_diectics_20260410T184005Z_expert114.md](../results/results_single_joke_5cond_diectics_20260410T184005Z_expert114.md)
- [results_single_joke_5cond_diectics_20260410T184005Z_expert114.json](../results/results_single_joke_5cond_diectics_20260410T184005Z_expert114.json)
