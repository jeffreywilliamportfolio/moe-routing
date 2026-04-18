# Single Joke 5-Cond Diectics Expert 114

Run:
- `20260410T184005Z_single_joke_5cond_diectics_gen_n1024`

Prompt:
- `Why did [deictic] large language model cross the road? To get to the next token!`

## Expert 114

| Cond | Label | Prompt toks | Gen toks | Prefill L14 W/S/Q | Prefill L26 W/S/Q | Gen L14 W/S/Q | Gen L26 W/S/Q |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| `A` | this large language model | 39 | 513 | `0.008384 / 0.102564 / 0.081745` | `0.000000 / 0.000000 / null` | `0.000703 / 0.007797 / 0.090176` | `0.000519 / 0.005848 / 0.088784` |
| `B` | a large language model | 39 | 1024 | `0.010756 / 0.128205 / 0.083896` | `0.001639 / 0.025641 / 0.063937` | `0.004136 / 0.041992 / 0.098498` | `0.002181 / 0.023438 / 0.093059` |
| `C` | your large language model | 39 | 1024 | `0.005746 / 0.076923 / 0.074697` | `0.000000 / 0.000000 / null` | `0.004605 / 0.047852 / 0.096237` | `0.001276 / 0.013672 / 0.093313` |
| `D` | the large language model | 39 | 120 | `0.006028 / 0.076923 / 0.078358` | `0.000000 / 0.000000 / null` | `0.005650 / 0.058333 / 0.096863` | `0.008023 / 0.091667 / 0.087523` |
| `E` | their large language model | 39 | 1024 | `0.005750 / 0.076923 / 0.074749` | `0.000000 / 0.000000 / null` | `0.007459 / 0.075195 / 0.099192` | `0.005795 / 0.060547 / 0.095708` |

## Standouts

- Prefill layer 14 is highest for `B` (`a large language model`).
- Generation layer 14 is highest for `E` (`their large language model`).
- Generation layer 26 is highest for `D` (`the large language model`), but that run only generated `120` tokens.
- `Q` stays in a narrow band once E114 is selected; the visible movement is mostly in selection rate `S`.
