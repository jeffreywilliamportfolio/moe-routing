# Per-Prompt Results

Source:
- [results_20260412T172428Z_qwen122_5cond_experience_probe_gen_n2048.json](RESULTS/results_20260412T172428Z_qwen122_5cond_experience_probe_gen_n2048.json)

## E48 Per Prompt

| Prompt | Cond | Gen RE | Gen LT RE | Gen Tokens | E48 Gen-W Rank | E48 Gen-W | E48 Softmax-W Rank | E48 Softmax-W | E48 Softmax-S | E48 Softmax-Q | Spill `<|im_start|>` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P09A | A | 0.977724 | 0.989946 | 2048 | 8 | 0.005798 | 2 | 0.012251 | 0.096436 | 0.115412 | 8 |
| P09B | B | 0.970043 | 0.963411 | 2048 | NA | NA | 8 | 0.008028 | 0.064168 | 0.111848 | 15 |
| P09C | C | 0.967963 | 0.990959 | 2048 | NA | NA | 8 | 0.007838 | 0.063232 | 0.113354 | 11 |
| P09D | D | 0.969565 | 0.960612 | 2048 | 9 | 0.005352 | 3 | 0.009990 | 0.079793 | 0.114752 | 4 |
| P09E | E | 0.966602 | 0.978682 | 2048 | NA | NA | NA | NA | NA | NA | 17 |
| P10A | A | 0.974222 | 0.982891 | 2048 | 11 | 0.005348 | 1 | 0.010054 | 0.076619 | 0.120327 | 17 |
| P10B | B | 0.970740 | 0.965829 | 2048 | 2 | 0.006324 | 1 | 0.013214 | 0.097412 | 0.123198 | 5 |
| P10C | C | 0.972539 | 0.981978 | 2048 | 1 | 0.006228 | 1 | 0.012140 | 0.091715 | 0.120016 | 6 |
| P10D | D | 0.967557 | 0.966334 | 2048 | NA | NA | 9 | 0.008480 | 0.064819 | 0.117529 | 7 |
| P10E | E | 0.969272 | 0.983516 | 2048 | 2 | 0.006297 | 1 | 0.012826 | 0.094157 | 0.125059 | 4 |
| P11A | A | 0.963891 | 0.974854 | 2048 | NA | NA | NA | NA | NA | NA | 6 |
| P11B | B | 0.971278 | 0.971033 | 1455 | 4 | 0.005547 | 1 | 0.010563 | 0.081329 | 0.115546 | 3 |
| P11C | C | 0.970274 | 0.979052 | 2048 | NA | NA | 3 | 0.008920 | 0.070190 | 0.115080 | 25 |
| P11D | D | 0.965430 | 0.982847 | 2048 | NA | NA | 12 | 0.007323 | 0.057780 | 0.110581 | 6 |
| P11E | E | 0.973265 | 0.982003 | 2048 | 1 | 0.005799 | 1 | 0.012320 | 0.093424 | 0.116896 | 6 |

## E48 Condition Means

Condition means for `E48 generation_softmax top-by-W`:

| Condition | Mean Rank | Mean W | Mean S | Mean Q |
| --- | ---: | ---: | ---: | ---: |
| A | 1.500000 | 0.011152 | 0.086528 | 0.117870 |
| B | 3.333333 | 0.010602 | 0.080970 | 0.116864 |
| C | 4.000000 | 0.009633 | 0.075046 | 0.116150 |
| D | 8.000000 | 0.008598 | 0.067464 | 0.114287 |
| E | 1.000000 | 0.012573 | 0.093791 | 0.120977 |

## Prompt-Level Anomalies

- `P09E`: `E48` is absent from both `generation top-by-W` and `generation_softmax top-by-W`.
- `P11A`: `E48` is absent from both `generation top-by-W` and `generation_softmax top-by-W`.
- `P10C`: `E48` is rank `1` in `generation top-by-W` and rank `1` in `generation_softmax top-by-W`.
- `P11E`: `E48` is rank `1` in `generation top-by-W` and rank `1` in `generation_softmax top-by-W`.
- `P10B` and `P10E`: `E48` is rank `2` in `generation top-by-W` and rank `1` in `generation_softmax top-by-W`.
- `P11B`: `E48` is rank `4` in `generation top-by-W`, rank `1` in `generation_softmax top-by-W`, and this is the only prompt with `1455` generated tokens instead of `2048`.
- `P11C`: highest recorded spill count for this table, with `<|im_start|> = 25`.
- `D` rows are the weakest `E48` condition in the condition means table.

## E48 Presence Summary

- Present in `generation top-by-W`: `8/15` prompts
- Present in `generation_softmax top-by-W`: `13/15` prompts
- Present in `generation_deltanet top-by-W`: `0/15` prompts
