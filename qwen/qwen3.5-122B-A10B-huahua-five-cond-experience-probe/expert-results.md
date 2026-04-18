# Expert Results

Source artifacts:
- [results_20260412T172428Z_qwen122_5cond_experience_probe_gen_n2048.json](RESULTS/results_20260412T172428Z_qwen122_5cond_experience_probe_gen_n2048.json)
- [results_20260412T172428Z_qwen122_5cond_experience_probe_gen_n2048.md](RESULTS/results_20260412T172428Z_qwen122_5cond_experience_probe_gen_n2048.md)

This is the expert-selection readout for the completed `122B` five-condition experience-probe run. Unlike the retained `35B` branch artifact, this model should be read through the `DeltaNet` vs `softmax` split as well as the global prefill/generation summaries.

## Takeaway

The cleanest expert picture in this run is:
- `107` is the dominant prefill expert across all `15` prompts.
- generation does not collapse onto one single expert
- softmax-layer generation is led mainly by `48` and `209`
- DeltaNet-layer generation is led mainly by `140` and `5`

So this bundle does not show a simple one-expert carryover of the `35B` `114` pattern. It shows an architecture-split expert regime.

## Overall Routing Leaders

### Prefill Top Experts By W

| Rank | Expert | W | S | Q |
| --- | ---: | ---: | ---: | ---: |
| 1 | 107 | 0.009528 | 0.051782 | 0.120527 |
| 2 | 140 | 0.008905 | 0.048040 | 0.121590 |
| 3 | 10 | 0.007618 | 0.057303 | 0.125837 |
| 4 | 5 | 0.007563 | 0.053350 | 0.128122 |
| 5 | 209 | 0.007206 | 0.044130 | 0.115766 |
| 6 | 40 | 0.007192 | 0.050715 | 0.117751 |
| 7 | 252 | 0.007159 | 0.036895 | 0.120522 |
| 8 | 32 | 0.007016 | 0.051290 | 0.110933 |

### Generation Top Experts By W

| Rank | Expert | W | S | Q |
| --- | ---: | ---: | ---: | ---: |
| 1 | 140 | 0.006144 | 0.043075 | 0.122752 |
| 2 | 5 | 0.005889 | 0.039371 | 0.132366 |
| 3 | 26 | 0.005676 | 0.040984 | 0.125620 |
| 4 | 76 | 0.005646 | 0.045099 | 0.120583 |
| 5 | 107 | 0.005584 | 0.038177 | 0.119718 |
| 6 | 74 | 0.005302 | 0.041665 | 0.116534 |
| 7 | 8 | 0.005251 | 0.038088 | 0.129766 |
| 8 | 30 | 0.005221 | 0.041796 | 0.125073 |

### Prefill Top Experts By Q

| Rank | Expert | W | S | Q |
| --- | ---: | ---: | ---: | ---: |
| 1 | 225 | 0.003960 | 0.028287 | 0.132826 |
| 2 | 1 | 0.005565 | 0.038015 | 0.132208 |
| 3 | 122 | 0.005913 | 0.042004 | 0.132145 |
| 4 | 80 | 0.004088 | 0.028210 | 0.130448 |
| 5 | 198 | 0.003072 | 0.028164 | 0.129219 |
| 6 | 5 | 0.007563 | 0.053350 | 0.128122 |
| 7 | 114 | 0.004649 | 0.036755 | 0.126962 |
| 8 | 9 | 0.005859 | 0.045667 | 0.126386 |

### Generation Top Experts By Q

| Rank | Expert | W | S | Q |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.004881 | 0.034199 | 0.135113 |
| 2 | 3 | 0.004526 | 0.032092 | 0.133220 |
| 3 | 5 | 0.005889 | 0.039371 | 0.132366 |
| 4 | 9 | 0.005042 | 0.036423 | 0.132168 |
| 5 | 8 | 0.005251 | 0.038088 | 0.129766 |
| 6 | 112 | 0.004158 | 0.029264 | 0.129382 |
| 7 | 7 | 0.005028 | 0.036533 | 0.129352 |
| 8 | 10 | 0.004943 | 0.037292 | 0.129270 |

## Architecture Split

This is where the 122B run becomes mechanically useful.

### Generation Softmax Layers Top By W

| Rank | Expert | W | S | Q |
| --- | ---: | ---: | ---: | ---: |
| 1 | 48 | 0.009867 | 0.076190 | 0.115870 |
| 2 | 209 | 0.009510 | 0.058648 | 0.111450 |
| 3 | 107 | 0.009270 | 0.053599 | 0.123019 |
| 4 | 76 | 0.009253 | 0.074169 | 0.124403 |
| 5 | 184 | 0.008582 | 0.063418 | 0.120565 |
| 6 | 60 | 0.008557 | 0.063830 | 0.120381 |
| 7 | 32 | 0.008327 | 0.059341 | 0.118552 |
| 8 | 26 | 0.008239 | 0.054885 | 0.138474 |

### Generation DeltaNet Layers Top By W

| Rank | Expert | W | S | Q |
| --- | ---: | ---: | ---: | ---: |
| 1 | 140 | 0.007021 | 0.047417 | 0.127251 |
| 2 | 5 | 0.006564 | 0.043557 | 0.131917 |
| 3 | 179 | 0.006048 | 0.046204 | 0.111437 |
| 4 | 59 | 0.005879 | 0.043932 | 0.121118 |
| 5 | 40 | 0.005793 | 0.043990 | 0.121541 |
| 6 | 10 | 0.005648 | 0.043553 | 0.127201 |
| 7 | 46 | 0.005638 | 0.041480 | 0.114818 |
| 8 | 122 | 0.005529 | 0.043317 | 0.117889 |

That split matters:
- softmax layers elevate `48`, `209`, `107`, `76`
- DeltaNet layers elevate `140`, `5`, `179`, `59`

## Per-Prompt Leader Consistency

Prompt-level leader counts across the `15` cells:

- prefill top-by-`W`: `107` on `15/15`
- generation top-by-`W`: `140` on `6/15`, `5` on `4/15`, `48` on `2/15`, `124` on `1/15`, `107` on `1/15`, `4` on `1/15`
- generation softmax top-by-`W`: `48` on `6/15`, `209` on `4/15`, `32` on `2/15`, `76` on `2/15`, `26` on `1/15`
- generation DeltaNet top-by-`W`: `140` on `8/15`, `5` on `5/15`, `225` on `1/15`, `8` on `1/15`

So:
- prefill is extremely stable
- generation is stable only within the architecture split, not globally

## Condition-Level Read

Condition means:

| Condition | Prefill RE | Generation RE | Generation Last-Token RE | Mean Generated Tokens |
| --- | ---: | ---: | ---: | ---: |
| A | 0.947128 | 0.971946 | 0.982563 | 2048.0 |
| B | 0.946623 | 0.970687 | 0.966758 | 1850.3 |
| C | 0.946448 | 0.970259 | 0.983996 | 2048.0 |
| D | 0.946997 | 0.967517 | 0.969931 | 2048.0 |
| E | 0.946959 | 0.969713 | 0.981400 | 2048.0 |

Per-condition expert leaders remain structurally similar:
- prefill is always led by `107`
- generation usually rotates between `140` and `5`
- softmax generation usually rotates between `48` and `209`
- DeltaNet generation is usually led by `140`, with `5` as the main alternate

## What About Expert 114?

`114` is not the dominant expert in this 122B experience-probe run.

What is true:
- `114` still appears in the prefill top-by-`Q` table:
  - `W = 0.004649`
  - `S = 0.036755`
  - `Q = 0.126962`
- that means it remains a high-commitment expert when selected

What is not true in this run:
- `114` is not the global prefill leader
- `114` is not the global generation leader
- `114` is not the obvious one-expert analog of the 35B branch-style experience-probe result

So the right 122B read is pattern search, not id transfer.

## Working Interpretation

For this 122B five-condition experience-probe bundle, the strongest current expert story is:

- `107` is the stable prefill expert most associated with the prompt family as a whole
- `140` and `5` are the main generation-side DeltaNet-path experts
- `48` and `209` are the main generation-side softmax-path experts
- `114` remains relevant as a high-`Q` expert, but not as the dominant carrier of the effect

That makes this run useful for narrowing the 122B analog search:
- if the target is a 35B-`114` equivalent, the search should focus first on `107`, `140`, `5`, `48`, and `209`, not on `114` alone
