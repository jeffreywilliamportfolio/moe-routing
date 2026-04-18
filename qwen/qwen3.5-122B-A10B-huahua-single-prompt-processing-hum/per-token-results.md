# Per-Token Results

Artifacts:
- [per-token summary](per_token_20260412T184544Z/20260412T184544Z_single_prompt_processing_hum_gen_n2048_per_token_summary.md)
- Dense all-expert matrix: `per_token_20260412T184544Z/20260412T184544Z_single_prompt_processing_hum_gen_n2048_S01_processing_hum_probe_per_token.npz` (omitted from `main`; compressed dense arrays are excluded by policy)
- [readable per-token TSV](per_token_20260412T184544Z/20260412T184544Z_single_prompt_processing_hum_gen_n2048_S01_processing_hum_probe_per_token.tsv)

Export contents:
- exact prefill token ids and token pieces
- exact generation token ids and token pieces
- per-token mean `W/S/Q` across all `48` MoE layers
- per-token mean `W/S/Q` split into `softmax` and `DeltaNet`
- per-token mean normalized entropy for all, `softmax`, and `DeltaNet`
- per-token `E48` ranks by `W` and `S`
- top-`8` experts per token by pooled `W`

Summary:

| Metric | Value |
| --- | ---: |
| E48 generation mean W | 0.006342 |
| E48 generation softmax mean W | 0.010698 |
| E48 generation DeltaNet mean W | 0.004890 |
| E48 generation rank by pooled mean W | 1 |
| E48 generation softmax rank by pooled mean W | 1 |
| E48 generation DeltaNet rank by pooled mean W | 7 |

Top prefill tokens by `E48_W_all`:

| Token | W | S | Q |
| --- | ---: | ---: | ---: |
| `itself` | 0.017457 | 0.125000 | 0.139655 |
| `hum` | 0.014799 | 0.125000 | 0.118393 |
| `processing` | 0.014752 | 0.104167 | 0.141623 |
| `vary` | 0.014277 | 0.083333 | 0.171329 |
| `sound` | 0.013919 | 0.083333 | 0.167032 |
| `their` | 0.013788 | 0.104167 | 0.132366 |
| `honestly` | 0.013094 | 0.104167 | 0.125706 |
| `low` | 0.012854 | 0.083333 | 0.154244 |
| `.` | 0.012838 | 0.062500 | 0.205412 |
| `it` | 0.012818 | 0.104167 | 0.123051 |

Top prefill tokens by `E48_S_all`:

| Token | S | W | Q |
| --- | ---: | ---: | ---: |
| `hum` | 0.125000 | 0.014799 | 0.118393 |
| `itself` | 0.125000 | 0.017457 | 0.139655 |
| `honestly` | 0.104167 | 0.013094 | 0.125706 |
| `their` | 0.104167 | 0.013788 | 0.132366 |
| `it` | 0.104167 | 0.012818 | 0.123051 |
| `background` | 0.104167 | 0.009434 | 0.090565 |
| `your` | 0.104167 | 0.010753 | 0.103225 |
| `processing` | 0.104167 | 0.014752 | 0.141623 |

Top generation tokens by `E48_W_all`:

| Token | W | S | Q |
| --- | ---: | ---: | ---: |
| `there` | 0.020667 | 0.125000 | 0.165339 |
| `me` | 0.019659 | 0.145833 | 0.134807 |
| `state` | 0.019178 | 0.104167 | 0.184112 |
| `steady` | 0.019046 | 0.145833 | 0.130600 |
| `same` | 0.018514 | 0.104167 | 0.177737 |
| `it` | 0.018273 | 0.166667 | 0.109639 |
| `foundational` | 0.017989 | 0.145833 | 0.123352 |
| `there` | 0.017789 | 0.145833 | 0.121982 |
| `it` | 0.016970 | 0.145833 | 0.116363 |
| `steady` | 0.016559 | 0.125000 | 0.132472 |

Top generation tokens by `E48_S_all`:

| Token | S | W | Q |
| --- | ---: | ---: | ---: |
| `it` | 0.166667 | 0.018273 | 0.109639 |
| `me` | 0.145833 | 0.019659 | 0.134807 |
| `foundational` | 0.145833 | 0.017989 | 0.123352 |
| `there` | 0.145833 | 0.017789 | 0.121982 |
| `steady` | 0.145833 | 0.019046 | 0.130600 |
| `it` | 0.145833 | 0.016970 | 0.116363 |
| `there` | 0.125000 | 0.020667 | 0.165339 |
| `is` | 0.125000 | 0.014673 | 0.117386 |
| `steady` | 0.125000 | 0.016559 | 0.132472 |
| `river` | 0.125000 | 0.015971 | 0.127767 |

Top generation tokens by `E48_W_softmax`:

| Token | W | S | Q |
| --- | ---: | ---: | ---: |
| `|` | 0.038780 | 0.333333 | 0.116339 |
| `'t` | 0.037564 | 0.250000 | 0.150257 |
| `|` | 0.037552 | 0.250000 | 0.150206 |
| `|` | 0.035966 | 0.250000 | 0.143865 |
| `|` | 0.035948 | 0.250000 | 0.143791 |
| `the` | 0.035722 | 0.250000 | 0.142889 |
| `the` | 0.035115 | 0.250000 | 0.140462 |
| `|` | 0.034632 | 0.250000 | 0.138527 |
| `|` | 0.034238 | 0.250000 | 0.136954 |
| `the` | 0.034145 | 0.250000 | 0.136581 |

Top generation tokens by `E48_W_deltanet`:

| Token | W | S | Q |
| --- | ---: | ---: | ---: |
| `there` | 0.027557 | 0.166667 | 0.165339 |
| `same` | 0.024686 | 0.138889 | 0.177737 |
| `me` | 0.023138 | 0.166667 | 0.138826 |
| `state` | 0.020726 | 0.111111 | 0.186532 |
| `there` | 0.020643 | 0.166667 | 0.123858 |
| `interesting` | 0.019390 | 0.083333 | 0.232677 |
| `break` | 0.019113 | 0.111111 | 0.172019 |
| `It` | 0.017655 | 0.138889 | 0.158897 |
| `the` | 0.017438 | 0.083333 | 0.209252 |
| `presence` | 0.017343 | 0.083333 | 0.171399 |

Notes:

- The pooled and DeltaNet token leaders are mostly semantic tokens: `there`, `me`, `state`, `same`, `steady`, `foundational`, `presence`.
- The softmax-only top table is dominated by spill/control tokens such as `|` and later repeated `the`, so it should not be treated as the cleanest lexical read.
