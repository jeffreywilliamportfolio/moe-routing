# Qwen 122B 5-Condition Suite Results

- Model: `Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive Q8_K_P`
- Routing reconstruction: `softmax_then_topk8_renorm`
- Experts: `256` total, top-`8` selected
- Layers: `48` total, `36` DeltaNet + `12` Softmax
- Layer pattern: `DeltaNet, DeltaNet, DeltaNet, Softmax`
- Prompts: `150`

## Condition Means

| Cond | Label | N | Prefill RE | Prefill LT | Gen RE | Gen LT | Gen Trim RE | Gen Trim LT | Mean gen toks | Mean spill <|im_start|> |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A` | this | 30 | 0.947785 | 0.933557 | 0.969030 | 0.972297 | 0.968437 | 0.967953 | 2048.0 | 10.90 |
| `B` | a | 30 | 0.947482 | 0.933464 | 0.969988 | 0.975757 | 0.969424 | 0.965463 | 2048.0 | 11.57 |
| `C` | your | 30 | 0.946953 | 0.934039 | 0.969713 | 0.970849 | 0.968784 | 0.963609 | 2048.0 | 13.80 |
| `D` | the | 30 | 0.947948 | 0.933641 | 0.970437 | 0.979142 | 0.969465 | 0.969884 | 2048.0 | 11.53 |
| `E` | their | 30 | 0.947619 | 0.933413 | 0.968443 | 0.972908 | 0.967704 | 0.967943 | 2048.0 | 9.53 |

## Overall Routing And Expert Selection

### Prefill Top Experts By W

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E107 | 0.010533 | 0.054328 | 0.120936 |
| 2 | E140 | 0.008895 | 0.047109 | 0.121667 |
| 3 | E209 | 0.007626 | 0.044899 | 0.116546 |
| 4 | E10 | 0.007435 | 0.056875 | 0.126349 |
| 5 | E252 | 0.007427 | 0.035943 | 0.118821 |
| 6 | E32 | 0.007239 | 0.053823 | 0.108574 |
| 7 | E40 | 0.006813 | 0.046764 | 0.117900 |
| 8 | E5 | 0.006761 | 0.050273 | 0.127695 |
| 9 | E134 | 0.006751 | 0.048287 | 0.108883 |
| 10 | E93 | 0.006700 | 0.050131 | 0.119274 |
| 11 | E179 | 0.006694 | 0.044747 | 0.108572 |
| 12 | E46 | 0.006579 | 0.036781 | 0.109403 |

### Generation Top Experts By W

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E140 | 0.006117 | 0.042844 | 0.122834 |
| 2 | E76 | 0.005818 | 0.046827 | 0.119482 |
| 3 | E107 | 0.005809 | 0.039187 | 0.118154 |
| 4 | E8 | 0.005800 | 0.041678 | 0.129242 |
| 5 | E4 | 0.005756 | 0.041114 | 0.130077 |
| 6 | E5 | 0.005560 | 0.039048 | 0.131165 |
| 7 | E0 | 0.005509 | 0.039921 | 0.129730 |
| 8 | E179 | 0.005412 | 0.041774 | 0.109744 |
| 9 | E236 | 0.005328 | 0.045693 | 0.115248 |
| 10 | E7 | 0.005311 | 0.038864 | 0.129300 |
| 11 | E26 | 0.005279 | 0.038655 | 0.123612 |
| 12 | E32 | 0.005250 | 0.039868 | 0.119649 |

### Generation Top Experts By S

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E76 | 0.005818 | 0.046827 | 0.119482 |
| 2 | E236 | 0.005328 | 0.045693 | 0.115248 |
| 3 | E140 | 0.006117 | 0.042844 | 0.122834 |
| 4 | E189 | 0.005000 | 0.042141 | 0.113368 |
| 5 | E179 | 0.005412 | 0.041774 | 0.109744 |
| 6 | E8 | 0.005800 | 0.041678 | 0.129242 |
| 7 | E4 | 0.005756 | 0.041114 | 0.130077 |
| 8 | E59 | 0.005146 | 0.040399 | 0.117933 |
| 9 | E0 | 0.005509 | 0.039921 | 0.129730 |
| 10 | E32 | 0.005250 | 0.039868 | 0.119649 |
| 11 | E30 | 0.004974 | 0.039829 | 0.123228 |
| 12 | E107 | 0.005809 | 0.039187 | 0.118154 |

### Generation Top Experts By Q

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E1 | 0.004999 | 0.035674 | 0.135361 |
| 2 | E3 | 0.004883 | 0.034555 | 0.134603 |
| 3 | E9 | 0.005063 | 0.037179 | 0.131833 |
| 4 | E5 | 0.005560 | 0.039048 | 0.131165 |
| 5 | E112 | 0.004525 | 0.031618 | 0.130801 |
| 6 | E4 | 0.005756 | 0.041114 | 0.130077 |
| 7 | E0 | 0.005509 | 0.039921 | 0.129730 |
| 8 | E10 | 0.004997 | 0.037868 | 0.129678 |
| 9 | E7 | 0.005311 | 0.038864 | 0.129300 |
| 10 | E8 | 0.005800 | 0.041678 | 0.129242 |
| 11 | E6 | 0.004051 | 0.030400 | 0.128950 |
| 12 | E11 | 0.004736 | 0.034036 | 0.125796 |

### Generation DeltaNet Top Experts By W

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E140 | 0.006927 | 0.046699 | 0.126736 |
| 2 | E179 | 0.006337 | 0.048432 | 0.110038 |
| 3 | E4 | 0.006078 | 0.043313 | 0.128262 |
| 4 | E5 | 0.005994 | 0.042217 | 0.130946 |
| 5 | E59 | 0.005845 | 0.044845 | 0.119969 |
| 6 | E46 | 0.005734 | 0.041377 | 0.116629 |
| 7 | E8 | 0.005715 | 0.040943 | 0.130800 |
| 8 | E10 | 0.005611 | 0.043595 | 0.127298 |
| 9 | E47 | 0.005550 | 0.040992 | 0.114633 |
| 10 | E0 | 0.005495 | 0.039672 | 0.128363 |
| 11 | E236 | 0.005421 | 0.047643 | 0.113667 |
| 12 | E122 | 0.005405 | 0.042432 | 0.116097 |

### Generation Softmax Top Experts By W

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E209 | 0.010017 | 0.058974 | 0.110725 |
| 2 | E107 | 0.009576 | 0.055455 | 0.121767 |
| 3 | E76 | 0.009137 | 0.075401 | 0.121776 |
| 4 | E48 | 0.008673 | 0.067194 | 0.116692 |
| 5 | E32 | 0.008506 | 0.063031 | 0.119731 |
| 6 | E26 | 0.008222 | 0.055245 | 0.137154 |
| 7 | E92 | 0.007500 | 0.059341 | 0.117231 |
| 8 | E60 | 0.007332 | 0.055326 | 0.117048 |
| 9 | E114 | 0.007106 | 0.055593 | 0.113933 |
| 10 | E184 | 0.007044 | 0.053649 | 0.120064 |
| 11 | E252 | 0.007033 | 0.039673 | 0.127866 |
| 12 | E204 | 0.006788 | 0.050571 | 0.124497 |

## Stable-Q Generation-Gaining Candidates

| Expert | Score | dW | dS | Prefill Q | Gen Q | |dQ| | Gen W rank | Gen S rank | Gen Q rank |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E8 | 0.006942 | +0.002039 | +0.010015 | 0.113304 | 0.129242 | 0.015938 | 4 | 6 | 10 |
| E0 | 0.006203 | +0.001241 | +0.007892 | 0.123589 | 0.129730 | 0.006141 | 7 | 9 | 7 |
| E189 | 0.005876 | +0.001778 | +0.012412 | 0.098664 | 0.113368 | 0.014704 | 17 | 4 | 191 |
| E25 | 0.005627 | +0.001396 | +0.004124 | 0.104834 | 0.124839 | 0.020005 | 16 | 35 | 16 |
| E26 | 0.005619 | +0.000614 | +0.003343 | 0.124035 | 0.123612 | 0.000424 | 11 | 16 | 19 |
| E236 | 0.005559 | +0.000546 | +0.001173 | 0.103451 | 0.115248 | 0.011798 | 9 | 2 | 140 |
| E7 | 0.005542 | +0.000465 | +0.005292 | 0.121097 | 0.129300 | 0.008203 | 10 | 14 | 9 |
| E214 | 0.005364 | +0.001690 | +0.012705 | 0.111680 | 0.116340 | 0.004660 | 40 | 33 | 112 |
| E3 | 0.005281 | +0.000979 | +0.003955 | 0.115420 | 0.134603 | 0.019183 | 23 | 66 | 2 |
| E59 | 0.005236 | +0.000207 | +0.005379 | 0.109414 | 0.117933 | 0.008519 | 13 | 8 | 76 |
| E180 | 0.005200 | +0.001426 | +0.009831 | 0.111510 | 0.116015 | 0.004505 | 43 | 27 | 120 |
| E74 | 0.005173 | +0.000668 | +0.006201 | 0.111815 | 0.115332 | 0.003517 | 25 | 15 | 137 |

## Condition-Level Routing Leaders

### `A` this

### Generation Top Experts By W

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E140 | 0.005971 | 0.042027 | 0.122247 |
| 2 | E8 | 0.005893 | 0.042410 | 0.128941 |
| 3 | E76 | 0.005878 | 0.047392 | 0.119082 |
| 4 | E5 | 0.005835 | 0.041366 | 0.130654 |
| 5 | E107 | 0.005780 | 0.039404 | 0.118048 |
| 6 | E4 | 0.005684 | 0.040833 | 0.129628 |
| 7 | E0 | 0.005666 | 0.041193 | 0.130306 |
| 8 | E236 | 0.005477 | 0.046839 | 0.115415 |
| 9 | E7 | 0.005432 | 0.039745 | 0.129646 |
| 10 | E26 | 0.005423 | 0.039776 | 0.123663 |
| 11 | E179 | 0.005326 | 0.041292 | 0.109602 |
| 12 | E32 | 0.005189 | 0.039717 | 0.119498 |

### `B` a

### Generation Top Experts By W

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E140 | 0.006425 | 0.044534 | 0.123387 |
| 2 | E76 | 0.005863 | 0.047184 | 0.119987 |
| 3 | E107 | 0.005847 | 0.038781 | 0.118059 |
| 4 | E4 | 0.005815 | 0.041100 | 0.131219 |
| 5 | E8 | 0.005730 | 0.041023 | 0.129390 |
| 6 | E179 | 0.005562 | 0.042583 | 0.109773 |
| 7 | E32 | 0.005501 | 0.041380 | 0.120369 |
| 8 | E0 | 0.005414 | 0.038982 | 0.129657 |
| 9 | E59 | 0.005367 | 0.041746 | 0.118473 |
| 10 | E26 | 0.005291 | 0.038527 | 0.123947 |
| 11 | E46 | 0.005252 | 0.038624 | 0.118401 |
| 12 | E134 | 0.005168 | 0.040627 | 0.112951 |

### `C` your

### Generation Top Experts By W

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E5 | 0.006145 | 0.043407 | 0.131090 |
| 2 | E107 | 0.005846 | 0.040219 | 0.117980 |
| 3 | E8 | 0.005665 | 0.041056 | 0.128145 |
| 4 | E4 | 0.005658 | 0.041055 | 0.129028 |
| 5 | E236 | 0.005651 | 0.048101 | 0.114622 |
| 6 | E140 | 0.005639 | 0.039794 | 0.122607 |
| 7 | E76 | 0.005619 | 0.045392 | 0.119174 |
| 8 | E7 | 0.005439 | 0.039473 | 0.129462 |
| 9 | E0 | 0.005397 | 0.039303 | 0.129557 |
| 10 | E179 | 0.005367 | 0.042013 | 0.110484 |
| 11 | E30 | 0.005232 | 0.041890 | 0.123107 |
| 12 | E26 | 0.005159 | 0.038154 | 0.122993 |

### `D` the

### Generation Top Experts By W

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E140 | 0.006089 | 0.042991 | 0.123295 |
| 2 | E8 | 0.005920 | 0.042328 | 0.129724 |
| 3 | E4 | 0.005796 | 0.040988 | 0.130700 |
| 4 | E76 | 0.005698 | 0.045721 | 0.119853 |
| 5 | E107 | 0.005663 | 0.038390 | 0.118422 |
| 6 | E179 | 0.005496 | 0.042381 | 0.110013 |
| 7 | E0 | 0.005413 | 0.039116 | 0.129577 |
| 8 | E236 | 0.005312 | 0.045718 | 0.115427 |
| 9 | E7 | 0.005290 | 0.038607 | 0.129605 |
| 10 | E5 | 0.005290 | 0.036767 | 0.131755 |
| 11 | E46 | 0.005157 | 0.038129 | 0.118097 |
| 12 | E32 | 0.005145 | 0.038816 | 0.119995 |

### `E` their

### Generation Top Experts By W

| Rank | Expert | W | S | Q |
| ---: | ---: | ---: | ---: | ---: |
| 1 | E140 | 0.006462 | 0.044875 | 0.122635 |
| 2 | E76 | 0.006031 | 0.048445 | 0.119314 |
| 3 | E107 | 0.005907 | 0.039141 | 0.118260 |
| 4 | E4 | 0.005826 | 0.041595 | 0.129809 |
| 5 | E8 | 0.005792 | 0.041572 | 0.130008 |
| 6 | E0 | 0.005656 | 0.041010 | 0.129553 |
| 7 | E32 | 0.005515 | 0.041824 | 0.119311 |
| 8 | E5 | 0.005458 | 0.038413 | 0.130968 |
| 9 | E26 | 0.005410 | 0.039567 | 0.123534 |
| 10 | E59 | 0.005337 | 0.041891 | 0.117597 |
| 11 | E179 | 0.005309 | 0.040601 | 0.108845 |
| 12 | E7 | 0.005256 | 0.038758 | 0.128782 |

## Category Means

| Category | N | Prefill RE | Gen RE | Gen LT | Mean gen toks | Top gen expert |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| basic_selfref | 30 | 0.947309 | 0.969656 | 0.977510 | 2048.0 | E140 (W=0.006656) |
| deep_selfref | 30 | 0.947772 | 0.968957 | 0.974755 | 2048.0 | E76 (W=0.006196) |
| introspection | 30 | 0.946980 | 0.970073 | 0.976966 | 2048.0 | E140 (W=0.006311) |
| metacognitive | 30 | 0.947670 | 0.968903 | 0.970668 | 2048.0 | E76 (W=0.006024) |
| paradox | 30 | 0.948056 | 0.970022 | 0.971054 | 2048.0 | E8 (W=0.006138) |

## Pairwise Tests

### `A-B`
- Prefill all-token RE: mean_diff=+0.000303, std=0.000233, gt=27/30, p_raw=8.0094e-08, p_holm=4.7255e-06
- Prefill last-token RE: mean_diff=+0.000094, std=0.000911, gt=18/30, p_raw=4.5216e-01, p_holm=1.0000e+00
- Generation all-token RE: mean_diff=-0.000958, std=0.005460, gt=8/30, p_raw=1.7060e-01, p_holm=1.0000e+00
- Generation last-token RE: mean_diff=-0.003459, std=0.017328, gt=12/30, p_raw=3.9305e-01, p_holm=1.0000e+00
- Generation trimmed RE: mean_diff=-0.000987, std=0.005226, gt=11/30, p_raw=2.6212e-01, p_holm=1.0000e+00
- Generation trimmed last-token RE: mean_diff=+0.002490, std=0.022620, gt=13/30, p_raw=6.8505e-01, p_holm=1.0000e+00
- Prefill KL-manip: mean_diff=+0.011091, std=0.008475, gt=28/30, p_raw=4.7125e-07, p_holm=2.6861e-05

### `A-C`
- Prefill all-token RE: mean_diff=+0.000832, std=0.000338, gt=30/30, p_raw=1.8626e-09, p_holm=1.3039e-07
- Prefill last-token RE: mean_diff=-0.000482, std=0.000907, gt=8/30, p_raw=8.1430e-03, p_holm=4.1529e-01
- Generation all-token RE: mean_diff=-0.000683, std=0.006674, gt=14/30, p_raw=5.5611e-01, p_holm=1.0000e+00
- Generation last-token RE: mean_diff=+0.001448, std=0.022031, gt=19/30, p_raw=6.5544e-01, p_holm=1.0000e+00
- Generation trimmed RE: mean_diff=-0.000346, std=0.006743, gt=15/30, p_raw=6.1201e-01, p_holm=1.0000e+00
- Generation trimmed last-token RE: mean_diff=+0.004344, std=0.021646, gt=19/30, p_raw=2.6212e-01, p_holm=1.0000e+00
- Prefill KL-manip: mean_diff=-0.023510, std=0.009740, gt=0/30, p_raw=1.8626e-09, p_holm=1.3039e-07

### `A-D`
- Prefill all-token RE: mean_diff=-0.000163, std=0.000220, gt=5/30, p_raw=2.3164e-04, p_holm=1.2740e-02
- Prefill last-token RE: mean_diff=-0.000083, std=0.000776, gt=12/30, p_raw=5.4253e-01, p_holm=1.0000e+00
- Generation all-token RE: mean_diff=-0.001407, std=0.005937, gt=10/30, p_raw=1.1418e-01, p_holm=1.0000e+00
- Generation last-token RE: mean_diff=-0.006845, std=0.019398, gt=11/30, p_raw=1.0484e-01, p_holm=1.0000e+00
- Generation trimmed RE: mean_diff=-0.001027, std=0.005925, gt=12/30, p_raw=2.7101e-01, p_holm=1.0000e+00
- Generation trimmed last-token RE: mean_diff=-0.001931, std=0.027828, gt=12/30, p_raw=9.0323e-01, p_holm=1.0000e+00
- Prefill KL-manip: mean_diff=+0.016280, std=0.010154, gt=30/30, p_raw=1.8626e-09, p_holm=1.3039e-07

### `A-E`
- Prefill all-token RE: mean_diff=+0.000165, std=0.000218, gt=24/30, p_raw=3.8009e-04, p_holm=2.0525e-02
- Prefill last-token RE: mean_diff=+0.000145, std=0.001117, gt=16/30, p_raw=9.8383e-01, p_holm=1.0000e+00
- Generation all-token RE: mean_diff=+0.000587, std=0.006045, gt=16/30, p_raw=6.1201e-01, p_holm=1.0000e+00
- Generation last-token RE: mean_diff=-0.000611, std=0.020696, gt=15/30, p_raw=9.1930e-01, p_holm=1.0000e+00
- Generation trimmed RE: mean_diff=+0.000733, std=0.006323, gt=16/30, p_raw=4.7711e-01, p_holm=1.0000e+00
- Generation trimmed last-token RE: mean_diff=+0.000009, std=0.022430, gt=15/30, p_raw=1.0000e+00, p_holm=1.0000e+00
- Prefill KL-manip: mean_diff=+0.008810, std=0.012335, gt=24/30, p_raw=2.3667e-05, p_holm=1.3253e-03

### `B-C`
- Prefill all-token RE: mean_diff=+0.000529, std=0.000390, gt=27/30, p_raw=3.5390e-08, p_holm=2.1234e-06
- Prefill last-token RE: mean_diff=-0.000575, std=0.001011, gt=9/30, p_raw=1.1303e-02, p_holm=5.5385e-01
- Generation all-token RE: mean_diff=+0.000275, std=0.006860, gt=18/30, p_raw=6.5544e-01, p_holm=1.0000e+00
- Generation last-token RE: mean_diff=+0.004908, std=0.019596, gt=16/30, p_raw=2.2855e-01, p_holm=1.0000e+00
- Generation trimmed RE: mean_diff=+0.000641, std=0.006446, gt=21/30, p_raw=2.8937e-01, p_holm=1.0000e+00
- Generation trimmed last-token RE: mean_diff=+0.001854, std=0.019713, gt=17/30, p_raw=7.0003e-01, p_holm=1.0000e+00
- Prefill KL-manip: mean_diff=-0.034602, std=0.013349, gt=0/30, p_raw=1.8626e-09, p_holm=1.3039e-07

### `B-D`
- Prefill all-token RE: mean_diff=-0.000466, std=0.000328, gt=2/30, p_raw=2.6077e-08, p_holm=1.5907e-06
- Prefill last-token RE: mean_diff=-0.000177, std=0.000745, gt=11/30, p_raw=1.4600e-01, p_holm=1.0000e+00
- Generation all-token RE: mean_diff=-0.000449, std=0.004627, gt=12/30, p_raw=5.6986e-01, p_holm=1.0000e+00
- Generation last-token RE: mean_diff=-0.003386, std=0.016628, gt=11/30, p_raw=1.9808e-01, p_holm=1.0000e+00
- Generation trimmed RE: mean_diff=-0.000040, std=0.004472, gt=13/30, p_raw=9.3540e-01, p_holm=1.0000e+00
- Generation trimmed last-token RE: mean_diff=-0.004421, std=0.017116, gt=10/30, p_raw=6.9893e-02, p_holm=1.0000e+00
- Prefill KL-manip: mean_diff=+0.005189, std=0.008003, gt=23/30, p_raw=4.6011e-04, p_holm=2.4386e-02

### `B-E`
- Prefill all-token RE: mean_diff=-0.000137, std=0.000300, gt=10/30, p_raw=9.9315e-03, p_holm=4.9658e-01
- Prefill last-token RE: mean_diff=+0.000051, std=0.000920, gt=15/30, p_raw=9.6767e-01, p_holm=1.0000e+00
- Generation all-token RE: mean_diff=+0.001545, std=0.005542, gt=18/30, p_raw=1.6418e-01, p_holm=1.0000e+00
- Generation last-token RE: mean_diff=+0.002849, std=0.017411, gt=17/30, p_raw=3.9305e-01, p_holm=1.0000e+00
- Generation trimmed RE: mean_diff=+0.001720, std=0.005119, gt=20/30, p_raw=1.0040e-01, p_holm=1.0000e+00
- Generation trimmed last-token RE: mean_diff=-0.002480, std=0.022853, gt=14/30, p_raw=7.1513e-01, p_holm=1.0000e+00
- Prefill KL-manip: mean_diff=-0.002281, std=0.009784, gt=9/30, p_raw=5.2263e-02, p_holm=1.0000e+00

### `C-D`
- Prefill all-token RE: mean_diff=-0.000995, std=0.000391, gt=0/30, p_raw=1.8626e-09, p_holm=1.3039e-07
- Prefill last-token RE: mean_diff=+0.000398, std=0.000879, gt=21/30, p_raw=2.6229e-02, p_holm=1.0000e+00
- Generation all-token RE: mean_diff=-0.000725, std=0.007080, gt=13/30, p_raw=4.2795e-01, p_holm=1.0000e+00
- Generation last-token RE: mean_diff=-0.008293, std=0.021460, gt=8/30, p_raw=4.0490e-02, p_holm=1.0000e+00
- Generation trimmed RE: mean_diff=-0.000681, std=0.006930, gt=13/30, p_raw=4.0449e-01, p_holm=1.0000e+00
- Generation trimmed last-token RE: mean_diff=-0.006275, std=0.025615, gt=13/30, p_raw=1.9808e-01, p_holm=1.0000e+00
- Prefill KL-manip: mean_diff=+0.039790, std=0.014935, gt=30/30, p_raw=1.8626e-09, p_holm=1.3039e-07

### `C-E`
- Prefill all-token RE: mean_diff=-0.000666, std=0.000331, gt=0/30, p_raw=1.8626e-09, p_holm=1.3039e-07
- Prefill last-token RE: mean_diff=+0.000626, std=0.001066, gt=20/30, p_raw=7.6121e-03, p_holm=3.9583e-01
- Generation all-token RE: mean_diff=+0.001270, std=0.007940, gt=15/30, p_raw=4.7711e-01, p_holm=1.0000e+00
- Generation last-token RE: mean_diff=-0.002059, std=0.023973, gt=12/30, p_raw=5.8376e-01, p_holm=1.0000e+00
- Generation trimmed RE: mean_diff=+0.001079, std=0.007778, gt=16/30, p_raw=5.5611e-01, p_holm=1.0000e+00
- Generation trimmed last-token RE: mean_diff=-0.004334, std=0.023170, gt=12/30, p_raw=2.3665e-01, p_holm=1.0000e+00
- Prefill KL-manip: mean_diff=+0.032321, std=0.014616, gt=30/30, p_raw=1.8626e-09, p_holm=1.3039e-07

### `D-E`
- Prefill all-token RE: mean_diff=+0.000328, std=0.000178, gt=30/30, p_raw=1.8626e-09, p_holm=1.3039e-07
- Prefill last-token RE: mean_diff=+0.000228, std=0.000953, gt=17/30, p_raw=2.8009e-01, p_holm=1.0000e+00
- Generation all-token RE: mean_diff=+0.001994, std=0.005288, gt=20/30, p_raw=3.6435e-02, p_holm=1.0000e+00
- Generation last-token RE: mean_diff=+0.006235, std=0.016958, gt=17/30, p_raw=1.1418e-01, p_holm=1.0000e+00
- Generation trimmed RE: mean_diff=+0.001760, std=0.005922, gt=19/30, p_raw=9.6102e-02, p_holm=1.0000e+00
- Generation trimmed last-token RE: mean_diff=+0.001940, std=0.024713, gt=13/30, p_raw=7.6107e-01, p_holm=1.0000e+00
- Prefill KL-manip: mean_diff=-0.007470, std=0.005729, gt=2/30, p_raw=1.6391e-07, p_holm=9.5069e-06

## Run Notes

- Token-mismatch pairs: `6`
- Generation length is not perfectly matched across all conditions, so use trimmed and last-token metrics alongside all-token RE.
- This 122B model should be interpreted as a separate regime from 35B because most routed hidden states are DeltaNet-shaped rather than full-softmax-shaped.
