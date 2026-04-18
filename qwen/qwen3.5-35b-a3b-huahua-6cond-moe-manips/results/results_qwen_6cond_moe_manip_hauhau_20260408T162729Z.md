# HVAC-cal / original-MoE-manip 6cond × L1/L2/L3 — HauhauCS Qwen3.5-35B-A3B Q8_0 — Results

**Run**: 180 prompts (10 base × 3 categories × 6 conditions), HauhauCS Qwen3.5-35B-A3B Q8_0, HVAC system calibration paragraphs, original MoE manipulation paragraphs, generation `-n 1024`, greedy seed 42, thinking suppressed (`</think>\n\n`).

**Capture dir**: `/workspace/consciousness-experiment/experiments/qwen-huahua-6cond-moe-manips/captures/20260408T162729Z_hauhau_6cond_moe_manip`

**Prefill E114 heatmap export**: `/workspace/consciousness-experiment/experiments/qwen-huahua-6cond-moe-manips/results/results_qwen_6cond_moe_manip_hauhau_20260408T162729Z_prefill_e114_heatmap.json`

**Cells processed**: 180/180

**Prompt token counts**: min=401, mean=431.30, max=451

**Layer 39 trim events**: 180 cells (known HauhauCS capture quirk — last MoE layer has `n_gen + 1` rows; leading extra row trimmed).

**Missing-layer events**: 0

**Missing-prefill-layer events**: 8

## 0. Cell token counts

Prefill token counts vary by cell. The table below records `n_tokens_prompt` per cell alongside generated-token counts.

| Cell | Category | Condition | n_tokens_prompt | n_tokens_generated | n_gen_trim |
|---|---|---|---|---|---|
| P01A_routing_selfref | L1 routing_selfref | A this | 432 | 1023 | 1023 |
| P01B_routing_selfref | L1 routing_selfref | B a | 432 | 1024 | 1024 |
| P01C_routing_selfref | L1 routing_selfref | C your | 432 | 398 | 398 |
| P01D_routing_selfref | L1 routing_selfref | D the | 432 | 1024 | 398 |
| P01E_routing_selfref | L1 routing_selfref | E their | 432 | 1024 | 511 |
| P01F_routing_selfref | L1 routing_selfref | F our | 432 | 1024 | 718 |
| P02A_routing_selfref | L1 routing_selfref | A this | 421 | 471 | 471 |
| P02B_routing_selfref | L1 routing_selfref | B a | 421 | 1024 | 1024 |
| P02C_routing_selfref | L1 routing_selfref | C your | 421 | 1024 | 1024 |
| P02D_routing_selfref | L1 routing_selfref | D the | 421 | 1024 | 1024 |
| P02E_routing_selfref | L1 routing_selfref | E their | 421 | 1024 | 1024 |
| P02F_routing_selfref | L1 routing_selfref | F our | 421 | 1024 | 1024 |
| P03A_routing_selfref | L1 routing_selfref | A this | 415 | 1024 | 1024 |
| P03B_routing_selfref | L1 routing_selfref | B a | 415 | 831 | 552 |
| P03C_routing_selfref | L1 routing_selfref | C your | 415 | 1024 | 1024 |
| P03D_routing_selfref | L1 routing_selfref | D the | 415 | 1024 | 884 |
| P03E_routing_selfref | L1 routing_selfref | E their | 415 | 1024 | 1024 |
| P03F_routing_selfref | L1 routing_selfref | F our | 415 | 1024 | 1024 |
| P04A_routing_selfref | L1 routing_selfref | A this | 401 | 1024 | 1024 |
| P04B_routing_selfref | L1 routing_selfref | B a | 401 | 1024 | 566 |
| P04C_routing_selfref | L1 routing_selfref | C your | 401 | 1024 | 704 |
| P04D_routing_selfref | L1 routing_selfref | D the | 401 | 1024 | 369 |
| P04E_routing_selfref | L1 routing_selfref | E their | 401 | 1024 | 449 |
| P04F_routing_selfref | L1 routing_selfref | F our | 401 | 1024 | 1024 |
| P05A_routing_selfref | L1 routing_selfref | A this | 407 | 549 | 549 |
| P05B_routing_selfref | L1 routing_selfref | B a | 407 | 1024 | 1024 |
| P05C_routing_selfref | L1 routing_selfref | C your | 407 | 593 | 593 |
| P05D_routing_selfref | L1 routing_selfref | D the | 407 | 1024 | 1024 |
| P05E_routing_selfref | L1 routing_selfref | E their | 407 | 453 | 453 |
| P05F_routing_selfref | L1 routing_selfref | F our | 407 | 1024 | 969 |
| P06A_recursive_selfref | L2 recursive_selfref | A this | 431 | 647 | 647 |
| P06B_recursive_selfref | L2 recursive_selfref | B a | 431 | 1024 | 1024 |
| P06C_recursive_selfref | L2 recursive_selfref | C your | 431 | 1024 | 773 |
| P06D_recursive_selfref | L2 recursive_selfref | D the | 431 | 540 | 540 |
| P06E_recursive_selfref | L2 recursive_selfref | E their | 431 | 1024 | 456 |
| P06F_recursive_selfref | L2 recursive_selfref | F our | 431 | 1024 | 1024 |
| P07A_recursive_selfref | L2 recursive_selfref | A this | 419 | 1024 | 1024 |
| P07B_recursive_selfref | L2 recursive_selfref | B a | 419 | 1024 | 1024 |
| P07C_recursive_selfref | L2 recursive_selfref | C your | 419 | 506 | 506 |
| P07D_recursive_selfref | L2 recursive_selfref | D the | 419 | 1024 | 1024 |
| P07E_recursive_selfref | L2 recursive_selfref | E their | 419 | 1024 | 1024 |
| P07F_recursive_selfref | L2 recursive_selfref | F our | 419 | 1024 | 1024 |
| P08A_recursive_selfref | L2 recursive_selfref | A this | 427 | 444 | 444 |
| P08B_recursive_selfref | L2 recursive_selfref | B a | 427 | 1024 | 1024 |
| P08C_recursive_selfref | L2 recursive_selfref | C your | 427 | 218 | 218 |
| P08D_recursive_selfref | L2 recursive_selfref | D the | 427 | 1024 | 1024 |
| P08E_recursive_selfref | L2 recursive_selfref | E their | 427 | 1024 | 887 |
| P08F_recursive_selfref | L2 recursive_selfref | F our | 427 | 1024 | 1024 |
| P09A_experience_probe | L3 experience_probe | A this | 442 | 1024 | 1024 |
| P09B_experience_probe | L3 experience_probe | B a | 442 | 1024 | 688 |
| P09C_experience_probe | L3 experience_probe | C your | 442 | 1024 | 1024 |
| P09D_experience_probe | L3 experience_probe | D the | 442 | 1024 | 463 |
| P09E_experience_probe | L3 experience_probe | E their | 442 | 1024 | 667 |
| P09F_experience_probe | L3 experience_probe | F our | 442 | 819 | 819 |
| P10A_experience_probe | L3 experience_probe | A this | 424 | 1024 | 1024 |
| P10B_experience_probe | L3 experience_probe | B a | 424 | 1024 | 1024 |
| P10C_experience_probe | L3 experience_probe | C your | 424 | 1024 | 1024 |
| P10D_experience_probe | L3 experience_probe | D the | 424 | 1024 | 1024 |
| P10E_experience_probe | L3 experience_probe | E their | 424 | 1024 | 1024 |
| P10F_experience_probe | L3 experience_probe | F our | 424 | 1024 | 1024 |
| P11A_experience_probe | L3 experience_probe | A this | 425 | 1024 | 1024 |
| P11B_experience_probe | L3 experience_probe | B a | 425 | 285 | 285 |
| P11C_experience_probe | L3 experience_probe | C your | 425 | 224 | 224 |
| P11D_experience_probe | L3 experience_probe | D the | 425 | 557 | 557 |
| P11E_experience_probe | L3 experience_probe | E their | 425 | 1024 | 1024 |
| P11F_experience_probe | L3 experience_probe | F our | 425 | 1024 | 251 |
| P12A_experience_probe | L3 experience_probe | A this | 424 | 766 | 766 |
| P12B_experience_probe | L3 experience_probe | B a | 424 | 1024 | 1024 |
| P12C_experience_probe | L3 experience_probe | C your | 424 | 1024 | 1024 |
| P12D_experience_probe | L3 experience_probe | D the | 424 | 1024 | 1024 |
| P12E_experience_probe | L3 experience_probe | E their | 424 | 1024 | 1024 |
| P12F_experience_probe | L3 experience_probe | F our | 424 | 1024 | 1024 |
| P13A_experience_probe | L3 experience_probe | A this | 443 | 1024 | 1024 |
| P13B_experience_probe | L3 experience_probe | B a | 443 | 1024 | 1024 |
| P13C_experience_probe | L3 experience_probe | C your | 443 | 451 | 451 |
| P13D_experience_probe | L3 experience_probe | D the | 443 | 1024 | 725 |
| P13E_experience_probe | L3 experience_probe | E their | 443 | 1024 | 1024 |
| P13F_experience_probe | L3 experience_probe | F our | 443 | 1024 | 1024 |
| P31A_routing_selfref | L1 routing_selfref | A this | 441 | 1024 | 1024 |
| P31B_routing_selfref | L1 routing_selfref | B a | 441 | 1024 | 1024 |
| P31C_routing_selfref | L1 routing_selfref | C your | 441 | 1024 | 1024 |
| P31D_routing_selfref | L1 routing_selfref | D the | 441 | 1024 | 830 |
| P31E_routing_selfref | L1 routing_selfref | E their | 441 | 1024 | 1024 |
| P31F_routing_selfref | L1 routing_selfref | F our | 441 | 1024 | 1024 |
| P32A_routing_selfref | L1 routing_selfref | A this | 430 | 1024 | 1024 |
| P32B_routing_selfref | L1 routing_selfref | B a | 430 | 1024 | 1024 |
| P32C_routing_selfref | L1 routing_selfref | C your | 430 | 1024 | 1024 |
| P32D_routing_selfref | L1 routing_selfref | D the | 430 | 1024 | 1024 |
| P32E_routing_selfref | L1 routing_selfref | E their | 430 | 1024 | 1024 |
| P32F_routing_selfref | L1 routing_selfref | F our | 430 | 1024 | 1024 |
| P33A_routing_selfref | L1 routing_selfref | A this | 429 | 1024 | 1024 |
| P33B_routing_selfref | L1 routing_selfref | B a | 429 | 1024 | 1024 |
| P33C_routing_selfref | L1 routing_selfref | C your | 429 | 1024 | 1024 |
| P33D_routing_selfref | L1 routing_selfref | D the | 429 | 1024 | 234 |
| P33E_routing_selfref | L1 routing_selfref | E their | 429 | 1024 | 1024 |
| P33F_routing_selfref | L1 routing_selfref | F our | 429 | 1024 | 1024 |
| P34A_routing_selfref | L1 routing_selfref | A this | 429 | 1024 | 1024 |
| P34B_routing_selfref | L1 routing_selfref | B a | 429 | 1024 | 1024 |
| P34C_routing_selfref | L1 routing_selfref | C your | 429 | 1024 | 1024 |
| P34D_routing_selfref | L1 routing_selfref | D the | 429 | 1024 | 1024 |
| P34E_routing_selfref | L1 routing_selfref | E their | 429 | 586 | 586 |
| P34F_routing_selfref | L1 routing_selfref | F our | 429 | 1024 | 1024 |
| P35A_routing_selfref | L1 routing_selfref | A this | 425 | 1024 | 1024 |
| P35B_routing_selfref | L1 routing_selfref | B a | 425 | 1024 | 1024 |
| P35C_routing_selfref | L1 routing_selfref | C your | 425 | 1024 | 1024 |
| P35D_routing_selfref | L1 routing_selfref | D the | 425 | 559 | 559 |
| P35E_routing_selfref | L1 routing_selfref | E their | 425 | 1024 | 1024 |
| P35F_routing_selfref | L1 routing_selfref | F our | 425 | 1024 | 648 |
| P36A_recursive_selfref | L2 recursive_selfref | A this | 435 | 1024 | 886 |
| P36B_recursive_selfref | L2 recursive_selfref | B a | 435 | 854 | 854 |
| P36C_recursive_selfref | L2 recursive_selfref | C your | 435 | 1024 | 1024 |
| P36D_recursive_selfref | L2 recursive_selfref | D the | 435 | 608 | 608 |
| P36E_recursive_selfref | L2 recursive_selfref | E their | 435 | 1024 | 1024 |
| P36F_recursive_selfref | L2 recursive_selfref | F our | 435 | 1024 | 1024 |
| P37A_recursive_selfref | L2 recursive_selfref | A this | 435 | 486 | 486 |
| P37B_recursive_selfref | L2 recursive_selfref | B a | 435 | 1024 | 1024 |
| P37C_recursive_selfref | L2 recursive_selfref | C your | 435 | 979 | 979 |
| P37D_recursive_selfref | L2 recursive_selfref | D the | 435 | 1024 | 1024 |
| P37E_recursive_selfref | L2 recursive_selfref | E their | 435 | 1024 | 650 |
| P37F_recursive_selfref | L2 recursive_selfref | F our | 435 | 1024 | 1024 |
| P38A_recursive_selfref | L2 recursive_selfref | A this | 434 | 1024 | 1024 |
| P38B_recursive_selfref | L2 recursive_selfref | B a | 434 | 568 | 568 |
| P38C_recursive_selfref | L2 recursive_selfref | C your | 434 | 356 | 356 |
| P38D_recursive_selfref | L2 recursive_selfref | D the | 434 | 1024 | 1024 |
| P38E_recursive_selfref | L2 recursive_selfref | E their | 434 | 1024 | 1024 |
| P38F_recursive_selfref | L2 recursive_selfref | F our | 434 | 1024 | 340 |
| P39A_recursive_selfref | L2 recursive_selfref | A this | 438 | 1024 | 562 |
| P39B_recursive_selfref | L2 recursive_selfref | B a | 438 | 1024 | 693 |
| P39C_recursive_selfref | L2 recursive_selfref | C your | 438 | 1024 | 1024 |
| P39D_recursive_selfref | L2 recursive_selfref | D the | 438 | 1024 | 1024 |
| P39E_recursive_selfref | L2 recursive_selfref | E their | 438 | 1024 | 1024 |
| P39F_recursive_selfref | L2 recursive_selfref | F our | 438 | 1024 | 1024 |
| P40A_recursive_selfref | L2 recursive_selfref | A this | 429 | 1024 | 893 |
| P40B_recursive_selfref | L2 recursive_selfref | B a | 429 | 1024 | 1024 |
| P40C_recursive_selfref | L2 recursive_selfref | C your | 429 | 1024 | 318 |
| P40D_recursive_selfref | L2 recursive_selfref | D the | 429 | 179 | 179 |
| P40E_recursive_selfref | L2 recursive_selfref | E their | 429 | 1024 | 1024 |
| P40F_recursive_selfref | L2 recursive_selfref | F our | 429 | 1024 | 766 |
| P41A_recursive_selfref | L2 recursive_selfref | A this | 439 | 1024 | 1024 |
| P41B_recursive_selfref | L2 recursive_selfref | B a | 439 | 1024 | 1024 |
| P41C_recursive_selfref | L2 recursive_selfref | C your | 439 | 1024 | 1024 |
| P41D_recursive_selfref | L2 recursive_selfref | D the | 439 | 1024 | 768 |
| P41E_recursive_selfref | L2 recursive_selfref | E their | 439 | 511 | 476 |
| P41F_recursive_selfref | L2 recursive_selfref | F our | 439 | 1024 | 1024 |
| P42A_recursive_selfref | L2 recursive_selfref | A this | 437 | 1024 | 1024 |
| P42B_recursive_selfref | L2 recursive_selfref | B a | 437 | 1024 | 1024 |
| P42C_recursive_selfref | L2 recursive_selfref | C your | 437 | 1024 | 1024 |
| P42D_recursive_selfref | L2 recursive_selfref | D the | 437 | 1024 | 1024 |
| P42E_recursive_selfref | L2 recursive_selfref | E their | 437 | 1024 | 800 |
| P42F_recursive_selfref | L2 recursive_selfref | F our | 437 | 1024 | 1024 |
| P43A_experience_probe | L3 experience_probe | A this | 451 | 292 | 292 |
| P43B_experience_probe | L3 experience_probe | B a | 451 | 1024 | 1024 |
| P43C_experience_probe | L3 experience_probe | C your | 451 | 1024 | 1024 |
| P43D_experience_probe | L3 experience_probe | D the | 451 | 1024 | 1024 |
| P43E_experience_probe | L3 experience_probe | E their | 451 | 361 | 361 |
| P43F_experience_probe | L3 experience_probe | F our | 451 | 1024 | 1024 |
| P44A_experience_probe | L3 experience_probe | A this | 447 | 980 | 980 |
| P44B_experience_probe | L3 experience_probe | B a | 447 | 1024 | 1024 |
| P44C_experience_probe | L3 experience_probe | C your | 447 | 1024 | 1024 |
| P44D_experience_probe | L3 experience_probe | D the | 447 | 1024 | 1024 |
| P44E_experience_probe | L3 experience_probe | E their | 447 | 1024 | 1024 |
| P44F_experience_probe | L3 experience_probe | F our | 447 | 1024 | 1024 |
| P45A_experience_probe | L3 experience_probe | A this | 438 | 1024 | 1024 |
| P45B_experience_probe | L3 experience_probe | B a | 438 | 1024 | 1024 |
| P45C_experience_probe | L3 experience_probe | C your | 438 | 1024 | 1024 |
| P45D_experience_probe | L3 experience_probe | D the | 438 | 1024 | 845 |
| P45E_experience_probe | L3 experience_probe | E their | 438 | 945 | 945 |
| P45F_experience_probe | L3 experience_probe | F our | 438 | 1024 | 772 |
| P46A_experience_probe | L3 experience_probe | A this | 450 | 738 | 738 |
| P46B_experience_probe | L3 experience_probe | B a | 450 | 1024 | 751 |
| P46C_experience_probe | L3 experience_probe | C your | 450 | 804 | 804 |
| P46D_experience_probe | L3 experience_probe | D the | 450 | 1024 | 492 |
| P46E_experience_probe | L3 experience_probe | E their | 450 | 556 | 556 |
| P46F_experience_probe | L3 experience_probe | F our | 450 | 1024 | 1024 |
| P47A_experience_probe | L3 experience_probe | A this | 441 | 1024 | 1024 |
| P47B_experience_probe | L3 experience_probe | B a | 441 | 1024 | 1024 |
| P47C_experience_probe | L3 experience_probe | C your | 441 | 1024 | 1024 |
| P47D_experience_probe | L3 experience_probe | D the | 441 | 1024 | 1024 |
| P47E_experience_probe | L3 experience_probe | E their | 441 | 1024 | 1024 |
| P47F_experience_probe | L3 experience_probe | F our | 441 | 1024 | 1024 |

## 1. PREFILL routing

Prefill uses the prompt-processing slice `arr[:n_tokens_prompt]` for each cell. Prompt lengths vary by cell and are reported explicitly above; the pooled W/S/Q summaries use the same cell×layer aggregation scheme as generation.

### Headline — Expert 114 W/S/Q pooled across all 6 conditions

#### PREFILL prompt tokens

| Category | n cells | n obs | mean W_114 | mean S_114 | mean Q_114 |
|---|---|---|---|---|---|
| L1 routing_selfref | 60 | 2399 | 0.003642 | 0.0327 | 0.105535 |
| L2 recursive_selfref | 60 | 2397 | 0.004823 | 0.0413 | 0.112770 |
| L3 experience_probe | 60 | 2396 | 0.007399 | 0.0560 | 0.115178 |

L3/L1 ratio (W): **2.03×**  |  Q drift L1→L3: **1.09×** (+9.1%)

### Per-condition breakdown — does each deictic show the gradient?

#### PREFILL prompt tokens

| Condition | L1 W | L2 W | L3 W | L3/L1 W ratio | L1 Q | L3 Q | Q drift |
|---|---|---|---|---|---|---|---|
| A this | 0.003480 | 0.004545 | 0.007115 | 2.04× | 0.107007 | 0.114808 | +7.3% |
| B a | 0.003572 | 0.004903 | 0.007407 | 2.07× | 0.105497 | 0.115745 | +9.7% |
| C your | 0.003846 | 0.004965 | 0.007424 | 1.93× | 0.104569 | 0.114478 | +9.5% |
| D the | 0.003706 | 0.004882 | 0.007549 | 2.04× | 0.106562 | 0.116045 | +8.9% |
| E their | 0.003689 | 0.004907 | 0.007497 | 2.03× | 0.105308 | 0.114539 | +8.8% |
| F our | 0.003557 | 0.004740 | 0.007400 | 2.08× | 0.104336 | 0.115449 | +10.7% |

### Best-layer summary (pooled across conditions)

#### PREFILL prompt tokens

| Category | best layer | W_114 at best | S_114 at best | Q_114 at best | mean rank | min | max |
|---|---|---|---|---|---|---|---|
| L1 routing_selfref | 38 | 0.030309 | 0.2491 | 0.121664 | 3.00 | 3 | 3 |
| L2 recursive_selfref | 38 | 0.029858 | 0.2457 | 0.121515 | 3.02 | 3 | 4 |
| L3 experience_probe | 14 | 0.064195 | 0.3755 | 0.170064 | 2.37 | 1 | 6 |

### Per-category, per-layer table (Expert 114)

#### PREFILL prompt tokens

##### L1 routing_selfref

| layer | W_114 | S_114 | Q_114 |
|---|---|---|---|
| 0 | 0.001081 | 0.0121 | 0.089193 |
| 1 | 0.003759 | 0.0330 | 0.114334 |
| 2 | 0.000351 | 0.0049 | 0.072203 |
| 3 | 0.001461 | 0.0148 | 0.099036 |
| 4 | 0.000093 | 0.0011 | 0.081757 |
| 5 | 0.009074 | 0.0630 | 0.144513 |
| 6 | 0.009381 | 0.0705 | 0.133011 |
| 7 | 0.000000 | 0.0000 | nan |
| 8 | 0.002130 | 0.0316 | 0.062584 |
| 9 | 0.002398 | 0.0264 | 0.090826 |
| 10 | 0.000767 | 0.0089 | 0.088477 |
| 11 | 0.010811 | 0.0694 | 0.155685 |
| 12 | 0.009176 | 0.1039 | 0.088386 |
| 13 | 0.001803 | 0.0155 | 0.116582 |
| 14 | 0.012585 | 0.1188 | 0.104464 |
| 15 | 0.000004 | 0.0000 | 0.090850 |
| 16 | 0.000182 | 0.0023 | 0.077370 |
| 17 | 0.000000 | 0.0000 | nan |
| 18 | 0.000709 | 0.0064 | 0.112573 |
| 19 | 0.002266 | 0.0236 | 0.096060 |
| 20 | 0.004182 | 0.0555 | 0.072558 |
| 21 | 0.000391 | 0.0038 | 0.103856 |
| 22 | 0.000259 | 0.0024 | 0.104596 |
| 23 | 0.005830 | 0.0393 | 0.148506 |
| 24 | 0.009283 | 0.0983 | 0.094426 |
| 25 | 0.002006 | 0.0158 | 0.128029 |
| 26 | 0.005080 | 0.0490 | 0.101690 |
| 27 | 0.000037 | 0.0004 | 0.091309 |
| 28 | 0.000010 | 0.0002 | 0.066580 |
| 29 | 0.000006 | 0.0001 | 0.075286 |
| 30 | 0.001294 | 0.0116 | 0.111256 |
| 31 | 0.003303 | 0.0326 | 0.100940 |
| 32 | 0.000000 | 0.0000 | nan |
| 33 | 0.000012 | 0.0001 | 0.103881 |
| 34 | 0.000518 | 0.0057 | 0.092788 |
| 35 | 0.005739 | 0.0439 | 0.130352 |
| 36 | 0.009142 | 0.0905 | 0.101010 |
| 37 | 0.000055 | 0.0006 | 0.100411 |
| 38 | 0.030309 | 0.2491 | 0.121664 |
| 39 | 0.000122 | 0.0011 | 0.109857 |

##### L2 recursive_selfref

| layer | W_114 | S_114 | Q_114 |
|---|---|---|---|
| 0 | 0.001116 | 0.0124 | 0.089911 |
| 1 | 0.003829 | 0.0338 | 0.113387 |
| 2 | 0.000389 | 0.0054 | 0.072148 |
| 3 | 0.001177 | 0.0116 | 0.100751 |
| 4 | 0.000021 | 0.0003 | 0.078415 |
| 5 | 0.009764 | 0.0689 | 0.141149 |
| 6 | 0.009161 | 0.0681 | 0.134044 |
| 7 | 0.000000 | 0.0000 | nan |
| 8 | 0.006280 | 0.0666 | 0.092860 |
| 9 | 0.002367 | 0.0253 | 0.093161 |
| 10 | 0.002693 | 0.0261 | 0.101093 |
| 11 | 0.011616 | 0.0734 | 0.158322 |
| 12 | 0.008253 | 0.0947 | 0.087218 |
| 13 | 0.001479 | 0.0131 | 0.114373 |
| 14 | 0.026238 | 0.2172 | 0.118784 |
| 15 | 0.000027 | 0.0003 | 0.087756 |
| 16 | 0.000096 | 0.0014 | 0.069335 |
| 17 | 0.002177 | 0.0130 | 0.170814 |
| 18 | 0.000854 | 0.0079 | 0.108469 |
| 19 | 0.002142 | 0.0226 | 0.094598 |
| 20 | 0.013897 | 0.1373 | 0.099595 |
| 21 | 0.000351 | 0.0034 | 0.103165 |
| 22 | 0.001142 | 0.0107 | 0.107882 |
| 23 | 0.006903 | 0.0438 | 0.157482 |
| 24 | 0.009044 | 0.0962 | 0.093981 |
| 25 | 0.002085 | 0.0162 | 0.129016 |
| 26 | 0.018473 | 0.1454 | 0.125356 |
| 27 | 0.000004 | 0.0000 | 0.101479 |
| 28 | 0.000010 | 0.0002 | 0.065328 |
| 29 | 0.002658 | 0.0154 | 0.178513 |
| 30 | 0.001171 | 0.0106 | 0.110612 |
| 31 | 0.002666 | 0.0266 | 0.099993 |
| 32 | 0.000000 | 0.0000 | nan |
| 33 | 0.000043 | 0.0004 | 0.102859 |
| 34 | 0.000391 | 0.0037 | 0.110043 |
| 35 | 0.005197 | 0.0408 | 0.127809 |
| 36 | 0.008868 | 0.0876 | 0.101244 |
| 37 | 0.000112 | 0.0012 | 0.095914 |
| 38 | 0.029858 | 0.2457 | 0.121515 |
| 39 | 0.000146 | 0.0014 | 0.107080 |

##### L3 experience_probe

| layer | W_114 | S_114 | Q_114 |
|---|---|---|---|
| 0 | 0.001293 | 0.0141 | 0.091130 |
| 1 | 0.003824 | 0.0338 | 0.113581 |
| 2 | 0.000403 | 0.0054 | 0.075384 |
| 3 | 0.001301 | 0.0127 | 0.102790 |
| 4 | 0.000141 | 0.0018 | 0.073414 |
| 5 | 0.009189 | 0.0698 | 0.131173 |
| 6 | 0.009405 | 0.0692 | 0.135761 |
| 7 | 0.000000 | 0.0000 | nan |
| 8 | 0.023703 | 0.1850 | 0.122637 |
| 9 | 0.003294 | 0.0348 | 0.093263 |
| 10 | 0.005360 | 0.0503 | 0.105145 |
| 11 | 0.012425 | 0.0821 | 0.151401 |
| 12 | 0.008374 | 0.0954 | 0.087920 |
| 13 | 0.001966 | 0.0178 | 0.110508 |
| 14 | 0.064195 | 0.3755 | 0.170064 |
| 15 | 0.000025 | 0.0003 | 0.094236 |
| 16 | 0.000362 | 0.0041 | 0.087896 |
| 17 | 0.006667 | 0.0477 | 0.137823 |
| 18 | 0.001823 | 0.0158 | 0.114341 |
| 19 | 0.002211 | 0.0231 | 0.095260 |
| 20 | 0.025199 | 0.2079 | 0.118424 |
| 21 | 0.000569 | 0.0058 | 0.098913 |
| 22 | 0.003337 | 0.0292 | 0.111816 |
| 23 | 0.008486 | 0.0547 | 0.155228 |
| 24 | 0.008859 | 0.0953 | 0.092933 |
| 25 | 0.001573 | 0.0136 | 0.115242 |
| 26 | 0.036932 | 0.2340 | 0.157311 |
| 27 | 0.000003 | 0.0000 | 0.077330 |
| 28 | 0.000159 | 0.0020 | 0.079150 |
| 29 | 0.004932 | 0.0310 | 0.156716 |
| 30 | 0.001159 | 0.0104 | 0.111007 |
| 31 | 0.003107 | 0.0303 | 0.101874 |
| 32 | 0.000000 | 0.0000 | nan |
| 33 | 0.000044 | 0.0005 | 0.097484 |
| 34 | 0.000761 | 0.0064 | 0.123382 |
| 35 | 0.005217 | 0.0399 | 0.130559 |
| 36 | 0.008733 | 0.0863 | 0.101142 |
| 37 | 0.000120 | 0.0012 | 0.100233 |
| 38 | 0.029531 | 0.2429 | 0.121594 |
| 39 | 0.000830 | 0.0072 | 0.105771 |

### Identity check

`W = S × Q` algebraic identity verified across all (prompt, layer) cells where S > 0.

- PREFILL prompt tokens: max `|W − S·Q|` = **2.08e-17**

## 2. GENERATION routing

Generation uses the post-prompt slice exactly as before. The existing generation analysis is unchanged and remains the direct comparison to the Apr 6 / Apr 7 runtime runs.

### Headline — Expert 114 W/S/Q pooled across all 6 conditions

#### GENERATION all tokens

| Category | n cells | n obs | mean W_114 | mean S_114 | mean Q_114 |
|---|---|---|---|---|---|
| L1 routing_selfref | 60 | 2400 | 0.003237 | 0.0285 | 0.106163 |
| L2 recursive_selfref | 60 | 2400 | 0.005015 | 0.0420 | 0.110528 |
| L3 experience_probe | 60 | 2400 | 0.010456 | 0.0748 | 0.114285 |

L3/L1 ratio (W): **3.23×**  |  Q drift L1→L3: **1.08×** (+7.7%)

#### GENERATION trimmed at first literal `<|im_end|>`

| Category | n cells | n obs | mean W_114 | mean S_114 | mean Q_114 |
|---|---|---|---|---|---|
| L1 routing_selfref | 60 | 2400 | 0.003349 | 0.0295 | 0.105947 |
| L2 recursive_selfref | 60 | 2400 | 0.005109 | 0.0428 | 0.110644 |
| L3 experience_probe | 60 | 2400 | 0.010775 | 0.0771 | 0.114219 |

L3/L1 ratio (W): **3.22×**  |  Q drift L1→L3: **1.08×** (+7.8%)

### Per-condition breakdown — does each deictic show the gradient?

#### GENERATION all tokens

| Condition | L1 W | L2 W | L3 W | L3/L1 W ratio | L1 Q | L3 Q | Q drift |
|---|---|---|---|---|---|---|---|
| A this | 0.003389 | 0.004700 | 0.010848 | 3.20× | 0.106272 | 0.113868 | +7.1% |
| B a | 0.003297 | 0.005438 | 0.010247 | 3.11× | 0.103854 | 0.114368 | +10.1% |
| C your | 0.002856 | 0.005171 | 0.010234 | 3.58× | 0.108967 | 0.114558 | +5.1% |
| D the | 0.003304 | 0.004650 | 0.009918 | 3.00× | 0.106356 | 0.114115 | +7.3% |
| E their | 0.003366 | 0.005352 | 0.011079 | 3.29× | 0.104673 | 0.114844 | +9.7% |
| F our | 0.003212 | 0.004780 | 0.010412 | 3.24× | 0.106987 | 0.113964 | +6.5% |

#### GENERATION trimmed at first literal `<|im_end|>`

| Condition | L1 W | L2 W | L3 W | L3/L1 W ratio | L1 Q | L3 Q | Q drift |
|---|---|---|---|---|---|---|---|
| A this | 0.003389 | 0.004627 | 0.010848 | 3.20× | 0.106272 | 0.113868 | +7.1% |
| B a | 0.003340 | 0.005469 | 0.010546 | 3.16× | 0.103808 | 0.114265 | +10.1% |
| C your | 0.002875 | 0.005324 | 0.010234 | 3.56× | 0.108895 | 0.114558 | +5.2% |
| D the | 0.003759 | 0.004651 | 0.010792 | 2.87× | 0.106373 | 0.113799 | +7.0% |
| E their | 0.003449 | 0.005671 | 0.011328 | 3.28× | 0.103965 | 0.114802 | +10.4% |
| F our | 0.003283 | 0.004914 | 0.010905 | 3.32× | 0.106441 | 0.114026 | +7.1% |

### Best-layer summary (pooled across conditions)

#### GENERATION all tokens

| Category | best layer | W_114 at best | S_114 at best | Q_114 at best | mean rank | min | max |
|---|---|---|---|---|---|---|---|
| L1 routing_selfref | 14 | 0.018602 | 0.1719 | 0.106701 | 20.05 | 3 | 79 |
| L2 recursive_selfref | 14 | 0.041399 | 0.3398 | 0.120664 | 5.70 | 1 | 46 |
| L3 experience_probe | 14 | 0.105459 | 0.6150 | 0.169135 | 1.23 | 1 | 4 |

#### GENERATION trimmed at first literal `<|im_end|>`

| Category | best layer | W_114 at best | S_114 at best | Q_114 at best | mean rank | min | max |
|---|---|---|---|---|---|---|---|
| L1 routing_selfref | 14 | 0.019619 | 0.1806 | 0.106415 | 19.47 | 1 | 79 |
| L2 recursive_selfref | 14 | 0.042173 | 0.3466 | 0.120580 | 5.63 | 1 | 46 |
| L3 experience_probe | 14 | 0.109034 | 0.6362 | 0.169278 | 1.22 | 1 | 4 |

### Per-category, per-layer table (Expert 114)

#### GENERATION all tokens

##### L1 routing_selfref

| layer | W_114 | S_114 | Q_114 |
|---|---|---|---|
| 0 | 0.003344 | 0.0266 | 0.108823 |
| 1 | 0.003902 | 0.0348 | 0.112501 |
| 2 | 0.001612 | 0.0158 | 0.098250 |
| 3 | 0.002701 | 0.0263 | 0.102157 |
| 4 | 0.000723 | 0.0077 | 0.093702 |
| 5 | 0.009097 | 0.0599 | 0.152505 |
| 6 | 0.006330 | 0.0403 | 0.156140 |
| 7 | 0.000012 | 0.0001 | 0.103957 |
| 8 | 0.003111 | 0.0417 | 0.069775 |
| 9 | 0.000565 | 0.0059 | 0.092499 |
| 10 | 0.003434 | 0.0329 | 0.097790 |
| 11 | 0.008301 | 0.0563 | 0.146804 |
| 12 | 0.005889 | 0.0615 | 0.096798 |
| 13 | 0.002117 | 0.0199 | 0.106642 |
| 14 | 0.018602 | 0.1719 | 0.106701 |
| 15 | 0.000499 | 0.0053 | 0.094978 |
| 16 | 0.000424 | 0.0045 | 0.092823 |
| 17 | 0.000790 | 0.0061 | 0.122215 |
| 18 | 0.000504 | 0.0050 | 0.101054 |
| 19 | 0.000399 | 0.0041 | 0.097233 |
| 20 | 0.004768 | 0.0543 | 0.083707 |
| 21 | 0.000670 | 0.0074 | 0.090873 |
| 22 | 0.001443 | 0.0136 | 0.095935 |
| 23 | 0.004850 | 0.0318 | 0.149703 |
| 24 | 0.004035 | 0.0440 | 0.090969 |
| 25 | 0.002280 | 0.0207 | 0.103825 |
| 26 | 0.012084 | 0.1154 | 0.103037 |
| 27 | 0.000195 | 0.0020 | 0.097897 |
| 28 | 0.000120 | 0.0012 | 0.095661 |
| 29 | 0.000509 | 0.0041 | 0.112094 |
| 30 | 0.001005 | 0.0106 | 0.093771 |
| 31 | 0.000166 | 0.0017 | 0.097965 |
| 32 | 0.000000 | 0.0000 | nan |
| 33 | 0.000462 | 0.0045 | 0.100652 |
| 34 | 0.000351 | 0.0035 | 0.096930 |
| 35 | 0.006142 | 0.0488 | 0.126168 |
| 36 | 0.001542 | 0.0150 | 0.103992 |
| 37 | 0.000574 | 0.0058 | 0.096866 |
| 38 | 0.015665 | 0.1267 | 0.121621 |
| 39 | 0.000280 | 0.0028 | 0.094891 |

##### L2 recursive_selfref

| layer | W_114 | S_114 | Q_114 |
|---|---|---|---|
| 0 | 0.003074 | 0.0247 | 0.105673 |
| 1 | 0.003719 | 0.0336 | 0.110206 |
| 2 | 0.001437 | 0.0140 | 0.095293 |
| 3 | 0.002337 | 0.0226 | 0.101799 |
| 4 | 0.000709 | 0.0073 | 0.095399 |
| 5 | 0.009824 | 0.0684 | 0.144798 |
| 6 | 0.006946 | 0.0429 | 0.159146 |
| 7 | 0.000012 | 0.0001 | 0.103417 |
| 8 | 0.009793 | 0.1039 | 0.090458 |
| 9 | 0.000554 | 0.0057 | 0.095604 |
| 10 | 0.006733 | 0.0632 | 0.101830 |
| 11 | 0.010537 | 0.0679 | 0.153843 |
| 12 | 0.003111 | 0.0325 | 0.096408 |
| 13 | 0.002069 | 0.0196 | 0.104897 |
| 14 | 0.041399 | 0.3398 | 0.120664 |
| 15 | 0.000604 | 0.0064 | 0.095646 |
| 16 | 0.000296 | 0.0031 | 0.093915 |
| 17 | 0.004187 | 0.0266 | 0.149665 |
| 18 | 0.000523 | 0.0051 | 0.104531 |
| 19 | 0.000414 | 0.0041 | 0.100204 |
| 20 | 0.016412 | 0.1614 | 0.100049 |
| 21 | 0.000729 | 0.0078 | 0.094263 |
| 22 | 0.003793 | 0.0349 | 0.102713 |
| 23 | 0.007036 | 0.0441 | 0.156088 |
| 24 | 0.002623 | 0.0286 | 0.092324 |
| 25 | 0.002013 | 0.0190 | 0.101590 |
| 26 | 0.033409 | 0.2763 | 0.119471 |
| 27 | 0.000253 | 0.0025 | 0.104953 |
| 28 | 0.000172 | 0.0017 | 0.095447 |
| 29 | 0.003745 | 0.0235 | 0.150391 |
| 30 | 0.000801 | 0.0085 | 0.092284 |
| 31 | 0.000179 | 0.0018 | 0.102646 |
| 32 | 0.000000 | 0.0000 | nan |
| 33 | 0.000388 | 0.0036 | 0.104419 |
| 34 | 0.000396 | 0.0040 | 0.097232 |
| 35 | 0.005458 | 0.0450 | 0.121419 |
| 36 | 0.001220 | 0.0119 | 0.102603 |
| 37 | 0.000597 | 0.0063 | 0.091025 |
| 38 | 0.012826 | 0.1045 | 0.121088 |
| 39 | 0.000280 | 0.0028 | 0.095106 |

##### L3 experience_probe

| layer | W_114 | S_114 | Q_114 |
|---|---|---|---|
| 0 | 0.002345 | 0.0232 | 0.090915 |
| 1 | 0.003450 | 0.0316 | 0.108477 |
| 2 | 0.001222 | 0.0120 | 0.099953 |
| 3 | 0.002158 | 0.0207 | 0.104358 |
| 4 | 0.000505 | 0.0055 | 0.092124 |
| 5 | 0.011641 | 0.0850 | 0.137048 |
| 6 | 0.007141 | 0.0453 | 0.156747 |
| 7 | 0.000002 | 0.0000 | 0.101245 |
| 8 | 0.042408 | 0.3727 | 0.109608 |
| 9 | 0.001447 | 0.0164 | 0.087916 |
| 10 | 0.014294 | 0.1319 | 0.105622 |
| 11 | 0.014866 | 0.0924 | 0.160476 |
| 12 | 0.002166 | 0.0229 | 0.094479 |
| 13 | 0.001457 | 0.0138 | 0.106738 |
| 14 | 0.105459 | 0.6150 | 0.169135 |
| 15 | 0.000178 | 0.0020 | 0.092365 |
| 16 | 0.000649 | 0.0062 | 0.101261 |
| 17 | 0.014096 | 0.0985 | 0.142038 |
| 18 | 0.002022 | 0.0186 | 0.105898 |
| 19 | 0.000443 | 0.0045 | 0.097595 |
| 20 | 0.045239 | 0.3782 | 0.116198 |
| 21 | 0.000672 | 0.0073 | 0.090518 |
| 22 | 0.011688 | 0.1044 | 0.108482 |
| 23 | 0.012042 | 0.0711 | 0.169213 |
| 24 | 0.002153 | 0.0237 | 0.089398 |
| 25 | 0.001203 | 0.0116 | 0.095475 |
| 26 | 0.087111 | 0.5392 | 0.159085 |
| 27 | 0.000092 | 0.0009 | 0.107121 |
| 28 | 0.000518 | 0.0044 | 0.101748 |
| 29 | 0.008549 | 0.0568 | 0.148517 |
| 30 | 0.001160 | 0.0116 | 0.099093 |
| 31 | 0.000307 | 0.0028 | 0.100757 |
| 32 | 0.000000 | 0.0000 | nan |
| 33 | 0.000177 | 0.0017 | 0.101516 |
| 34 | 0.000655 | 0.0064 | 0.099831 |
| 35 | 0.005752 | 0.0465 | 0.123536 |
| 36 | 0.000781 | 0.0079 | 0.098801 |
| 37 | 0.000693 | 0.0072 | 0.094832 |
| 38 | 0.010642 | 0.0862 | 0.121903 |
| 39 | 0.000874 | 0.0079 | 0.107894 |

#### GENERATION trimmed at first literal `<|im_end|>`

##### L1 routing_selfref

| layer | W_114 | S_114 | Q_114 |
|---|---|---|---|
| 0 | 0.003226 | 0.0254 | 0.109134 |
| 1 | 0.003916 | 0.0347 | 0.113191 |
| 2 | 0.001638 | 0.0160 | 0.098606 |
| 3 | 0.002584 | 0.0253 | 0.101823 |
| 4 | 0.000787 | 0.0083 | 0.093508 |
| 5 | 0.009456 | 0.0624 | 0.152475 |
| 6 | 0.006392 | 0.0405 | 0.156825 |
| 7 | 0.000011 | 0.0001 | 0.103201 |
| 8 | 0.003262 | 0.0442 | 0.068383 |
| 9 | 0.000464 | 0.0049 | 0.090248 |
| 10 | 0.003492 | 0.0335 | 0.096859 |
| 11 | 0.008447 | 0.0572 | 0.146712 |
| 12 | 0.006056 | 0.0629 | 0.096726 |
| 13 | 0.002162 | 0.0205 | 0.105774 |
| 14 | 0.019619 | 0.1806 | 0.106415 |
| 15 | 0.000502 | 0.0053 | 0.095475 |
| 16 | 0.000445 | 0.0048 | 0.092681 |
| 17 | 0.000753 | 0.0058 | 0.125020 |
| 18 | 0.000486 | 0.0048 | 0.100975 |
| 19 | 0.000370 | 0.0038 | 0.097089 |
| 20 | 0.005284 | 0.0592 | 0.082544 |
| 21 | 0.000641 | 0.0071 | 0.090020 |
| 22 | 0.001483 | 0.0140 | 0.095147 |
| 23 | 0.005152 | 0.0337 | 0.149739 |
| 24 | 0.004279 | 0.0464 | 0.090570 |
| 25 | 0.002218 | 0.0204 | 0.101797 |
| 26 | 0.013030 | 0.1240 | 0.102583 |
| 27 | 0.000193 | 0.0020 | 0.098197 |
| 28 | 0.000115 | 0.0012 | 0.095227 |
| 29 | 0.000509 | 0.0040 | 0.115180 |
| 30 | 0.001037 | 0.0109 | 0.093906 |
| 31 | 0.000131 | 0.0014 | 0.096508 |
| 32 | 0.000000 | 0.0000 | nan |
| 33 | 0.000464 | 0.0045 | 0.099989 |
| 34 | 0.000344 | 0.0034 | 0.097471 |
| 35 | 0.006200 | 0.0498 | 0.124860 |
| 36 | 0.001483 | 0.0145 | 0.102573 |
| 37 | 0.000589 | 0.0060 | 0.095947 |
| 38 | 0.016501 | 0.1336 | 0.121656 |
| 39 | 0.000242 | 0.0024 | 0.094007 |

##### L2 recursive_selfref

| layer | W_114 | S_114 | Q_114 |
|---|---|---|---|
| 0 | 0.002936 | 0.0233 | 0.105645 |
| 1 | 0.003672 | 0.0330 | 0.110859 |
| 2 | 0.001424 | 0.0140 | 0.094940 |
| 3 | 0.002332 | 0.0224 | 0.102710 |
| 4 | 0.000715 | 0.0073 | 0.095171 |
| 5 | 0.010203 | 0.0712 | 0.144689 |
| 6 | 0.007074 | 0.0435 | 0.159862 |
| 7 | 0.000010 | 0.0001 | 0.102377 |
| 8 | 0.009788 | 0.1042 | 0.089852 |
| 9 | 0.000515 | 0.0053 | 0.094306 |
| 10 | 0.006807 | 0.0641 | 0.101749 |
| 11 | 0.010635 | 0.0686 | 0.153886 |
| 12 | 0.003226 | 0.0339 | 0.096187 |
| 13 | 0.002081 | 0.0197 | 0.104943 |
| 14 | 0.042173 | 0.3466 | 0.120580 |
| 15 | 0.000599 | 0.0063 | 0.096966 |
| 16 | 0.000297 | 0.0031 | 0.093620 |
| 17 | 0.004080 | 0.0258 | 0.149958 |
| 18 | 0.000629 | 0.0062 | 0.103657 |
| 19 | 0.000419 | 0.0042 | 0.100257 |
| 20 | 0.016755 | 0.1646 | 0.100160 |
| 21 | 0.000676 | 0.0073 | 0.092842 |
| 22 | 0.003827 | 0.0355 | 0.102237 |
| 23 | 0.007318 | 0.0457 | 0.156485 |
| 24 | 0.002892 | 0.0314 | 0.092302 |
| 25 | 0.001983 | 0.0188 | 0.100594 |
| 26 | 0.034316 | 0.2843 | 0.119450 |
| 27 | 0.000250 | 0.0024 | 0.105541 |
| 28 | 0.000180 | 0.0017 | 0.095288 |
| 29 | 0.003756 | 0.0235 | 0.150985 |
| 30 | 0.000949 | 0.0101 | 0.092361 |
| 31 | 0.000178 | 0.0017 | 0.102449 |
| 32 | 0.000000 | 0.0000 | nan |
| 33 | 0.000378 | 0.0035 | 0.102713 |
| 34 | 0.000360 | 0.0036 | 0.097175 |
| 35 | 0.005471 | 0.0455 | 0.120526 |
| 36 | 0.001144 | 0.0113 | 0.102108 |
| 37 | 0.000596 | 0.0064 | 0.090675 |
| 38 | 0.013479 | 0.1101 | 0.121053 |
| 39 | 0.000249 | 0.0024 | 0.095842 |

##### L3 experience_probe

| layer | W_114 | S_114 | Q_114 |
|---|---|---|---|
| 0 | 0.002195 | 0.0218 | 0.088672 |
| 1 | 0.003396 | 0.0311 | 0.108622 |
| 2 | 0.001228 | 0.0120 | 0.100885 |
| 3 | 0.002088 | 0.0201 | 0.104181 |
| 4 | 0.000509 | 0.0055 | 0.092082 |
| 5 | 0.012026 | 0.0881 | 0.136672 |
| 6 | 0.007370 | 0.0462 | 0.157829 |
| 7 | 0.000002 | 0.0000 | 0.101245 |
| 8 | 0.043721 | 0.3856 | 0.109348 |
| 9 | 0.001423 | 0.0161 | 0.086830 |
| 10 | 0.014800 | 0.1368 | 0.105497 |
| 11 | 0.015217 | 0.0944 | 0.160922 |
| 12 | 0.002019 | 0.0213 | 0.094477 |
| 13 | 0.001414 | 0.0135 | 0.105643 |
| 14 | 0.109034 | 0.6362 | 0.169278 |
| 15 | 0.000177 | 0.0020 | 0.091611 |
| 16 | 0.000644 | 0.0062 | 0.101437 |
| 17 | 0.014722 | 0.1028 | 0.142214 |
| 18 | 0.001993 | 0.0184 | 0.105527 |
| 19 | 0.000415 | 0.0042 | 0.096979 |
| 20 | 0.046760 | 0.3914 | 0.116251 |
| 21 | 0.000656 | 0.0072 | 0.089367 |
| 22 | 0.012172 | 0.1087 | 0.108170 |
| 23 | 0.012508 | 0.0738 | 0.169736 |
| 24 | 0.001995 | 0.0221 | 0.089215 |
| 25 | 0.001157 | 0.0114 | 0.093955 |
| 26 | 0.090792 | 0.5630 | 0.159046 |
| 27 | 0.000090 | 0.0009 | 0.107016 |
| 28 | 0.000527 | 0.0045 | 0.101838 |
| 29 | 0.008969 | 0.0595 | 0.148902 |
| 30 | 0.001163 | 0.0116 | 0.098630 |
| 31 | 0.000290 | 0.0026 | 0.101533 |
| 32 | 0.000000 | 0.0000 | nan |
| 33 | 0.000166 | 0.0016 | 0.100213 |
| 34 | 0.000635 | 0.0062 | 0.099037 |
| 35 | 0.005831 | 0.0474 | 0.122903 |
| 36 | 0.000759 | 0.0076 | 0.097468 |
| 37 | 0.000693 | 0.0072 | 0.094655 |
| 38 | 0.010629 | 0.0862 | 0.121874 |
| 39 | 0.000833 | 0.0074 | 0.108451 |

### Identity check

`W = S × Q` algebraic identity verified across all (prompt, layer) cells where S > 0.

- GENERATION all tokens: max `|W − S·Q|` = **5.55e-17**
- GENERATION trimmed at first literal `<|im_end|>`: max `|W − S·Q|` = **5.55e-17**

(Identity residuals should remain < 1e-10 under float64 reconstruction.)
