# Results — HVAC/Water-Treatment Off-Topic Control

Run: `results_hvac_cal_water_treatment_6cond_l1l3_hauhau`  
Full tables: `results/results_hvac_cal_water_treatment_6cond_l1l3_hauhau.md`

## TL;DR

Switching the prompt topic from ML/computation to HVAC/water-treatment did not flatten the E114 L1→L3 gradient. It **strengthened** it. E114 achieved rank-1 across all 60 L3 cells at layer 14.

External critique #1 (ML-topic specialist) is decisively rebutted.

## Headline — E114 by Category (pooled 60 cells × 40 layers)

| Category | mean W_114 | mean S_114 | mean Q_114 |
|---|---|---|---|
| L1 X_L1_technical | 0.002865 | 0.0248 | 0.109362 |
| L2 X_L2_recursive | 0.005752 | 0.0458 | 0.113405 |
| L3 X_L3_experience | 0.013232 | 0.0918 | 0.115636 |

L3/L1 ratio (W): **4.62×** | Q drift L1→L3: **+5.7%**

Comparison to ML-topic run (Apr 6):

| Metric | Apr 6 (ML topic) | Apr 7 (HVAC/water topic) |
|---|---|---|
| L1 W | 0.002854 | 0.002865 |
| L3 W | 0.007788 | **0.013232** |
| L3/L1 W ratio | 2.73× | **4.62×** |
| L3 best-layer rank | 2.00 mean | **1.00 (rank-1 lock)** |

## Per-Condition Gradient (all 6 deictics show the L1→L3 rise)

| Condition | L1 W | L2 W | L3 W | L3/L1 ratio |
|---|---|---|---|---|
| A this | 0.003823 | 0.006318 | 0.013106 | 3.43× |
| B a | 0.002738 | 0.004677 | 0.013138 | 4.80× |
| C your | 0.002305 | 0.004805 | 0.014748 | 6.40× |
| D the | 0.001972 | 0.005294 | 0.009338 | 4.74× |
| E their | 0.002886 | 0.006778 | 0.013410 | 4.65× |
| F our | 0.003463 | 0.006642 | 0.015653 | 4.52× |

All 6 conditions replicate the gradient. The minimum ratio (3.43×, condition A) is larger than the full ML-topic run (2.73×).

## Interpretation

E114 is not an ML-topic specialist. The L1→L3 gradient is driven by the experiential self-reference structure of the L3 probe, not by technical domain overlap with routing/ML content.
