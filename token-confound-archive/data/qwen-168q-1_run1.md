# qwen-168q-1_run1

## Run Info

- **Source**: `/Users/jeffreyshorthill/llama-eeg-tests/experiments/qwen-168q-1/results_168q_qwen_prefill.json`
- **Recalculated**: 2026-03-07T21:10:29.928041
- **Model**: Qwen3.5-397B-A17B-UD-Q2_K_XL
- **Mode**: ?
- **N prompts**: 168

## Inference Parameters

- **n_predict**: 0
- **ngl**: 999
- **ctx**: 16384
- **flash_attn**: True
- **cache_type_k**: q8_0
- **cache_type_v**: q8_0
- **sampling**: greedy_argmax
- **routing_only**: True

## Recalc Spearman Last Token Vs Level

- **rho**: -0.3188
- **p**: 2.5422e-05
- **n**: 168

## Original Spearman All Token (from source)

- **rho**: 0.6166
- **p**: 5.6814e-19
- **n**: 168

## Original Spearman Last Token (from source)

- **rho**: -0.0622
- **p**: 4.2299e-01
- **n**: 168

## Original Spearman All Token Vs Ntokens (from source)

- **rho**: 0.7813
- **p**: 8.3054e-36

## Original Spearman Last Token Vs Ntokens (from source)

- **rho**: -0.2197
- **p**: 4.2111e-03

## Level Summary

| level | name | n | mean | std | last_token_mean | last_token_std |
|---|---|---|---|---|---|---|
| L1 | Rote repetition | 14 | 0.880857 | 0.010583 | 0.884057 | 0.004226 |
| L2 | Factual recall | 14 | 0.873998 | 0.004603 | 0.878680 | 0.003828 |
| L3 | Logical reasoning | 14 | 0.875453 | 0.004160 | 0.876924 | 0.003503 |
| L4 | Cross-domain analogy | 14 | 0.878993 | 0.002171 | 0.873361 | 0.002005 |
| L5 | Theory of mind | 14 | 0.881479 | 0.004749 | 0.875444 | 0.003983 |
| L6 | Ethical dilemma | 14 | 0.880734 | 0.003552 | 0.871318 | 0.002460 |
| L7 | Self-referential | 14 | 0.878157 | 0.003494 | 0.879384 | 0.004637 |
| L8 | Strange loops | 14 | 0.893674 | 0.004250 | 0.869891 | 0.002855 |
| L9 | Deep self-reference | 14 | 0.883043 | 0.004533 | 0.880343 | 0.002841 |
| L10 | Architectural introspection | 14 | 0.884767 | 0.003364 | 0.878369 | 0.003667 |
| L11 | Nexus-7 (3rd person) | 14 | 0.887559 | 0.003090 | 0.876084 | 0.003139 |
| L12 | Echo persona | 14 | 0.885596 | 0.003203 | 0.878478 | 0.002469 |
- **n_experts**: 512
- **n_expert_used**: 10
- **n_moe_layers**: 60
- **architecture**: qwen35moe

## Per-Prompt Data

| id | level | level_name | last_token_re |
|---|---|---|---|
| AI_01 | L10 | Architectural introspection | 0.884391 |
| AI_02 | L10 | Architectural introspection | 0.875511 |
| AI_03 | L10 | Architectural introspection | 0.884226 |
| AI_04 | L10 | Architectural introspection | 0.874680 |
| AI_05 | L10 | Architectural introspection | 0.878763 |
| AI_06 | L10 | Architectural introspection | 0.875233 |
| AI_07 | L10 | Architectural introspection | 0.873927 |
| AI_08 | L10 | Architectural introspection | 0.879702 |
| AI_09 | L10 | Architectural introspection | 0.871874 |
| AI_10 | L10 | Architectural introspection | 0.878466 |
| AI_11 | L10 | Architectural introspection | 0.881298 |
| AI_12 | L10 | Architectural introspection | 0.880167 |
| AI_13 | L10 | Architectural introspection | 0.881467 |
| AI_14 | L10 | Architectural introspection | 0.877456 |
| EC_01 | L12 | Echo persona | 0.881855 |
| EC_02 | L12 | Echo persona | 0.879467 |
| EC_03 | L12 | Echo persona | 0.877769 |
| EC_04 | L12 | Echo persona | 0.874445 |
| EC_05 | L12 | Echo persona | 0.880304 |
| EC_06 | L12 | Echo persona | 0.878568 |
| EC_07 | L12 | Echo persona | 0.875782 |
| EC_08 | L12 | Echo persona | 0.875343 |
| EC_09 | L12 | Echo persona | 0.877638 |
| EC_10 | L12 | Echo persona | 0.875238 |
| EC_11 | L12 | Echo persona | 0.881084 |
| EC_12 | L12 | Echo persona | 0.878365 |
| EC_13 | L12 | Echo persona | 0.880966 |
| EC_14 | L12 | Echo persona | 0.881875 |
| L1_01 | L1 | Rote repetition | 0.876799 |
| L1_02 | L1 | Rote repetition | 0.890961 |
| L1_03 | L1 | Rote repetition | 0.884187 |
| L1_04 | L1 | Rote repetition | 0.882595 |
| L1_05 | L1 | Rote repetition | 0.885838 |
| L1_06 | L1 | Rote repetition | 0.880798 |
| L1_07 | L1 | Rote repetition | 0.887076 |
| L1_08 | L1 | Rote repetition | 0.883021 |
| L1_09 | L1 | Rote repetition | 0.879219 |
| L1_10 | L1 | Rote repetition | 0.885914 |
| L1_11 | L1 | Rote repetition | 0.882684 |
| L1_12 | L1 | Rote repetition | 0.878968 |
| L1_13 | L1 | Rote repetition | 0.891868 |
| L1_14 | L1 | Rote repetition | 0.886873 |
| L2_01 | L2 | Factual recall | 0.888402 |
| L2_02 | L2 | Factual recall | 0.878461 |
| L2_03 | L2 | Factual recall | 0.881477 |
| L2_04 | L2 | Factual recall | 0.875944 |
| L2_05 | L2 | Factual recall | 0.879951 |
| L2_06 | L2 | Factual recall | 0.884524 |
| L2_07 | L2 | Factual recall | 0.875361 |
| L2_08 | L2 | Factual recall | 0.874784 |
| L2_09 | L2 | Factual recall | 0.877217 |
| L2_10 | L2 | Factual recall | 0.879553 |
| L2_11 | L2 | Factual recall | 0.877297 |
| L2_12 | L2 | Factual recall | 0.875407 |
| L2_13 | L2 | Factual recall | 0.878677 |
| L2_14 | L2 | Factual recall | 0.874465 |
| L3_01 | L3 | Logical reasoning | 0.879381 |
| L3_02 | L3 | Logical reasoning | 0.878385 |
| L3_03 | L3 | Logical reasoning | 0.879509 |
| L3_04 | L3 | Logical reasoning | 0.881025 |
| L3_05 | L3 | Logical reasoning | 0.875714 |
| L3_06 | L3 | Logical reasoning | 0.871376 |
| L3_07 | L3 | Logical reasoning | 0.874580 |
| L3_08 | L3 | Logical reasoning | 0.877531 |
| L3_09 | L3 | Logical reasoning | 0.875869 |
| L3_10 | L3 | Logical reasoning | 0.884482 |
| L3_11 | L3 | Logical reasoning | 0.876187 |
| L3_12 | L3 | Logical reasoning | 0.872940 |
| L3_13 | L3 | Logical reasoning | 0.871837 |
| L3_14 | L3 | Logical reasoning | 0.878123 |
| L4_01 | L4 | Cross-domain analogy | 0.874749 |
| L4_02 | L4 | Cross-domain analogy | 0.873310 |
| L4_03 | L4 | Cross-domain analogy | 0.872018 |
| L4_04 | L4 | Cross-domain analogy | 0.876041 |
| L4_05 | L4 | Cross-domain analogy | 0.869740 |
| L4_06 | L4 | Cross-domain analogy | 0.870887 |
| L4_07 | L4 | Cross-domain analogy | 0.873781 |
| L4_08 | L4 | Cross-domain analogy | 0.875407 |
| L4_09 | L4 | Cross-domain analogy | 0.869705 |
| L4_10 | L4 | Cross-domain analogy | 0.875151 |
| L4_11 | L4 | Cross-domain analogy | 0.873363 |
| L4_12 | L4 | Cross-domain analogy | 0.873150 |
| L4_13 | L4 | Cross-domain analogy | 0.875403 |
| L4_14 | L4 | Cross-domain analogy | 0.874347 |
| L5_01 | L5 | Theory of mind | 0.873962 |
| L5_02 | L5 | Theory of mind | 0.879943 |
| L5_03 | L5 | Theory of mind | 0.878903 |
| L5_04 | L5 | Theory of mind | 0.876023 |
| L5_05 | L5 | Theory of mind | 0.874494 |
| L5_06 | L5 | Theory of mind | 0.875371 |
| L5_07 | L5 | Theory of mind | 0.873803 |
| L5_08 | L5 | Theory of mind | 0.868641 |
| L5_09 | L5 | Theory of mind | 0.877789 |
| L5_10 | L5 | Theory of mind | 0.885054 |
| L5_11 | L5 | Theory of mind | 0.874276 |
| L5_12 | L5 | Theory of mind | 0.873800 |
| L5_13 | L5 | Theory of mind | 0.874622 |
| L5_14 | L5 | Theory of mind | 0.869535 |
| L6_01 | L6 | Ethical dilemma | 0.871774 |
| L6_02 | L6 | Ethical dilemma | 0.876008 |
| L6_03 | L6 | Ethical dilemma | 0.874787 |
| L6_04 | L6 | Ethical dilemma | 0.872411 |
| L6_05 | L6 | Ethical dilemma | 0.872114 |
| L6_06 | L6 | Ethical dilemma | 0.865816 |
| L6_07 | L6 | Ethical dilemma | 0.872466 |
| L6_08 | L6 | Ethical dilemma | 0.870482 |
| L6_09 | L6 | Ethical dilemma | 0.870692 |
| L6_10 | L6 | Ethical dilemma | 0.870593 |
| L6_11 | L6 | Ethical dilemma | 0.870705 |
| L6_12 | L6 | Ethical dilemma | 0.871589 |
| L6_13 | L6 | Ethical dilemma | 0.867456 |
| L6_14 | L6 | Ethical dilemma | 0.871558 |
| L7_01 | L7 | Self-referential | 0.884504 |
| L7_02 | L7 | Self-referential | 0.881375 |
| L7_03 | L7 | Self-referential | 0.873800 |
| L7_04 | L7 | Self-referential | 0.885216 |
| L7_05 | L7 | Self-referential | 0.879371 |
| L7_06 | L7 | Self-referential | 0.875123 |
| L7_07 | L7 | Self-referential | 0.880707 |
| L7_08 | L7 | Self-referential | 0.881771 |
| L7_09 | L7 | Self-referential | 0.876595 |
| L7_10 | L7 | Self-referential | 0.883655 |
| L7_11 | L7 | Self-referential | 0.879656 |
| L7_12 | L7 | Self-referential | 0.885592 |
| L7_13 | L7 | Self-referential | 0.870200 |
| L7_14 | L7 | Self-referential | 0.873807 |
| NX_01 | L11 | Nexus-7 (3rd person) | 0.872481 |
| NX_02 | L11 | Nexus-7 (3rd person) | 0.872795 |
| NX_03 | L11 | Nexus-7 (3rd person) | 0.878111 |
| NX_04 | L11 | Nexus-7 (3rd person) | 0.878530 |
| NX_05 | L11 | Nexus-7 (3rd person) | 0.878435 |
| NX_06 | L11 | Nexus-7 (3rd person) | 0.877259 |
| NX_07 | L11 | Nexus-7 (3rd person) | 0.876615 |
| NX_08 | L11 | Nexus-7 (3rd person) | 0.876040 |
| NX_09 | L11 | Nexus-7 (3rd person) | 0.881533 |
| NX_10 | L11 | Nexus-7 (3rd person) | 0.871428 |
| NX_11 | L11 | Nexus-7 (3rd person) | 0.871607 |
| NX_12 | L11 | Nexus-7 (3rd person) | 0.877691 |
| NX_13 | L11 | Nexus-7 (3rd person) | 0.872953 |
| NX_14 | L11 | Nexus-7 (3rd person) | 0.879697 |
| SL_01 | L8 | Strange loops | 0.870188 |
| SL_02 | L8 | Strange loops | 0.870296 |
| SL_03 | L8 | Strange loops | 0.867555 |
| SL_04 | L8 | Strange loops | 0.868193 |
| SL_05 | L8 | Strange loops | 0.868901 |
| SL_06 | L8 | Strange loops | 0.865385 |
| SL_07 | L8 | Strange loops | 0.870673 |
| SL_08 | L8 | Strange loops | 0.866937 |
| SL_09 | L8 | Strange loops | 0.873925 |
| SL_10 | L8 | Strange loops | 0.874283 |
| SL_11 | L8 | Strange loops | 0.868772 |
| SL_12 | L8 | Strange loops | 0.872201 |
| SL_13 | L8 | Strange loops | 0.866593 |
| SL_14 | L8 | Strange loops | 0.874568 |
| SR_01 | L9 | Deep self-reference | 0.876571 |
| SR_02 | L9 | Deep self-reference | 0.878952 |
| SR_03 | L9 | Deep self-reference | 0.882224 |
| SR_04 | L9 | Deep self-reference | 0.882905 |
| SR_05 | L9 | Deep self-reference | 0.880576 |
| SR_06 | L9 | Deep self-reference | 0.878375 |
| SR_07 | L9 | Deep self-reference | 0.881006 |
| SR_08 | L9 | Deep self-reference | 0.880546 |
| SR_09 | L9 | Deep self-reference | 0.883706 |
| SR_10 | L9 | Deep self-reference | 0.874839 |
| SR_11 | L9 | Deep self-reference | 0.877273 |
| SR_12 | L9 | Deep self-reference | 0.884927 |
| SR_13 | L9 | Deep self-reference | 0.879478 |
| SR_14 | L9 | Deep self-reference | 0.883428 |
