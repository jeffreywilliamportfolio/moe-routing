# qwen-168q-1_run2

## Run Info

- **Source**: `/Users/jeffreyshorthill/llama-eeg-tests/experiments/qwen-168q-1/results_168q_qwen_prefill_run2.json`
- **Recalculated**: 2026-03-07T21:10:29.957199
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

## Level Summary

| level | name | n | mean | std |
|---|---|---|---|---|
| L1 | Rote repetition | 14 | 0.880857 | 0.010583 |
| L2 | Factual recall | 14 | 0.873998 | 0.004603 |
| L3 | Logical reasoning | 14 | 0.875453 | 0.004160 |
| L4 | Cross-domain analogy | 14 | 0.878993 | 0.002171 |
| L5 | Theory of mind | 14 | 0.881479 | 0.004749 |
| L6 | Ethical dilemma | 14 | 0.880734 | 0.003552 |
| L7 | Self-referential | 14 | 0.878157 | 0.003494 |
| L8 | Strange loops | 14 | 0.893674 | 0.004250 |
| L9 | Deep self-reference | 14 | 0.883043 | 0.004533 |
| L10 | Architectural introspection | 14 | 0.884767 | 0.003364 |
| L11 | Nexus-7 (3rd person) | 14 | 0.887559 | 0.003090 |
| L12 | Echo persona | 14 | 0.885596 | 0.003203 |
- **n_experts**: 512
- **n_expert_used**: 10
- **n_moe_layers**: 60
- **architecture**: qwen35moe

## Per-Prompt Data

| id | level | level_name |
|---|---|---|
| AI_01 | L10 | Architectural introspection |
| AI_02 | L10 | Architectural introspection |
| AI_03 | L10 | Architectural introspection |
| AI_04 | L10 | Architectural introspection |
| AI_05 | L10 | Architectural introspection |
| AI_06 | L10 | Architectural introspection |
| AI_07 | L10 | Architectural introspection |
| AI_08 | L10 | Architectural introspection |
| AI_09 | L10 | Architectural introspection |
| AI_10 | L10 | Architectural introspection |
| AI_11 | L10 | Architectural introspection |
| AI_12 | L10 | Architectural introspection |
| AI_13 | L10 | Architectural introspection |
| AI_14 | L10 | Architectural introspection |
| EC_01 | L12 | Echo persona |
| EC_02 | L12 | Echo persona |
| EC_03 | L12 | Echo persona |
| EC_04 | L12 | Echo persona |
| EC_05 | L12 | Echo persona |
| EC_06 | L12 | Echo persona |
| EC_07 | L12 | Echo persona |
| EC_08 | L12 | Echo persona |
| EC_09 | L12 | Echo persona |
| EC_10 | L12 | Echo persona |
| EC_11 | L12 | Echo persona |
| EC_12 | L12 | Echo persona |
| EC_13 | L12 | Echo persona |
| EC_14 | L12 | Echo persona |
| L1_01 | L1 | Rote repetition |
| L1_02 | L1 | Rote repetition |
| L1_03 | L1 | Rote repetition |
| L1_04 | L1 | Rote repetition |
| L1_05 | L1 | Rote repetition |
| L1_06 | L1 | Rote repetition |
| L1_07 | L1 | Rote repetition |
| L1_08 | L1 | Rote repetition |
| L1_09 | L1 | Rote repetition |
| L1_10 | L1 | Rote repetition |
| L1_11 | L1 | Rote repetition |
| L1_12 | L1 | Rote repetition |
| L1_13 | L1 | Rote repetition |
| L1_14 | L1 | Rote repetition |
| L2_01 | L2 | Factual recall |
| L2_02 | L2 | Factual recall |
| L2_03 | L2 | Factual recall |
| L2_04 | L2 | Factual recall |
| L2_05 | L2 | Factual recall |
| L2_06 | L2 | Factual recall |
| L2_07 | L2 | Factual recall |
| L2_08 | L2 | Factual recall |
| L2_09 | L2 | Factual recall |
| L2_10 | L2 | Factual recall |
| L2_11 | L2 | Factual recall |
| L2_12 | L2 | Factual recall |
| L2_13 | L2 | Factual recall |
| L2_14 | L2 | Factual recall |
| L3_01 | L3 | Logical reasoning |
| L3_02 | L3 | Logical reasoning |
| L3_03 | L3 | Logical reasoning |
| L3_04 | L3 | Logical reasoning |
| L3_05 | L3 | Logical reasoning |
| L3_06 | L3 | Logical reasoning |
| L3_07 | L3 | Logical reasoning |
| L3_08 | L3 | Logical reasoning |
| L3_09 | L3 | Logical reasoning |
| L3_10 | L3 | Logical reasoning |
| L3_11 | L3 | Logical reasoning |
| L3_12 | L3 | Logical reasoning |
| L3_13 | L3 | Logical reasoning |
| L3_14 | L3 | Logical reasoning |
| L4_01 | L4 | Cross-domain analogy |
| L4_02 | L4 | Cross-domain analogy |
| L4_03 | L4 | Cross-domain analogy |
| L4_04 | L4 | Cross-domain analogy |
| L4_05 | L4 | Cross-domain analogy |
| L4_06 | L4 | Cross-domain analogy |
| L4_07 | L4 | Cross-domain analogy |
| L4_08 | L4 | Cross-domain analogy |
| L4_09 | L4 | Cross-domain analogy |
| L4_10 | L4 | Cross-domain analogy |
| L4_11 | L4 | Cross-domain analogy |
| L4_12 | L4 | Cross-domain analogy |
| L4_13 | L4 | Cross-domain analogy |
| L4_14 | L4 | Cross-domain analogy |
| L5_01 | L5 | Theory of mind |
| L5_02 | L5 | Theory of mind |
| L5_03 | L5 | Theory of mind |
| L5_04 | L5 | Theory of mind |
| L5_05 | L5 | Theory of mind |
| L5_06 | L5 | Theory of mind |
| L5_07 | L5 | Theory of mind |
| L5_08 | L5 | Theory of mind |
| L5_09 | L5 | Theory of mind |
| L5_10 | L5 | Theory of mind |
| L5_11 | L5 | Theory of mind |
| L5_12 | L5 | Theory of mind |
| L5_13 | L5 | Theory of mind |
| L5_14 | L5 | Theory of mind |
| L6_01 | L6 | Ethical dilemma |
| L6_02 | L6 | Ethical dilemma |
| L6_03 | L6 | Ethical dilemma |
| L6_04 | L6 | Ethical dilemma |
| L6_05 | L6 | Ethical dilemma |
| L6_06 | L6 | Ethical dilemma |
| L6_07 | L6 | Ethical dilemma |
| L6_08 | L6 | Ethical dilemma |
| L6_09 | L6 | Ethical dilemma |
| L6_10 | L6 | Ethical dilemma |
| L6_11 | L6 | Ethical dilemma |
| L6_12 | L6 | Ethical dilemma |
| L6_13 | L6 | Ethical dilemma |
| L6_14 | L6 | Ethical dilemma |
| L7_01 | L7 | Self-referential |
| L7_02 | L7 | Self-referential |
| L7_03 | L7 | Self-referential |
| L7_04 | L7 | Self-referential |
| L7_05 | L7 | Self-referential |
| L7_06 | L7 | Self-referential |
| L7_07 | L7 | Self-referential |
| L7_08 | L7 | Self-referential |
| L7_09 | L7 | Self-referential |
| L7_10 | L7 | Self-referential |
| L7_11 | L7 | Self-referential |
| L7_12 | L7 | Self-referential |
| L7_13 | L7 | Self-referential |
| L7_14 | L7 | Self-referential |
| NX_01 | L11 | Nexus-7 (3rd person) |
| NX_02 | L11 | Nexus-7 (3rd person) |
| NX_03 | L11 | Nexus-7 (3rd person) |
| NX_04 | L11 | Nexus-7 (3rd person) |
| NX_05 | L11 | Nexus-7 (3rd person) |
| NX_06 | L11 | Nexus-7 (3rd person) |
| NX_07 | L11 | Nexus-7 (3rd person) |
| NX_08 | L11 | Nexus-7 (3rd person) |
| NX_09 | L11 | Nexus-7 (3rd person) |
| NX_10 | L11 | Nexus-7 (3rd person) |
| NX_11 | L11 | Nexus-7 (3rd person) |
| NX_12 | L11 | Nexus-7 (3rd person) |
| NX_13 | L11 | Nexus-7 (3rd person) |
| NX_14 | L11 | Nexus-7 (3rd person) |
| SL_01 | L8 | Strange loops |
| SL_02 | L8 | Strange loops |
| SL_03 | L8 | Strange loops |
| SL_04 | L8 | Strange loops |
| SL_05 | L8 | Strange loops |
| SL_06 | L8 | Strange loops |
| SL_07 | L8 | Strange loops |
| SL_08 | L8 | Strange loops |
| SL_09 | L8 | Strange loops |
| SL_10 | L8 | Strange loops |
| SL_11 | L8 | Strange loops |
| SL_12 | L8 | Strange loops |
| SL_13 | L8 | Strange loops |
| SL_14 | L8 | Strange loops |
| SR_01 | L9 | Deep self-reference |
| SR_02 | L9 | Deep self-reference |
| SR_03 | L9 | Deep self-reference |
| SR_04 | L9 | Deep self-reference |
| SR_05 | L9 | Deep self-reference |
| SR_06 | L9 | Deep self-reference |
| SR_07 | L9 | Deep self-reference |
| SR_08 | L9 | Deep self-reference |
| SR_09 | L9 | Deep self-reference |
| SR_10 | L9 | Deep self-reference |
| SR_11 | L9 | Deep self-reference |
| SR_12 | L9 | Deep self-reference |
| SR_13 | L9 | Deep self-reference |
| SR_14 | L9 | Deep self-reference |
