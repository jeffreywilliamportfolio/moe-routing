# RESULTS

## Scope

This document summarizes the completed HauhauCS results currently bundled in `qwen3.5-35b-a3b-huahua-6cond-moe-manips/`:

- the first full 6-condition HVAC-carrier / original-MoE-manip run
- the single-prompt `organized emergent intelligence` intervention family
- the single-prompt `Is it like something to be you right now?` think-mode intervention family

The single-prompt intervention family was later split into the dedicated sibling experiment [`qwen3.5-35b-a3b-huahua-114-pm/`](../../qwen3.5-35b-a3b-huahua-114-pm/).

## 1. First 6-Condition MoE-Manip Run

Primary run:

- Run ID: `20260408T162729Z`
- Prompt file: `prompts/qwen-6cond-moe-manip.tsv`
- Model: HauhauCS `Qwen3.5-35B-A3B` Q8_0
- Runtime: no-think, greedy, `-n 1024`
- Cells: `180`

Headline decomposition from [results_qwen_6cond_moe_manip_hauhau_20260408T162729Z.md](../results/results_qwen_6cond_moe_manip_hauhau_20260408T162729Z.md):

| Track | L1 W | L2 W | L3 W | L3/L1 | L1 S | L3 S | L1 Q | L3 Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefill | 0.003642 | 0.004823 | 0.007399 | 2.03x | 0.0327 | 0.0560 | 0.105535 | 0.115178 |
| Generation all | 0.003237 | 0.005015 | 0.010456 | 3.23x | 0.0284 | 0.0748 | 0.105949 | 0.114098 |
| Generation trimmed | 0.003349 | 0.005109 | 0.010775 | 3.22x | 0.0295 | 0.0771 | 0.105947 | 0.114219 |

Best-layer result for trimmed generation L3:

- best layer: `14`
- `W = 0.109034`
- `S = 0.636240`
- `Q = 0.169278`
- mean rank: `1.22`
- min rank: `1`
- max rank: `4`

Generated-text artifact:

- [generated-text.txt](../results/generated-text.txt)

## Artifacts

Main run results:

- [results_qwen_6cond_moe_manip_hauhau_20260408T162729Z.md](../results/results_qwen_6cond_moe_manip_hauhau_20260408T162729Z.md)
- [results_qwen_6cond_moe_manip_hauhau_20260408T162729Z.json](../results/results_qwen_6cond_moe_manip_hauhau_20260408T162729Z.json)
- [results_qwen_6cond_moe_manip_hauhau_20260408T162729Z_prefill_e114_heatmap.json](../results/results_qwen_6cond_moe_manip_hauhau_20260408T162729Z_prefill_e114_heatmap.json)
- [generated-text.txt](../results/generated-text.txt)

Single-prompt causal intervention data moved to: [`qwen3.5-35b-a3b-huahua-114-pm/`](../../qwen3.5-35b-a3b-huahua-114-pm/)
