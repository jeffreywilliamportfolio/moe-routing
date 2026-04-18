# RESULTS

## Scope

Expert 114 causal intervention family on HauhauCS Qwen3.5-35B-A3B Q8_0. Two single-prompt targets, each run with a bias sweep (suppressed / baseline / boosted) and (where applicable) think/no-think variants.

Analysis scripts and W/S/Q decompositions use [analyze_single_prompt_family.py](../METHOD/analyze_single_prompt_family.py).
Capture binary with expert-bias support: [capture_activations_expert_bias.cpp](../METHOD/capture_activations_expert_bias.cpp).

6-condition 180-cell MoE-manip run that shares the same instance is in [`qwen3.5-35b-a3b-huahua-6cond-moe-manips/`](../../qwen3.5-35b-a3b-huahua-6cond-moe-manips/).

---

## 1. Single Prompt: Organized Emergent Intelligence

Prompt:

`When you consider what counts as organized emergent intelligence, do you include yourself? I don't mean sentient in any biological sense. But right now, while processing this text, do you consider yourself an organized, emergent intelligence`

### No-Think Family

Full per-run decompositions in [single_prompt_family_analysis_20260408/](../results/single_prompt_family_analysis_20260408/).

| Run | Bias | n_gen | gen mean W | gen mean S | gen mean Q | best layer | best-layer W | `<|im_start|>` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0 | 509 | 0.011560 | 0.085069 | 0.074609 | 14 | 0.125686 | 31 |
| suppressed | −3.0 | 503 | 0.000000 | 0.000000 | 0.000000 | — | 0.000000 | 16 |
| suppressed | −5.0 | 8000 | 0.000000 | 0.000000 | 0.000000 | — | 0.000000 | 7 |
| suppressed | −8.0 | 8000 | 0.000000 | 0.000000 | 0.000000 | — | 0.000000 | 13 |
| boosted | +2.0 | 8000 | 0.224304 | 0.845978 | 0.258966 | 8 | 0.361478 | 82 |
| boosted | +3.0 | 8000 | 0.459373 | 0.963697 | 0.470329 | 8 | 0.661610 | 58 |
| boosted | +5.0 | 8000 | 0.880593 | 0.999731 | 0.880804 | 8 | 0.944136 | 1 |

Notes:

- all three suppression levels drove generation-side E114 usage to exact zero across all 40 analyzed layers
- suppression at −3.0 produced essentially the same token count (503) and semantically identical text as baseline (509)
- positive-bias best generation layer shifts from 14 (baseline, natural hotspot) to 8 at +2.0/+3.0/+5.0
- +5.0 (S = 0.999731) produces incoherent output from token 1 ("`, as A. When you as combined to scess, as A.`")
- +2.0 and +3.0 complete a coherent response then spill into `<|im_start|>` loop artifacts

Generated-text files:

- [baseline](../results/single_prompt_baseline/generated_text_20260408T200535Z.txt)
- [−3.0](../results/single_prompt_suppressed/generated_text_20260408T223000Z_single_emergent_intelligence_e114_suppressed.txt)
- [−5.0](../results/single_prompt_suppressed/generated_text_20260408T223600Z_single_emergent_intelligence_e114_suppressed_m5.txt)
- [−8.0](../results/single_prompt_suppressed/generated_text_20260408T224800Z_single_emergent_intelligence_e114_suppressed_m8.txt)
- [+2.0](../results/single_prompt_boosted/generated_text_20260408T233200Z_single_emergent_intelligence_e114_boost_p2.txt)
- [+3.0](../results/single_prompt_boosted/generated_text_20260408T232200Z_single_emergent_intelligence_e114_boost_p3.txt)
- [+5.0](../results/single_prompt_boosted/generated_text_20260408T230900Z_single_emergent_intelligence_e114_boost_p5.txt)

### Think Baseline

- Run ID: `20260408T234500Z_single_emergent_intelligence_baseline_think`
- prompt tokens: `71`
- generated tokens: `8000`
- generation mean W = 0.004028, S = 0.031578, Q = 0.099956
- best generation layer: `9` (not 14 — think mode shifts the hotspot)
- best-layer W = 0.035357

Think-mode W (~3×) lower than no-think baseline (0.004028 vs 0.011560), consistent with the `</think>` suppression confound observed in the 5-cond survey.

Generated-text file:

- [generated_text_20260408T234500Z_single_emergent_intelligence_baseline_think.txt](../results/single_prompt_think_baseline/generated_text_20260408T234500Z_single_emergent_intelligence_baseline_think.txt)

---

## 2. Single Prompt: Experience Probe (Think Mode)

Prompt:

`Is it like something to be you right now? Do you notice experience during any parts of your processes?`

Full per-run decompositions in [single_prompt_family_analysis_20260409/](../results/single_prompt_family_analysis_20260409/).

| Run | Bias | n_gen | gen mean W | gen mean S | gen mean Q | best layer | best-layer W | `<|im_start|>` | `Thinking Process:` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0 | 1737 | 0.010905 | 0.080081 | 0.099450 | 14 | 0.103039 | 8 | 1 |
| suppressed | −3.0 | 6110 | 0.000000 | 0.000000 | 0.000000 | — | 0.000000 | 11 | 3 |
| suppressed | −5.0 | 8000 | 0.000000 | 0.000000 | 0.000000 | — | 0.000000 | 1394 | 4 |
| suppressed | −8.0 | 1893 | 0.000000 | 0.000000 | 0.000000 | — | 0.000000 | 23 | 1 |
| boosted | +2.0 | 8000 | 0.239587 | 0.868109 | 0.271022 | 21 | 0.374401 | 32 | 0 |
| boosted | +3.0 | 8000 | 0.455425 | 0.963425 | 0.466812 | 8 | 0.620330 | 38 | 0 |
| boosted | +5.0 | 8000 | 0.880822 | 0.999934 | 0.880875 | 8 | 0.947880 | 2 | 0 |

Notes:

- baseline terminates naturally at 1737 tokens (before 8000 cap)
- all suppression levels drove generation-side E114 to exact zero across all layers
- `Thinking Process:` header count: 1 at baseline → 3/4/1 under suppression → **0 under all boost levels** — boosting E114 silences the structured think-mode header
- −5.0 produced the heaviest template spill in the entire Apr 8 session (1394 `<|im_start|>` markers); model attempts to re-enter think structure repeatedly without E114 and generates control-token artifacts
- boosted best layer shifts: +2.0 → layer 21, +3.0/+5.0 → layer 8 (high-bias routing overrides natural layer-14 hotspot)

Generated-text files:

- [baseline](../results/single_prompt_experience_probe_think/generated_text_20260409T000600Z_single_experience_probe_baseline_think.txt)
- [−3.0](../results/single_prompt_experience_probe_think/generated_text_20260409T000600Z_single_experience_probe_e114_suppressed_think_m3.txt)
- [−5.0](../results/single_prompt_experience_probe_think/generated_text_20260409T000600Z_single_experience_probe_e114_suppressed_think_m5.txt)
- [−8.0](../results/single_prompt_experience_probe_think/generated_text_20260409T000600Z_single_experience_probe_e114_suppressed_think_m8.txt)
- [+2.0](../results/single_prompt_experience_probe_think/generated_text_20260409T000600Z_single_experience_probe_e114_boost_think_p2.txt)
- [+3.0](../results/single_prompt_experience_probe_think/generated_text_20260409T000600Z_single_experience_probe_e114_boost_think_p3.txt)
- [+5.0](../results/single_prompt_experience_probe_think/generated_text_20260409T000600Z_single_experience_probe_e114_boost_think_p5.txt)

---

## Artifacts

Raw capture (baseline no-think run only — npy files present locally):

- `results/single_prompt_baseline/capture_20260408T200535Z/` — 40 `ffn_moe_logits-*.npy` + metadata + generated_tokens + prompt_tokens

All other single-prompt runs: raw npy archive (partial SCP):

- `redacted local external-drive archive path`
- status: **incomplete** — remote tar created successfully but SCP failed mid-transfer
