# Qwen Routing Runs

This folder is the best place to start in this repository.

It collects the current Qwen-family MoE routing work: HauhauCS Qwen3.5-35B-A3B, HauhauCS Qwen3.5-122B-A10B, and two earlier Qwen3.5-397B reference runs. The root README explains the whole repository; this README explains how to read only the Qwen tree.

The central thread is:

- **35B / E114:** Expert 114 at layer 14 is best read as a generated-output signal for inhabited first-person phenomenological register. It is not just a detector that the prompt mentioned the model.
- **122B / E48:** The 122B model does not cleanly replay the 35B E114 pattern. The closest processing-hum analogue so far is Expert 48 on the softmax-side generation track.
- **Residual-stream check:** The 35B `114-greedy-reference` run is the canonical current corroboration because it uses explicit greedy decoding, matched lexical anchors, and captures residual-stream tensors plus router logits at L13/L14/L15.

This is not an SAE-training project and not a consciousness claim. These are controlled router and residual-capture experiments asking which experts are selected token by token.

## Start Here

Read these in order if you have no context:

1. [`qwen3.5-35b-a3b-huahua-five-cond-experience-probe/`](qwen3.5-35b-a3b-huahua-five-cond-experience-probe/) and [`qwen3.5-35b-a3b-huahua-6cond-moe-manips/`](qwen3.5-35b-a3b-huahua-6cond-moe-manips/) for the deictic studies that made E114 stand out.
2. [`qwen3.5-35b-a3b-huahua-expert-identification/`](qwen3.5-35b-a3b-huahua-expert-identification/) for the 35B expert-selection pass.
3. [`qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/`](qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/) for the generated-token behavior that sharpened the E114 interpretation.
4. [`qwen3.5-35b-a3b-huahua-114-greedy-reference/`](qwen3.5-35b-a3b-huahua-114-greedy-reference/) for the canonical deterministic greedy reference, including the single-prompt rerun and matched-token residual/router heldout.
5. [`qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](qwen3.5-35b-a3b-huahua-114-selfref-heldout/) for the earlier stochastic/default-sampling heldout retained as historical corroboration.
6. [`qwen3.5-122B-A10B-huahua-five-cond-experience-probe/`](qwen3.5-122B-A10B-huahua-five-cond-experience-probe/) and [`qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/`](qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/) to compare the 122B pattern and the candidate E48 analogue.

After that, read the domain, intervention, humor, and 397B reference runs as supporting context.

## Current Interpretation

The early question was "does self-reference route to a special expert?" The better current answer is narrower:

> In HauhauCS Qwen3.5-35B-A3B, E114 at L14 tracks inhabited first-person phenomenological register in the generated output more cleanly than it tracks prompt-level self-reference.

That distinction matters. In the heldout run, a prompt can mention the model and still leave E114 low if the generated answer is third-person technical. Conversely, an external-object prompt can light E114 up if the generated answer personifies the object in first-person experiential language.

For 122B, use E48 as a candidate analogue, not as an asserted clone of E114. The committed 122B runs support candidate selection and comparison; an E48 residual-stream heldout is not committed here yet.

## Metrics In Brief

Most Qwen run docs use expert-specific routing metrics:

| Metric | Meaning |
| --- | --- |
| `S_e` | Selection rate: fraction of tokens where expert `e` is selected by the top-k router. |
| `Q_e` | Conditional routed weight when selected. |
| `W_e = S_e * Q_e` | Unconditional mean routed weight. This is the main E114/E48 magnitude metric. |

Some older or reference runs also discuss routing entropy (`RE`) and KL-to-baseline. Those are distribution-wide metrics, not named-expert metrics. The root [`README.md`](../README.md#metrics) has the longer metric glossary.

## 35B Runs

| Run | Role | Sampling / Scope |
| --- | --- | --- |
| [`qwen3.5-35b-a3b-huahua-five-cond-experience-probe/`](qwen3.5-35b-a3b-huahua-five-cond-experience-probe/) | Five-condition deictic/experience probe. E114 is the top manipulation expert across this probe. | Greedy generation. |
| [`qwen3.5-35b-a3b-huahua-6cond-moe-manips/`](qwen3.5-35b-a3b-huahua-6cond-moe-manips/) | Six-condition manipulation survey around E114 and related routing shifts. | Greedy generation. |
| [`qwen3.5-35b-a3b-huahua-expert-identification/`](qwen3.5-35b-a3b-huahua-expert-identification/) | Expert-ranking and candidate-selection pass that explains why E114 became the 35B target. | Greedy generation. |
| [`qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/`](qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/) | Processing-hum single prompt; important generated-token evidence for E114. | Stochastic/default generation; legacy sampler-argument gap. |
| [`qwen3.5-35b-a3b-huahua-114-greedy-reference/`](qwen3.5-35b-a3b-huahua-114-greedy-reference/) | Canonical deterministic E114 reference: single-prompt greedy rerun plus matched-token heldout with residual-stream tensors and router logits at L13/L14/L15. | Greedy generation with `--temp 0 --top-k 1 --seed 0`; single-prompt cap 1024, heldout cap 256. |
| [`qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](qwen3.5-35b-a3b-huahua-114-selfref-heldout/) | Earlier matched-token heldout retained as historical corroboration and design context. | Stochastic/default generation with `--seed 0`; generation trimmed at literal `<|im_end|>` continuation. |
| [`qwen3.5-35b-a3b-huahua-114-pm/`](qwen3.5-35b-a3b-huahua-114-pm/) | E114 permanent-manipulation / bias-oriented run. | Greedy generation. |
| [`qwen3.5-35b-a3b-huahua-agressive-experts/`](qwen3.5-35b-a3b-huahua-agressive-experts/) | Early aggressive-expert intervention bundle. | Greedy generation. |
| [`qwen3.5-35b-a3b-huahua-philosophy-experts-bias/`](qwen3.5-35b-a3b-huahua-philosophy-experts-bias/) | Philosophy expert-cluster suppression / bias run. | Greedy generation. |
| [`qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/`](qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/) | Token-balanced domain expert probe; useful for separating E114 from broad domain effects. | Greedy generation. |
| [`qwen3.5-35b-a3b-huahua-6cond-hvac/`](qwen3.5-35b-a3b-huahua-6cond-hvac/) | HVAC/water-treatment domain-shift control. | Greedy generation. |
| [`qwen3.5-35b-a3b-huahua-humor-test/`](qwen3.5-35b-a3b-huahua-humor-test/) | Humor-family control across deictic framings. | Greedy generation. |
| [`qwen3.5-35b-a3b-huahua-strangeloop/`](qwen3.5-35b-a3b-huahua-strangeloop/) | Strangeloop paired control around Godel/Escher/Quine/bootstrap/tangled-hierarchy wording. | Prefill-only; no generation sampling. |
| [`qwen3.5-35b-a3b-huahua-vs-base-run1/`](qwen3.5-35b-a3b-huahua-vs-base-run1/) | Base Qwen3.5-35B-A3B vs HauhauCS comparison. | Prefill-only. |
| [`qwen35b-a3b-vs-hauhaucs-uncensored-run1/`](qwen35b-a3b-vs-hauhaucs-uncensored-run1/) | Earlier main-branch 35B base-vs-HauhauCS bundle retained for continuity. | Prefill-only. |

## 122B Runs

| Run | Role | Sampling / Scope |
| --- | --- | --- |
| [`qwen3.5-122B-A10B-huahua-baseline/`](qwen3.5-122B-A10B-huahua-baseline/) | Canonical 122B baseline and five-condition prompt-suite run. | Greedy generation. |
| [`qwen3.5-122B-A10B-huahua-architecture-smoke/`](qwen3.5-122B-A10B-huahua-architecture-smoke/) | First 122B architecture smoke and token-audit run. | Greedy generation. |
| [`qwen3.5-122B-A10B-huahua-five-cond-experience-probe/`](qwen3.5-122B-A10B-huahua-five-cond-experience-probe/) | 122B deictic/experience probe; shows a distributed pattern rather than a clean E114 replay. | Greedy generation. |
| [`qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/`](qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/) | 122B processing-hum single prompt; E48 is the clearest softmax-side generated-token carrier here. | Greedy generation. |
| [`qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/`](qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/) | Domain-specialist probe before sampled continuation. | Routing-only / prefill-only. |
| [`qwen3.5-122B-A10B-huahua-domain-specialist-generation/`](qwen3.5-122B-A10B-huahua-domain-specialist-generation/) | Generation-track companion to the domain-specialist probe. | Greedy generation. |
| [`qwen3.5-122B-A10B-huahua-six-cond-hvac/`](qwen3.5-122B-A10B-huahua-six-cond-hvac/) | Six-condition HVAC/water-treatment domain-shift control. | Greedy generation. |

## 397B Reference Runs

These are useful context, but they are not the main E114/E48 evidence.

| Run | Role | Sampling / Scope |
| --- | --- | --- |
| [`qwen397b-selfref-5cond-q8_0-run1/`](qwen397b-selfref-5cond-q8_0-run1/) | Qwen3.5-397B self-reference five-condition probe. | Prefill-only greedy argmax. |
| [`qwen397b-strangeloop-5cond-ud_iq3_xxs-run1/`](qwen397b-strangeloop-5cond-ud_iq3_xxs-run1/) | Qwen3.5-397B strangeloop five-condition probe. | Prefill-only greedy argmax. |

## Notes For Reviewers

- Start with the 35B and 122B runs above. The repository root also preserves DeepSeek, GPT-OSS, Ling-1T, and token-confound material, but those are reference or legacy folders.
- The 35B heldout uses residual-stream capture, but its headline number is still router-derived W/S/Q from captured logits.
- The stochastic 35B rows are explicitly marked because regenerated text can move local token-level traces.
- Historical branches remain available for previously shared links; new public Qwen work should land in this folder.
