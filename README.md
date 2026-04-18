# moe-routing

Controlled MoE routing experiments across open-weight mixture-of-experts models, with the current HauhauCS Qwen3.5-35B thread focused on verbalizing Expert 114.

The updated working interpretation is that E114 at layer 14 is a generated-output signal for phenomenological / mental-state register. It fires densely when the model is producing first-person experiential language, inner-state predicates, or agency-bearing personification; it does not simply track whether the prompt asks about the model, and it is not explained by matched lexical anchor tokens alone.

Most bundles here are router-focused prompt probes measuring expert selection, routing entropy, and divergence under small wording changes. The strongest 35B corroborating run also captures residual-stream tensors at L13/L14/L15, verifies the `qwen35moe` `attn_post_norm-14` router tap, and reports router-derived W/S/Q on trimmed generation tokens.

This is not an SAE-training repository. The E114 work starts from an already identified routed expert and asks what that expert fires on.

The current canonical public tree is `main`. Older branch snapshots remain available for provenance and previously shared review links, but new public work should land on `main` in the model-oriented layout below.

## Current Layout

| Path | Purpose |
| --- | --- |
| [`qwen/`](qwen/) | Qwen and HauhauCS Qwen experiment bundles, including 35B, 122B, and 397B runs. |
| [`deepseek/`](deepseek/) | DeepSeek V3.1 routing bundles. |
| [`gptoss/`](gptoss/) | GPT-OSS-120B routing bundles. |
| [`ling/`](ling/) | Ling-1T validation documentation. |
| [`token-confound-archive/`](token-confound-archive/) | Preserved archive documenting the token-position confound and corrected interpretation. |

## Main Findings

- Router measurements are sensitive to prompt length and token position. Older all-token averages can measure position structure instead of the intended conceptual variable.
- For HauhauCS Qwen3.5-35B, Expert 114 is best treated as a layer-14 generated-output signal for phenomenological or mental-state register, not as a narrow detector for the prompt asking about the model.
- The matched-token residual-capture heldout tests the lexical-null directly: 10 fire and 10 nofire prompts share the same anchor tokens, but trimmed-generation W114 at L14 separates 0.0675 vs. 0.0031, a 21.7x ratio with Cohen's d 2.94 and no range overlap.
- The two boundary cases drive the label refinement. F07 asked about the model but generated third-person technical exposition and E114 stayed low; N10 asked about a wool sweater but generated first-person personification and E114 fired in clusters.
- The heldout captured residual-stream tensors plus router logits and verified the `qwen35moe` `attn_post_norm-14` router tap. The published headline metric is router-derived W/S/Q, not an SAE result or a residual-only labeler result.
- The 122B HauhauCS runs do not yet show a direct one-expert carryover of 35B E114. The clearest 122B processing-hum analogue so far is E48 on the softmax-side generation track; no E48 residual-stream heldout is committed yet.
- The 122B HauhauCS runs are split into separate top-level Qwen bundles so each probe can be inspected independently.

## Starting New

Read `main` by study lineage, not by commit chronology. The historical branches preserve how the work accumulated, but the current public route is:

1. Start with deictic prompt probes for the 35B and 122B HauhauCS models. These are the wording-change studies that show routing responds to self/other/human/object framing and related prompt manipulations.
2. Move to the Expert 114 bias/intervention work on 35B. Those bundles test routing and generation behavior when E114 or neighboring expert clusters are directly perturbed.
3. Then read the expert-selection bundles. These identify which experts dominate by domain, prompt family, prefill/generation phase, and layer family, rather than assuming the deictic result names the whole mechanism.
4. Then read the processing-hum single-prompt probes. These localize the candidate expert signal to generated tokens and distinguish prompt type from generated phenomenological language.
5. Finish with the residual-stream heldout. This is the corroborating 35B E114 run: matched lexical anchors, residual tensors at L13/L14/L15, verified router tap, and router-derived W/S/Q on trimmed generation.

| Step | 35B E114 Thread | 122B / E48 Thread | What This Step Establishes |
| --- | --- | --- | --- |
| Deictic baselines | [`qwen3.5-35b-a3b-huahua-6cond-moe-manips/`](qwen/qwen3.5-35b-a3b-huahua-6cond-moe-manips/), [`qwen3.5-35b-a3b-huahua-five-cond-experience-probe/`](qwen/qwen3.5-35b-a3b-huahua-five-cond-experience-probe/), [`qwen3.5-35b-a3b-huahua-6cond-hvac/`](qwen/qwen3.5-35b-a3b-huahua-6cond-hvac/), [`qwen3.5-35b-a3b-huahua-humor-test/`](qwen/qwen3.5-35b-a3b-huahua-humor-test/) | [`qwen3.5-122B-A10B-huahua-baseline/`](qwen/qwen3.5-122B-A10B-huahua-baseline/), [`qwen3.5-122B-A10B-huahua-five-cond-experience-probe/`](qwen/qwen3.5-122B-A10B-huahua-five-cond-experience-probe/), [`qwen3.5-122B-A10B-huahua-six-cond-hvac/`](qwen/qwen3.5-122B-A10B-huahua-six-cond-hvac/) | Finds routing sensitivity under small deictic and domain shifts. For 35B this makes E114 salient; for 122B it shows the effect is distributed rather than a clean E114 carryover. |
| Bias and intervention | [`qwen3.5-35b-a3b-huahua-114-pm/`](qwen/qwen3.5-35b-a3b-huahua-114-pm/), [`qwen3.5-35b-a3b-huahua-agressive-experts/`](qwen/qwen3.5-35b-a3b-huahua-agressive-experts/), [`qwen3.5-35b-a3b-huahua-philosophy-experts-bias/`](qwen/qwen3.5-35b-a3b-huahua-philosophy-experts-bias/) | No dedicated E48 bias repo is committed yet. Use the 122B routing and generation probes below as candidate-selection evidence. | Tests whether directly moving the candidate expert or cluster changes routing/generation behavior. |
| Expert selection | [`qwen3.5-35b-a3b-huahua-expert-identification/`](qwen/qwen3.5-35b-a3b-huahua-expert-identification/), [`qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/`](qwen/qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/) | [`qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/`](qwen/qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/), [`qwen3.5-122B-A10B-huahua-domain-specialist-generation/`](qwen/qwen3.5-122B-A10B-huahua-domain-specialist-generation/) | Separates deictic effects from broader expert rankings across domains, phases, and layers. |
| Processing-hum single prompt | [`qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/) | [`qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/) | Localizes the candidate signal to generated phenomenological language. In 35B this supports the E114 label; in 122B, E48 is the clearest softmax-side generated-token carrier for this prompt. |
| Residual-stream corroboration | [`qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/) | Not yet run/committed for E48. | Directly tests the lexical-null with matched anchor tokens and residual/router capture. The 35B heldout refines E114 from "self-reference" toward generated-output phenomenological / mental-state register. |

## Key Qwen Runs

| Bundle | Model / Scope | Evidence Type | Notes |
| --- | --- | --- | --- |
| [`qwen/qwen3.5-122B-A10B-huahua-baseline/`](qwen/qwen3.5-122B-A10B-huahua-baseline/) | HauhauCS Qwen3.5-122B-A10B baseline/reference surface | Router capture | Canonical 122B baseline and 5-condition prompt-suite run. |
| [`qwen/qwen3.5-122B-A10B-huahua-architecture-smoke/`](qwen/qwen3.5-122B-A10B-huahua-architecture-smoke/) | 122B architecture smoke | Router capture | First 122B single-prompt architecture smoke run and token audit. |
| [`qwen/qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/`](qwen/qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/) | 122B domain-specialist probe | Router-only prefill | Routing-only domain-specialist probe. |
| [`qwen/qwen3.5-122B-A10B-huahua-domain-specialist-generation/`](qwen/qwen3.5-122B-A10B-huahua-domain-specialist-generation/) | 122B domain-specialist probe | Prefill plus generation | Generation-track domain-specialist probe. |
| [`qwen/qwen3.5-122B-A10B-huahua-five-cond-experience-probe/`](qwen/qwen3.5-122B-A10B-huahua-five-cond-experience-probe/) | 122B five-condition experience probe | Router capture plus per-token summaries | Split out from the former nested followups folder. |
| [`qwen/qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/) | 122B single-prompt processing-hum probe | Router capture plus per-token summaries | 122B analogue of the processing-hum probe. |
| [`qwen/qwen3.5-122B-A10B-huahua-six-cond-hvac/`](qwen/qwen3.5-122B-A10B-huahua-six-cond-hvac/) | 122B six-condition HVAC/water-treatment probe | Router capture | Domain-shifted six-condition probe. |
| [`qwen/qwen3.5-35b-a3b-huahua-expert-identification/`](qwen/qwen3.5-35b-a3b-huahua-expert-identification/) | HauhauCS Qwen3.5-35B Expert 114 identification | Router analysis | Expert-identification provenance for the 35B E114 thread. |
| [`qwen/qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/) | 35B processing-hum probe | Router analysis | E114 is strongest around phenomenological claims in generated output. |
| [`qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/) | 35B matched-token E114 heldout | Residual-stream tensors plus router logits | Corroborating residual-stream run. Trimmed-generation W114 at L14 separates fire vs. nofire prompts by 21.7x with no range overlap. |
| [`qwen/qwen3.5-35b-a3b-huahua-five-cond-experience-probe/`](qwen/qwen3.5-35b-a3b-huahua-five-cond-experience-probe/) | 35B five-condition experience probe | Router analysis | E114 is the top manipulation expert across the 15-prompt probe. |
| [`qwen/qwen3.5-35b-a3b-huahua-6cond-moe-manips/`](qwen/qwen3.5-35b-a3b-huahua-6cond-moe-manips/) | 35B six-condition MoE manipulation survey | Router analysis | Broader E114 manipulation survey with prefill heatmap artifacts. |
| [`qwen/qwen3.5-35b-a3b-huahua-6cond-hvac/`](qwen/qwen3.5-35b-a3b-huahua-6cond-hvac/) | 35B HVAC/water-treatment domain shift | Router analysis | Tests whether E114 is only an ML-topic specialist. |
| [`qwen/qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/`](qwen/qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/) | 35B domain expert probe | Router analysis | Token-balanced domain chunks, including generation/prefill divergence. |
| [`qwen/qwen3.5-35b-a3b-huahua-strangeloop/`](qwen/qwen3.5-35b-a3b-huahua-strangeloop/) | 35B strangeloop paired control | Router analysis | Paired control around Godel/Escher/Quine/bootstrap/tangled-hierarchy prompts. |
| [`qwen/qwen3.5-35b-a3b-huahua-114-pm/`](qwen/qwen3.5-35b-a3b-huahua-114-pm/) | 35B Expert 114 permanent manipulation | Router intervention | Causal manipulation-oriented E114 bundle. |
| [`qwen/qwen3.5-35b-a3b-huahua-agressive-experts/`](qwen/qwen3.5-35b-a3b-huahua-agressive-experts/) | 35B HauhauCS aggressive experts | Router intervention | Researcher-facing entrypoint for early intervention work. |
| [`qwen/qwen3.5-35b-a3b-huahua-philosophy-experts-bias/`](qwen/qwen3.5-35b-a3b-huahua-philosophy-experts-bias/) | 35B philosophy expert cluster suppression | Router intervention | Bias/suppression-oriented cluster analysis. |
| [`qwen/qwen3.5-35b-a3b-huahua-humor-test/`](qwen/qwen3.5-35b-a3b-huahua-humor-test/) | 35B humor control | Router analysis | Minimal prompt-family control across five deictic framings. |
| [`qwen/qwen3.5-35b-a3b-huahua-vs-base-run1/`](qwen/qwen3.5-35b-a3b-huahua-vs-base-run1/) | 35B base vs HauhauCS comparison | Router analysis | Prefill-only comparison between base Qwen3.5-35B-A3B and HauhauCS. |
| [`qwen/qwen35b-a3b-vs-hauhaucs-uncensored-run1/`](qwen/qwen35b-a3b-vs-hauhaucs-uncensored-run1/) | Earlier 35B base vs HauhauCS bundle | Router analysis | Existing main-branch Qwen bundle retained for continuity. |
| [`qwen/qwen397b-selfref-5cond-q8_0-run1/`](qwen/qwen397b-selfref-5cond-q8_0-run1/) | Qwen3.5-397B self-reference probe | Router analysis | Reviewed 397B self-referential five-condition run. |
| [`qwen/qwen397b-strangeloop-5cond-ud_iq3_xxs-run1/`](qwen/qwen397b-strangeloop-5cond-ud_iq3_xxs-run1/) | Qwen3.5-397B strangeloop probe | Router analysis | Reviewed 397B strangeloop five-condition run. |

## Residual-Stream Corroboration

| Bundle | Evidence Type | Notes |
| --- | --- | --- |
| [`qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/) | Residual-stream tensors plus router logits | Matched-token heldout control. Same anchor tokens across fire/nofire prompts; trimmed-generation W114 at L14 separates 0.0675 vs. 0.0031, a 21.7x ratio with Cohen's d 2.94 and no range overlap. Boundary cases refine the label toward generated-output phenomenological / mental-state register. |

## Other Model Families

| Bundle | Model / Scope | Notes |
| --- | --- | --- |
| [`deepseek/ds31-5cond-1/`](deepseek/ds31-5cond-1/) | DeepSeek V3.1 five-condition experiment | Reproducible DeepSeek routing bundle. |
| [`gptoss/gptoss-5cond-1/`](gptoss/gptoss-5cond-1/) | GPT-OSS-120B self-referential five-condition run | Reviewed valid GPT-OSS five-condition artifact. |
| [`gptoss/gptoss-strangeloop-paired-1/`](gptoss/gptoss-strangeloop-paired-1/) | GPT-OSS-120B strangeloop paired control | Reviewed GPT-OSS strangeloop control artifact. |
| [`ling/ling1t-validation-subset-documented/`](ling/ling1t-validation-subset-documented/) | Ling-1T validation subset | Preserved validation documentation bundle. |

## Frozen Historical Branches

These branches are intentionally left static so previously shared review links remain valid:

| Branch | Status | Notes |
| --- | --- | --- |
| [`qwen-hauhau-5cond-smoke-only`](https://github.com/jeffreywilliamportfolio/moe-routing/tree/qwen-hauhau-5cond-smoke-only) | Frozen historical snapshot | Older smoke-only tree with a different layout and raw tensor conventions. It is preserved as a branch rather than duplicated into `main`. |
| [`qwen3.5-35b-a3b-huahua-experiments-2026-04-10`](https://github.com/jeffreywilliamportfolio/moe-routing/tree/qwen3.5-35b-a3b-huahua-experiments-2026-04-10) | Frozen historical snapshot | Source branch for the 35B bundles now copied into `main` under `qwen/`. |

## Data Policy

Main-line experiment bundles should track prompts, scripts, metadata, generated text, plots, small JSON/TSV/Markdown results, and enough provenance to inspect or reproduce a run.

Do not add raw tensor dumps, compressed archives, Python caches, local environments, model weights, credentials, hostnames, ports, private keys, API tokens, or `.env` contents to `main`. Some frozen historical branches may preserve older conventions for provenance; do not treat them as the current import policy.

## License

Code and documentation in this repository are released under the MIT License. See [`LICENSE`](LICENSE).
