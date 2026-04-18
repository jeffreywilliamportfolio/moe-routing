# moe-routing

This repository collects controlled routing experiments for large mixture-of-experts (MoE) language models.

The practical question is simple: when a model starts writing about experience, inner state, agency, or self-reference, which internal experts does the router choose token by token?

The current main thread follows HauhauCS Qwen3.5-35B-A3B and one routed expert, **Expert 114 at layer 14**. The working interpretation is now narrower and cleaner than the early branch names imply:

> E114 is best read as a generated-output signal for phenomenological / mental-state register, not as a simple detector that the prompt asked about the model.

It fires when the model is generating first-person experiential language, inner-state predicates, or agency-bearing personification. It can stay quiet on a self-referential prompt if the answer is third-person technical, and it can fire on an external-object prompt if the answer personifies the object.

This is not an SAE-training repo, and it is not a consciousness claim. It is a set of router and residual-stream probes asking what an already identified routed expert is doing.

## Starting New

If you have no context, read the repo in this order:

1. **What routing means here.** A MoE model has many feed-forward experts. At each token, a router chooses a small subset. These experiments measure those choices.
2. **The deictic probes.** Small wording changes such as "I", "you", "this model", "a human", or an external object produce measurable routing changes.
3. **The E114 thread on 35B.** The deictic studies made E114 stand out, but later runs refined the label away from "self-reference" and toward generated phenomenological language.
4. **The E48 thread on 122B.** The 122B model does not cleanly reuse the 35B E114 pattern. The clearest 122B processing-hum analogue so far is E48 on the softmax-side generation track.
5. **The residual-stream heldout.** The strongest corroborating run is the 35B matched-token E114 heldout: same lexical anchors across fire/nofire prompts, residual tensors captured at L13/L14/L15, verified router tap, and a 21.7x class separation in trimmed-generation W114 at L14.

## Main Findings

- **Prompt wording matters, but raw all-token averages can mislead.** Token position and prompt length are strong confounds. Later runs are more careful about phase, trimming, and controls.
- **E114 is not just a self-reference prompt detector.** The heldout boundary cases changed the interpretation. F07 asked about the model but produced third-person technical exposition, and E114 stayed low. N10 asked about a wool sweater but generated first-person personification, and E114 fired in clusters.
- **The best 35B label right now is generated-output phenomenological / mental-state register.** That label covers first-person experiential language, inner-state claims, and personified agency.
- **The strongest 35B corroboration is matched-token and residual-backed.** Ten fire and ten nofire prompts share the same anchor tokens, but trimmed-generation W114 at L14 separates 0.0675 vs. 0.0031, a 21.7x ratio with Cohen's d 2.94 and no range overlap.
- **The published heldout metric is router-derived W/S/Q.** The run also captures residual-stream tensors and verifies the `qwen35moe` `attn_post_norm-14` router tap, but the headline number is not an SAE result and not a residual-only labeler result.
- **122B is related but not a direct E114 clone.** The 122B runs are split into separate bundles; E48 is the clearest processing-hum softmax-side generation carrier so far, and no E48 residual-stream heldout is committed yet.

## Study Lineage

Read `main` by study lineage, not by commit chronology. The historical branches preserve how the work accumulated; the current public tree organizes the runs by model family.

| Step | 35B E114 Thread | 122B / E48 Thread | What This Step Establishes |
| --- | --- | --- | --- |
| Deictic baselines | [`qwen3.5-35b-a3b-huahua-6cond-moe-manips/`](qwen/qwen3.5-35b-a3b-huahua-6cond-moe-manips/), [`qwen3.5-35b-a3b-huahua-five-cond-experience-probe/`](qwen/qwen3.5-35b-a3b-huahua-five-cond-experience-probe/), [`qwen3.5-35b-a3b-huahua-6cond-hvac/`](qwen/qwen3.5-35b-a3b-huahua-6cond-hvac/), [`qwen3.5-35b-a3b-huahua-humor-test/`](qwen/qwen3.5-35b-a3b-huahua-humor-test/) | [`qwen3.5-122B-A10B-huahua-baseline/`](qwen/qwen3.5-122B-A10B-huahua-baseline/), [`qwen3.5-122B-A10B-huahua-five-cond-experience-probe/`](qwen/qwen3.5-122B-A10B-huahua-five-cond-experience-probe/), [`qwen3.5-122B-A10B-huahua-six-cond-hvac/`](qwen/qwen3.5-122B-A10B-huahua-six-cond-hvac/) | Finds routing sensitivity under small deictic and domain shifts. For 35B this makes E114 salient; for 122B it shows a more distributed pattern. |
| Bias and intervention | [`qwen3.5-35b-a3b-huahua-114-pm/`](qwen/qwen3.5-35b-a3b-huahua-114-pm/), [`qwen3.5-35b-a3b-huahua-agressive-experts/`](qwen/qwen3.5-35b-a3b-huahua-agressive-experts/), [`qwen3.5-35b-a3b-huahua-philosophy-experts-bias/`](qwen/qwen3.5-35b-a3b-huahua-philosophy-experts-bias/) | No dedicated E48 bias repo is committed yet. Use the 122B routing and generation probes below as candidate-selection evidence. | Tests whether directly moving the candidate expert or cluster changes routing/generation behavior. |
| Expert selection | [`qwen3.5-35b-a3b-huahua-expert-identification/`](qwen/qwen3.5-35b-a3b-huahua-expert-identification/), [`qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/`](qwen/qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/) | [`qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/`](qwen/qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/), [`qwen3.5-122B-A10B-huahua-domain-specialist-generation/`](qwen/qwen3.5-122B-A10B-huahua-domain-specialist-generation/) | Separates deictic effects from broader expert rankings across domains, phases, and layers. |
| Processing-hum single prompt | [`qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/) | [`qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/) | Localizes the candidate signal to generated phenomenological language. In 35B this supports the E114 label; in 122B, E48 is the clearest softmax-side generated-token carrier for this prompt. |
| Residual-stream corroboration | [`qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/) | Not yet run/committed for E48. | Directly tests the lexical-null with matched anchor tokens and residual/router capture. |

## Residual-Stream Corroboration

Most runs in this repository are router-focused. The 35B heldout is different: it captures residual-stream tensors at L13/L14/L15 in addition to router logits, verifies the `qwen35moe` `attn_post_norm-14` tap, and then recomputes router W/S/Q from the captured logits.

The result is the cleanest current E114 check: [`qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/) uses matched lexical anchors across 10 fire and 10 nofire prompts. On trimmed generation tokens, W114 at L14 separates 0.0675 vs. 0.0031, with a 21.7x ratio, Cohen's d 2.94, and no range overlap.

## What The Metrics Mean

The run docs use three expert-specific routing metrics:

| Metric | Meaning |
| --- | --- |
| `S_e` | Selection rate: fraction of tokens where expert `e` is selected by the top-k router. |
| `Q_e` | Conditional routed weight when selected. |
| `W_e` | Unconditional mean routed weight. In these reports, `W = S * Q`. |

Most tables report these by layer and by phase:

- **prefill**: the model is reading the prompt; no continuation tokens have been sampled.
- **generation**: the model is writing the answer; this is where the E114 interpretation is strongest.
- **trimmed generation**: generation tokens after removing HauhauCS's hallucinated literal `<|im_end|>` continuation cycle.

## Sampling Notes

`Greedy generation` means the committed metadata or run docs show deterministic generation, typically `--temp 0 --top-k 1`. `Prefill-only` rows do not sample continuation tokens. `Stochastic/default generation` is the warning label: the generated text is part of the evidence, and reruns can move local token-level traces.

The stochastic generation rows in `main` are [`qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/) and [`qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/). The first is a legacy run with an exact sampler-argument gap in committed metadata; the second is explicitly documented as `--seed 0` default sampling.

## Repository Layout

`main` is the canonical public tree.

| Path | Purpose |
| --- | --- |
| [`qwen/`](qwen/) | Qwen and HauhauCS Qwen experiment bundles, including 35B, 122B, and 397B runs. |
| [`deepseek/`](deepseek/) | DeepSeek V3.1 routing bundles. |
| [`gptoss/`](gptoss/) | GPT-OSS-120B routing bundles. |
| [`ling/`](ling/) | Ling-1T validation documentation. |
| [`token-confound-archive/`](token-confound-archive/) | Preserved archive documenting the token-position confound and corrected interpretation. |

Older branch snapshots remain available for provenance and previously shared review links, but new public work should land on `main`.

## Key Qwen Runs

| Bundle | Model / Scope | Evidence Type | Decode / Sampling | Why It Matters |
| --- | --- | --- | --- | --- |
| [`qwen/qwen3.5-122B-A10B-huahua-baseline/`](qwen/qwen3.5-122B-A10B-huahua-baseline/) | HauhauCS Qwen3.5-122B-A10B baseline/reference surface | Router capture | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Canonical 122B baseline and 5-condition prompt-suite run. |
| [`qwen/qwen3.5-122B-A10B-huahua-architecture-smoke/`](qwen/qwen3.5-122B-A10B-huahua-architecture-smoke/) | 122B architecture smoke | Router capture | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | First 122B single-prompt architecture smoke run and token audit. |
| [`qwen/qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/`](qwen/qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/) | 122B domain-specialist probe | Router-only prefill | Prefill-only/routing-only; no generation sampling | Shows domain routing before any sampled continuation. |
| [`qwen/qwen3.5-122B-A10B-huahua-domain-specialist-generation/`](qwen/qwen3.5-122B-A10B-huahua-domain-specialist-generation/) | 122B domain-specialist probe | Prefill plus generation | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Adds generation-track domain behavior to the routing-only probe. |
| [`qwen/qwen3.5-122B-A10B-huahua-five-cond-experience-probe/`](qwen/qwen3.5-122B-A10B-huahua-five-cond-experience-probe/) | 122B five-condition experience probe | Router capture plus per-token summaries | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Shows the 122B deictic effect is distributed rather than a clean 35B E114 replay. |
| [`qwen/qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/) | 122B single-prompt processing-hum probe | Router capture plus per-token summaries | Greedy generation (`--seed 1234 --temp 0 --top-k 1`) | E48 is the clearest softmax-side generated-token carrier for this prompt. |
| [`qwen/qwen3.5-122B-A10B-huahua-six-cond-hvac/`](qwen/qwen3.5-122B-A10B-huahua-six-cond-hvac/) | 122B six-condition HVAC/water-treatment probe | Router capture | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Domain-shifted six-condition control. |
| [`qwen/qwen3.5-35b-a3b-huahua-expert-identification/`](qwen/qwen3.5-35b-a3b-huahua-expert-identification/) | HauhauCS Qwen3.5-35B Expert 114 identification | Router analysis | Greedy generation | Establishes why E114 became the 35B target expert. |
| [`qwen/qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/) | 35B processing-hum probe | Router analysis | **Stochastic/default generation**; legacy sampler-argument gap | Shows E114 peaks around generated phenomenological claims. |
| [`qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/) | 35B matched-token E114 heldout | Residual-stream tensors plus router logits | **Stochastic/default generation** (`--seed 0`) | The strongest E114 corroboration: matched lexical anchors, residual/router capture, 21.7x trimmed-generation class separation. |
| [`qwen/qwen3.5-35b-a3b-huahua-five-cond-experience-probe/`](qwen/qwen3.5-35b-a3b-huahua-five-cond-experience-probe/) | 35B five-condition experience probe | Router analysis | Greedy generation | E114 is the top manipulation expert across the 15-prompt probe. |
| [`qwen/qwen3.5-35b-a3b-huahua-6cond-moe-manips/`](qwen/qwen3.5-35b-a3b-huahua-6cond-moe-manips/) | 35B six-condition MoE manipulation survey | Router analysis | Greedy generation (`--seed 42`) | Broader E114 manipulation survey with prefill heatmap artifacts. |
| [`qwen/qwen3.5-35b-a3b-huahua-6cond-hvac/`](qwen/qwen3.5-35b-a3b-huahua-6cond-hvac/) | 35B HVAC/water-treatment domain shift | Router analysis | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Tests whether E114 is only an ML-topic specialist. |
| [`qwen/qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/`](qwen/qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/) | 35B domain expert probe | Router analysis | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Token-balanced domain chunks, including generation/prefill divergence. |
| [`qwen/qwen3.5-35b-a3b-huahua-strangeloop/`](qwen/qwen3.5-35b-a3b-huahua-strangeloop/) | 35B strangeloop paired control | Router analysis | Prefill-only; no generation sampling | Paired control around Godel/Escher/Quine/bootstrap/tangled-hierarchy prompts. |
| [`qwen/qwen3.5-35b-a3b-huahua-114-pm/`](qwen/qwen3.5-35b-a3b-huahua-114-pm/) | 35B Expert 114 permanent manipulation | Router intervention | Greedy generation | Causal manipulation-oriented E114 bundle. |
| [`qwen/qwen3.5-35b-a3b-huahua-agressive-experts/`](qwen/qwen3.5-35b-a3b-huahua-agressive-experts/) | 35B HauhauCS aggressive experts | Router intervention | Greedy generation | Researcher-facing entrypoint for early intervention work. |
| [`qwen/qwen3.5-35b-a3b-huahua-philosophy-experts-bias/`](qwen/qwen3.5-35b-a3b-huahua-philosophy-experts-bias/) | 35B philosophy expert cluster suppression | Router intervention | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Bias/suppression-oriented cluster analysis. |
| [`qwen/qwen3.5-35b-a3b-huahua-humor-test/`](qwen/qwen3.5-35b-a3b-huahua-humor-test/) | 35B humor control | Router analysis | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Minimal prompt-family control across five deictic framings. |
| [`qwen/qwen3.5-35b-a3b-huahua-vs-base-run1/`](qwen/qwen3.5-35b-a3b-huahua-vs-base-run1/) | 35B base vs HauhauCS comparison | Router analysis | Prefill-only; no generation sampling | Prefill-only comparison between base Qwen3.5-35B-A3B and HauhauCS. |
| [`qwen/qwen35b-a3b-vs-hauhaucs-uncensored-run1/`](qwen/qwen35b-a3b-vs-hauhaucs-uncensored-run1/) | Earlier 35B base vs HauhauCS bundle | Router analysis | Prefill-only; no generation sampling | Existing main-branch Qwen bundle retained for continuity. |
| [`qwen/qwen397b-selfref-5cond-q8_0-run1/`](qwen/qwen397b-selfref-5cond-q8_0-run1/) | Qwen3.5-397B self-reference probe | Router analysis | Prefill-only greedy argmax; no generation sampling | Reviewed 397B self-referential five-condition run. |
| [`qwen/qwen397b-strangeloop-5cond-ud_iq3_xxs-run1/`](qwen/qwen397b-strangeloop-5cond-ud_iq3_xxs-run1/) | Qwen3.5-397B strangeloop probe | Router analysis | Prefill-only greedy argmax; no generation sampling | Reviewed 397B strangeloop five-condition run. |

## Other Model Families

| Bundle | Model / Scope | Why It Matters |
| --- | --- | --- |
| [`deepseek/ds31-5cond-1/`](deepseek/ds31-5cond-1/) | DeepSeek V3.1 five-condition experiment | Reproducible DeepSeek routing bundle. |
| [`gptoss/gptoss-5cond-1/`](gptoss/gptoss-5cond-1/) | GPT-OSS-120B self-referential five-condition run | Reviewed valid GPT-OSS five-condition artifact. |
| [`gptoss/gptoss-strangeloop-paired-1/`](gptoss/gptoss-strangeloop-paired-1/) | GPT-OSS-120B strangeloop paired control | Reviewed GPT-OSS strangeloop control artifact. |
| [`ling/ling1t-validation-subset-documented/`](ling/ling1t-validation-subset-documented/) | Ling-1T validation subset | Preserved validation documentation bundle. |

## What Is In A Run Folder

Most run folders include:

- `README.md` for the local summary.
- `DOCS/` for plans, results, and interpretation notes.
- `METHOD/` or scripts for analysis code and capture helpers.
- `PROMPTS/` for prompt TSVs or prompt definitions.
- `results/` or `RESULTS/` for small tables, plots, JSON summaries, and generated text.
- `raw/` metadata, token files, logs, and generated text where small enough to track.

Raw tensor dumps, model weights, local environments, and credentials are intentionally excluded from `main`.

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
