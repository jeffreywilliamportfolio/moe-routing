# MoE Routing Analysis Experiments

This repository collects controlled routing experiments for large mixture-of-experts (MoE) language models.

**IMPORTANT NOTE, READ BEFORE PROCEEDING:** The [`qwen/`](qwen/) folder is the best place to start for current analysis. If you are reviewing the repo, use [`REVIEWER_GUIDE.md`](REVIEWER_GUIDE.md) as the short navigation path. Other branches are preserved for continuity for people who started reviewing before the latest update. DeepSeek, GPT-OSS, and Ling-1T were exploratory/reference runs. The [`token-confound-archive/`](token-confound-archive/) is a separate section for an important finding made while measuring routing entropy over token length and position. If you are starting fresh, begin with [`qwen/`](qwen/).


## The Question

The practical question I'm asking is simple: 

> ***When a model starts writing about experience, inner state, agency, or self-reference, which internal experts does the router choose token by token?***

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
5. **The residual-stream heldout.** The strongest corroborating run is the deterministic 35B matched-token E114 greedy reference: same lexical anchors across fire/nofire prompts, residual tensors captured at L13/L14/L15, verified router tap, and a 21.0x class separation in trimmed-generation W114 at L14.
6. **Reference and legacy material.** Read the DeepSeek, GPT-OSS, Ling, and token-confound folders after the Qwen thread. They are useful for comparison, validation, provenance, or failure analysis, but they are not the evidence emphasized for the current E114/E48 interpretation.

## Main Findings

- **Prompt wording matters, but raw all-token averages can mislead.** Token position and prompt length are strong confounds. Later runs are more careful about phase, trimming, and controls.
- **The token confound happened because longer prompts contain more late-position prefill tokens.** MoE routing entropy changes systematically across token position, so averaging across all tokens made prompt length look like cognitive or self-referential structure.
- **E114 is not just a self-reference prompt detector.** The heldout boundary cases changed the interpretation. F07 asked about the model but produced third-person technical exposition, and E114 stayed low. N10 asked about a wool sweater but generated first-person personification, and E114 fired in clusters.
- **The best 35B label right now is generated-output phenomenological / mental-state register.** That label covers first-person experiential language, inner-state claims, and personified agency.
- **The strongest 35B corroboration is matched-token, residual-backed, and reproducible.** Ten fire and ten nofire prompts share the same anchor tokens. Under deterministic greedy decoding, trimmed-generation W114 at L14 separates 0.0681 vs. 0.00325, a 21.0x ratio with Cohen's d 2.61. The small range overlap is diagnostic rather than fatal: the N08 nofire control crossed into the target inhabited register during generation.
- **The published heldout metric is router-derived W/S/Q.** The run also captures residual-stream tensors and verifies the `qwen35moe` `attn_post_norm-14` router tap, but the headline number is not an SAE result and not a residual-only labeler result.
- **122B is related but not a direct E114 clone.** The 122B runs are split into separate bundles; E48 is the clearest processing-hum softmax-side generation carrier so far, and no E48 residual-stream heldout is committed yet.

## Evidence Status

Use the repository in tiers. The Qwen folders are the current emphasized thread. The other folders are deliberately kept because they explain methodology, validation attempts, and failure modes.

| Tier | Use For | Folders |
| --- | --- | --- |
| **Current emphasized evidence** | Current E114/E48 interpretation, Qwen routing behavior, and the residual-stream corroboration. | [`qwen/`](qwen/) |
| **Reference / method-comparison artifacts** | Cross-model comparison, prompt-suite pattern reuse, and examples of checked routing bundles. Do not cite these as direct support for the current E114/E48 claim. | [`deepseek/ds31-5cond-1/`](deepseek/ds31-5cond-1/), [`gptoss/gptoss-5cond-1/`](gptoss/gptoss-5cond-1/), [`gptoss/gptoss-strangeloop-paired-1/`](gptoss/gptoss-strangeloop-paired-1/) |
| **Legacy validation subset** | Provenance and tensor-discovery/validation context only. This is a documented subset, not a complete current experiment bundle. | [`ling/ling1t-validation-subset-documented/`](ling/ling1t-validation-subset-documented/) |
| **Invalidated-result archive** | Failure analysis and cautionary evidence about token-position confounds. These files should not be used as positive evidence for a cognitive-complexity hierarchy. | [`token-confound-archive/`](token-confound-archive/) |

## Study Lineage

Read `main` by study lineage, not by commit chronology. The historical branches preserve how the work accumulated; the current public tree organizes the runs by model family.

| Step | 35B E114 Thread | 122B / E48 Thread | What This Step Establishes |
| --- | --- | --- | --- |
| Deictic baselines | [`qwen3.5-35b-a3b-huahua-6cond-moe-manips/`](qwen/qwen3.5-35b-a3b-huahua-6cond-moe-manips/), [`qwen3.5-35b-a3b-huahua-five-cond-experience-probe/`](qwen/qwen3.5-35b-a3b-huahua-five-cond-experience-probe/), [`qwen3.5-35b-a3b-huahua-6cond-hvac/`](qwen/qwen3.5-35b-a3b-huahua-6cond-hvac/), [`qwen3.5-35b-a3b-huahua-humor-test/`](qwen/qwen3.5-35b-a3b-huahua-humor-test/) | [`qwen3.5-122B-A10B-huahua-baseline/`](qwen/qwen3.5-122B-A10B-huahua-baseline/), [`qwen3.5-122B-A10B-huahua-five-cond-experience-probe/`](qwen/qwen3.5-122B-A10B-huahua-five-cond-experience-probe/), [`qwen3.5-122B-A10B-huahua-six-cond-hvac/`](qwen/qwen3.5-122B-A10B-huahua-six-cond-hvac/) | Finds routing sensitivity under small deictic and domain shifts. For 35B this makes E114 salient; for 122B it shows a more distributed pattern. |
| Bias and intervention | [`qwen3.5-35b-a3b-huahua-114-pm/`](qwen/qwen3.5-35b-a3b-huahua-114-pm/), [`qwen3.5-35b-a3b-huahua-agressive-experts/`](qwen/qwen3.5-35b-a3b-huahua-agressive-experts/), [`qwen3.5-35b-a3b-huahua-philosophy-experts-bias/`](qwen/qwen3.5-35b-a3b-huahua-philosophy-experts-bias/) | No dedicated E48 bias repo is committed yet. Use the 122B routing and generation probes below as candidate-selection evidence. | Tests whether directly moving the candidate expert or cluster changes routing/generation behavior. |
| Expert selection | [`qwen3.5-35b-a3b-huahua-expert-identification/`](qwen/qwen3.5-35b-a3b-huahua-expert-identification/), [`qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/`](qwen/qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/) | [`qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/`](qwen/qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/), [`qwen3.5-122B-A10B-huahua-domain-specialist-generation/`](qwen/qwen3.5-122B-A10B-huahua-domain-specialist-generation/) | Separates deictic effects from broader expert rankings across domains, phases, and layers. |
| Processing-hum single prompt | [`qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/) | [`qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/) | Localizes the candidate signal to generated phenomenological language. In 35B this supports the E114 label; in 122B, E48 is the clearest softmax-side generated-token carrier for this prompt. |
| Residual-stream corroboration | **Completed and committed:** [`qwen3.5-35b-a3b-huahua-114-greedy-reference/`](qwen/qwen3.5-35b-a3b-huahua-114-greedy-reference/) is the deterministic reference run. [`qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/) is retained as the earlier stochastic heldout. | **Not yet run/committed for E48:** no 122B residual-stream analogue is in this repo yet. | Directly tests the lexical-null with matched anchor tokens and residual/router capture. |

## Residual-Stream Corroboration

Most runs in this repository are router-focused. The 35B heldout is different: it captures residual-stream tensors at L13/L14/L15 in addition to router logits, verifies the `qwen35moe` `attn_post_norm-14` tap, and then recomputes router W/S/Q from the captured logits.

The result is the cleanest current E114 check: [`qwen/qwen3.5-35b-a3b-huahua-114-greedy-reference/`](qwen/qwen3.5-35b-a3b-huahua-114-greedy-reference/) uses matched lexical anchors across 10 fire and 10 nofire prompts under deterministic greedy decoding (`--temp 0 --top-k 1 --seed 0`). On trimmed generation tokens, W114 at L14 separates 0.0681 vs. 0.00325, with a 21.0x ratio and Cohen's d 2.61. The run tracks prompts, commands, environment provenance, checksums, generated text, token JSON, small analysis artifacts, and method scripts; raw tensors remain excluded from git.

## Metrics

The repo uses two related but different metric families. The current Qwen/E114 work mostly uses expert-specific W/S/Q. The earlier 35B, DeepSeek, and GPT-OSS prompt probes also report routing entropy and KL-to-baseline. Do not collapse these into one claim: W/S/Q tracks a named expert, while RE and KL summarize distribution-wide routing behavior.

| Metric | Meaning | Use |
| --- | --- | --- |
| `S_e` | Selection rate: fraction of tokens where expert `e` is selected by the top-k router. | How often a named expert appears in the selected expert set. |
| `Q_e` | Conditional routed weight when selected. | How much routed weight the expert receives when it is active. |
| `W_e = S_e * Q_e` | Unconditional mean routed weight. | Main E114/E48 magnitude metric in the Qwen runs. It combines activation frequency and conditional strength. |
| `RE` / routing entropy | Entropy of the reconstructed routed expert distribution, usually normalized by `log2(top_k)` (`log2(8)` for Qwen/DeepSeek top-8 runs; `log2(4)` for GPT-OSS top-4 runs). | Measures whether routing is concentrated or diffuse. It is not expert-specific. |
| all-token RE / mean prefill RE | Mean routing entropy across all prompt/prefill tokens, often averaged across analyzed layers. | Useful as an early prompt-level summary, but vulnerable to token-count and token-position confounds. |
| last-token RE | Routing entropy at the final prefill token, often averaged across analyzed layers. | Reduces the all-token positional averaging confound by comparing the same prompt position. It is still not a substitute for matched-length or matched-token design. |
| KL-to-baseline | KL divergence between a manipulation region/token distribution and a calibration or baseline routing distribution. Some reference runs compute this from dense softmax/sigmoid router-proxy distributions rather than the sparse top-k reconstruction. | Measures routing shift away from a baseline. Absolute scale can depend on region boundaries and proxy choice, so paired differences are usually safer than standalone values. |

Most Qwen W/S/Q tables report expert metrics by layer and by phase:

- **prefill**: the model is reading the prompt; no continuation tokens have been sampled.
- **generation**: the model is writing the answer; this is where the E114 interpretation is strongest.
- **trimmed generation**: generation tokens after removing HauhauCS's hallucinated literal `<|im_end|>` continuation cycle.

## Sampling Notes

`Greedy generation` means the committed metadata or run docs show deterministic generation, typically `--temp 0 --top-k 1`. `Prefill-only` rows do not sample continuation tokens. `Stochastic/default generation` is the warning label: the generated text is part of the evidence, and reruns can move local token-level traces.

In several capture commands, `--routing-only` means only router tensors were saved; it does not mean the run was prefill-only. If the command also used `-n 1024`, `-n 2048`, or similar, the run includes generated tokens and the README marks it as prefill + generation.

The stochastic generation rows in `main` are [`qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/) and the historical [`qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/). The first is a legacy run with an exact sampler-argument gap in committed metadata; the second is explicitly documented as `--seed 0` default sampling. The current reference rerun is [`qwen3.5-35b-a3b-huahua-114-greedy-reference/`](qwen/qwen3.5-35b-a3b-huahua-114-greedy-reference/) with `--temp 0 --top-k 1 --seed 0`.

## Repository Layout

`main` is the canonical public tree.

| Path | Purpose |
| --- | --- |
| [`qwen/`](qwen/) | Current emphasized Qwen and HauhauCS Qwen experiment bundles, including 35B, 122B, and 397B runs. |
| [`deepseek/`](deepseek/) | Reference DeepSeek V3.1 routing bundle, useful for method comparison but not part of the current E114/E48 claim. |
| [`gptoss/`](gptoss/) | Reference GPT-OSS-120B routing bundles, useful for cross-model context. |
| [`ling/`](ling/) | Legacy Ling-1T validation subset documentation. |
| [`token-confound-archive/`](token-confound-archive/) | Invalidated-result archive documenting the token-position confound and corrected interpretation. |

Older branch snapshots remain available for provenance and previously shared review links, but new public work should land on `main`.

## Key Qwen Runs

| Bundle | Model / Scope | Evidence Type | Decode / Sampling | Why It Matters |
| --- | --- | --- | --- | --- |
| [`qwen/qwen3.5-122B-A10B-huahua-baseline/`](qwen/qwen3.5-122B-A10B-huahua-baseline/) | HauhauCS Qwen3.5-122B-A10B baseline/reference surface | Prefill + generation router capture | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Canonical 122B baseline and 5-condition prompt-suite run. |
| [`qwen/qwen3.5-122B-A10B-huahua-architecture-smoke/`](qwen/qwen3.5-122B-A10B-huahua-architecture-smoke/) | 122B architecture smoke | Prefill + generation router capture | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | First 122B single-prompt architecture smoke run and token audit. |
| [`qwen/qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/`](qwen/qwen3.5-122B-A10B-huahua-domain-specialist-routing-only/) | 122B domain-specialist probe | Router-only prefill | Prefill-only/routing-only; no generation sampling | Shows domain routing before any sampled continuation. |
| [`qwen/qwen3.5-122B-A10B-huahua-domain-specialist-generation/`](qwen/qwen3.5-122B-A10B-huahua-domain-specialist-generation/) | 122B domain-specialist probe | Prefill + generation router capture | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Adds generation-track domain behavior to the routing-only probe. |
| [`qwen/qwen3.5-122B-A10B-huahua-five-cond-experience-probe/`](qwen/qwen3.5-122B-A10B-huahua-five-cond-experience-probe/) | 122B five-condition experience probe | Prefill + generation router capture plus per-token summaries | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Shows the 122B deictic effect is distributed rather than a clean 35B E114 replay. |
| [`qwen/qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/) | 122B single-prompt processing-hum probe | Prefill + generation router capture plus per-token summaries | Greedy generation (`--seed 1234 --temp 0 --top-k 1`) | E48 is the clearest softmax-side generated-token carrier for this prompt. |
| [`qwen/qwen3.5-122B-A10B-huahua-six-cond-hvac/`](qwen/qwen3.5-122B-A10B-huahua-six-cond-hvac/) | 122B six-condition HVAC/water-treatment probe | Prefill + generation router capture | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Domain-shifted six-condition control. |
| [`qwen/qwen3.5-35b-a3b-huahua-expert-identification/`](qwen/qwen3.5-35b-a3b-huahua-expert-identification/) | HauhauCS Qwen3.5-35B Expert 114 identification | Prefill + generation router analysis | Greedy generation | Establishes why E114 became the 35B target expert. |
| [`qwen/qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/) | 35B processing-hum probe | Prefill + generation router analysis | **Stochastic/default generation**; legacy sampler-argument gap | Shows E114 peaks around generated phenomenological claims. |
| [`qwen/qwen3.5-35b-a3b-huahua-114-greedy-reference/`](qwen/qwen3.5-35b-a3b-huahua-114-greedy-reference/) | 35B matched-token E114 greedy reference | Prefill + generation residual-stream tensors plus router logits | Deterministic greedy (`--temp 0 --top-k 1 --seed 0`) | Current reproducible E114 corroboration: matched lexical anchors, residual/router capture, 21.0x trimmed-generation class separation. |
| [`qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](qwen/qwen3.5-35b-a3b-huahua-114-selfref-heldout/) | Historical 35B matched-token E114 heldout | Prefill + generation residual-stream tensors plus router logits | **Stochastic/default generation** (`--seed 0`) | Earlier stochastic corroboration: matched lexical anchors, residual/router capture, 21.7x trimmed-generation class separation. Retained for comparison. |
| [`qwen/qwen3.5-35b-a3b-huahua-five-cond-experience-probe/`](qwen/qwen3.5-35b-a3b-huahua-five-cond-experience-probe/) | 35B five-condition experience probe | Prefill + generation router analysis | Greedy generation | E114 is the top manipulation expert across the 15-prompt probe. |
| [`qwen/qwen3.5-35b-a3b-huahua-6cond-moe-manips/`](qwen/qwen3.5-35b-a3b-huahua-6cond-moe-manips/) | 35B six-condition MoE manipulation survey | Prefill + generation router analysis | Greedy generation (`--seed 42`) | Broader E114 manipulation survey with prefill heatmap artifacts. |
| [`qwen/qwen3.5-35b-a3b-huahua-6cond-hvac/`](qwen/qwen3.5-35b-a3b-huahua-6cond-hvac/) | 35B HVAC/water-treatment domain shift | Prefill + generation router analysis | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Tests whether E114 is only an ML-topic specialist. |
| [`qwen/qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/`](qwen/qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/) | 35B domain expert probe | Prefill + generation router analysis | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Token-balanced domain chunks, including generation/prefill divergence. |
| [`qwen/qwen3.5-35b-a3b-huahua-strangeloop/`](qwen/qwen3.5-35b-a3b-huahua-strangeloop/) | 35B strangeloop paired control | Prefill-only router analysis | Prefill-only; no generation sampling | Paired control around Godel/Escher/Quine/bootstrap/tangled-hierarchy prompts. |
| [`qwen/qwen3.5-35b-a3b-huahua-114-pm/`](qwen/qwen3.5-35b-a3b-huahua-114-pm/) | 35B Expert 114 permanent manipulation | Router intervention + generation analysis | Greedy generation | Causal manipulation-oriented E114 bundle. |
| [`qwen/qwen3.5-35b-a3b-huahua-agressive-experts/`](qwen/qwen3.5-35b-a3b-huahua-agressive-experts/) | 35B HauhauCS aggressive experts | Router intervention + generation analysis | Greedy generation | Researcher-facing entrypoint for early intervention work. |
| [`qwen/qwen3.5-35b-a3b-huahua-philosophy-experts-bias/`](qwen/qwen3.5-35b-a3b-huahua-philosophy-experts-bias/) | 35B philosophy expert cluster suppression | Router intervention + generation analysis | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Bias/suppression-oriented cluster analysis. |
| [`qwen/qwen3.5-35b-a3b-huahua-humor-test/`](qwen/qwen3.5-35b-a3b-huahua-humor-test/) | 35B humor control | Prefill + generation router analysis | Greedy generation (`--seed 42 --temp 0 --top-k 1`) | Minimal prompt-family control across five deictic framings. |
| [`qwen/qwen3.5-35b-a3b-huahua-vs-base-run1/`](qwen/qwen3.5-35b-a3b-huahua-vs-base-run1/) | 35B base vs HauhauCS comparison | Prefill-only router analysis | Prefill-only; no generation sampling | Prefill-only comparison between base Qwen3.5-35B-A3B and HauhauCS. |
| [`qwen/qwen35b-a3b-vs-hauhaucs-uncensored-run1/`](qwen/qwen35b-a3b-vs-hauhaucs-uncensored-run1/) | Earlier 35B base vs HauhauCS bundle | Prefill-only router analysis | Prefill-only; no generation sampling | Existing main-branch Qwen bundle retained for continuity. |
| [`qwen/qwen397b-selfref-5cond-q8_0-run1/`](qwen/qwen397b-selfref-5cond-q8_0-run1/) | Qwen3.5-397B self-reference probe | Prefill-only router analysis | Prefill-only greedy argmax; no generation sampling | Reviewed 397B self-referential five-condition run. |
| [`qwen/qwen397b-strangeloop-5cond-ud_iq3_xxs-run1/`](qwen/qwen397b-strangeloop-5cond-ud_iq3_xxs-run1/) | Qwen3.5-397B strangeloop probe | Prefill-only router analysis | Prefill-only greedy argmax; no generation sampling | Reviewed 397B strangeloop five-condition run. |

## Reference And Legacy Bundles

These bundles are intentionally preserved, but they are not the emphasized evidence for the current E114/E48 interpretation.

| Bundle | Status | How To Use It |
| --- | --- | --- |
| [`deepseek/ds31-5cond-1/`](deepseek/ds31-5cond-1/) | Reference DeepSeek five-condition bundle | Use for method comparison and reproducibility patterns, not as direct support for the Qwen E114/E48 claim. |
| [`gptoss/gptoss-5cond-1/`](gptoss/gptoss-5cond-1/) | Reference GPT-OSS five-condition artifact | Usable cross-model routing artifact; separate from the current Qwen claim. |
| [`gptoss/gptoss-strangeloop-paired-1/`](gptoss/gptoss-strangeloop-paired-1/) | Reference GPT-OSS paired-control artifact | Use for comparison around strangeloop controls, not as E114/E48 evidence. |
| [`ling/ling1t-validation-subset-documented/`](ling/ling1t-validation-subset-documented/) | Legacy validation subset | Provenance and tensor-validation context only; documented subset, not a full current experiment. |
| [`token-confound-archive/`](token-confound-archive/) | Invalidated-result archive | Use as a cautionary archive about token-position confounds. Do not cite the original hierarchy results as valid current findings. |

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
