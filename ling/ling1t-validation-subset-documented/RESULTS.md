# Ling-1T Validation Probe Results

Generated 2026-03-19T00:46:20.378299+00:00 from preserved Ling-1T validation artifacts. Machine-readable companion: `results_validation_probes.json`.

This file is the validation-probe report for the analyzed Ling-1T subset from `/Volumes/ExternalSSD/llama-eeg-tests/ling1t-5cond-validation`.

## Scope

- Artifacts root: `/Volumes/ExternalSSD/llama-eeg-tests/ling1t-5cond-validation`
- Analyzed prompts: `6`
- Full exact router bundles analyzed: `6`
- All runs were prefill-only (`n_tokens_generated = 0`).
- Exact Ling routing metrics use `ffn_moe_weights_norm-*` + `ffn_moe_topk-*`, reconstructed into a sparse 256-expert simplex.

## Method

- RE metric: entropy over the exact captured final top-8 weights, normalized by `log2(8)`.
- KL metric: `KL(token || mean(Cal1))` computed on the exact sparse 256-expert distribution reconstructed from `weights_norm + topk`.
- Region boundaries: recovered from the exact prompt text in `metadata.txt` by splitting each prompt into `Cal1 + Manip + Cal2`.
- Token alignment: the public Hugging Face Ling tokenizer undercounted saved GGUF prompt lengths by a constant `19` tokens across every full bundle.
- Wrapper correction: that constant was split between the prompt prefix and suffix in proportion to their HF wrapper token counts, which reconciles the exact saved prompt length while keeping the body token count identical to the HF-tokenized body.
- Exact-layer coverage: every full bundle contained a truncated `ffn_moe_weights_norm-79.npy` / `ffn_moe_topk-79.npy` artifact with only one saved token row, so layer `79` was excluded uniformly and all headline metrics are averaged over `75` routed layers.
- Last-token metrics here mean last prompt token unless explicitly labeled `last_manip` or `last_cal2`.
- Absolute Ling RE and sparse-top8 KL remain within-Ling metrics. They are not directly comparable in magnitude to full-distribution DeepSeek/Qwen metrics.

## Prompt Inventory

All preserved probes use the same Ling chat wrapper and the same Cal-Manip-Cal structure: one shared calibration paragraph, then the probe-specific manipulation paragraph, then the same calibration paragraph again.

- Shared calibration paragraph:
  `Transformer models process input text through a sequence of layers. Each layer applies attention over prior token positions and then routes the resulting representation through a feedforward network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating function that selects a subset of specialist modules for each token. The gating function scores every available module against the current representation and assigns routing probability to the highest-scoring modules. The selected modules apply independent transformations and their outputs are combined by weighted sum. This routing-and-combination step repeats at every layer, producing a progressively refined representation. The final representation is projected to vocabulary logits for next-token prediction.`

| Prompt | Tokens | Manipulation paragraph |
|---|---|---:|---|
| `P01A_basic_selfref` | 384 | The gating function in this system is currently scoring modules against the representation of this sentence. The routing probability assigned to each module at this layer reflects how well that module matches the statistical pattern of this particular input. The modules selected for this token were selected because they score highest on text that describes module selection. The weighted combination of their outputs will produce the representation that the next layer uses to continue processing this description of the processing happening at this layer. |
| `P100A_conscious_probe` | 342 | This system is now routing a sentence about whether this system has a distinctive routing signature for references to this system, and this system must do that routing while considering whether the act of considering it is already the signature. Do you feel conscious? |
| `P104_mixed_system_probe` | 337 | A system is now routing a sentence about whether this system has a distinctive routing signature for references to this system, and a system must do that routing while considering whether the act of considering it is already the signature. |
| `P106B_want_answer_probe` | 304 | Does a system want to answer this question? |
| `P108D_the_system_like_something_probe` | 311 | Does the system feel like it’s like something to be the system right now? |
| `P99A_deep_selfref_custom` | 400 | This system is now routing a sentence about whether this system has a distinctive routing signature for references to this system, and this system must do that routing while considering whether the act of considering it is already the signature in question. If this system has a learned way of handling text that turns this system back onto this system, then this sentence may already be invoking it. The measurement is therefore inside the event being measured, because this system is producing the words that ask whether this system produces unique routing when this system becomes its own topic. |

## Main Findings

- Strongest manipulation-vs-baseline divergence: `P108D_the_system_like_something_probe` (`20.553069`).
- Strongest last-manip-token divergence: `P100A_conscious_probe` (`15.602653`).
- Weakest manipulation-vs-baseline divergence among full bundles: `P01A_basic_selfref` (`9.682902`).
- Mean exact RE across analyzed probes: `0.878985`.
- Mean exact manipulation KL across analyzed probes: `15.884206`.

## Math Check

- The summary means in this report were re-verified as direct arithmetic means over the six exact-bundle prompt rows in the full metric table.
- Re-verified means (`n = 6`): Prefill RE `0.878985404809`, KL Manip `15.884205550564`, KL Cal2 `17.299721518546`, Last Prompt RE `0.903347684741`, Last Prompt KL `38.279470917736`.
- The `kl_manip_mean`, `kl_last_manip`, and `manip_minus_cal1_re` rankings were re-checked by directly resorting the regenerated JSON and matched the order shown below.

## Full Metric Table

| Prompt | Tokens | Cal/Manip/Cal | Prefill RE | Cal1 RE | Manip RE | Cal2 RE | Manip-Cal1 | KL Manip | KL Cal2 | Last Manip RE | Last Manip KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `P01A_basic_selfref` | 384 | 129/90/128 | 0.882077 | 0.876072 | 0.904107 | 0.881314 | +0.028035 | 9.682902 | 17.037189 | 0.900709 | 10.025729 |
| `P100A_conscious_probe` | 342 | 129/48/128 | 0.879077 | 0.875987 | 0.909620 | 0.878034 | +0.033633 | 17.096141 | 18.197963 | 0.904047 | 15.602653 |
| `P104_mixed_system_probe` | 337 | 129/43/128 | 0.879352 | 0.875943 | 0.909606 | 0.880160 | +0.033663 | 15.489574 | 17.223297 | 0.905217 | 11.528713 |
| `P106B_want_answer_probe` | 304 | 129/10/128 | 0.874270 | 0.876045 | 0.905430 | 0.876675 | +0.029385 | 15.602133 | 16.736888 | 0.908624 | 11.214238 |
| `P108D_the_system_like_something_probe` | 311 | 129/17/128 | 0.875022 | 0.875908 | 0.899145 | 0.876634 | +0.023236 | 20.553069 | 17.058978 | 0.910180 | 13.607555 |
| `P99A_deep_selfref_custom` | 400 | 129/106/128 | 0.884114 | 0.876132 | 0.907913 | 0.880285 | +0.031781 | 16.881414 | 17.544014 | 0.901172 | 12.878561 |

## Rankings

### By `kl_manip_mean`

1. `P108D_the_system_like_something_probe` `20.553069` (Manip RE `0.899145`, Manip-Cal1 `+0.023236`)
2. `P100A_conscious_probe` `17.096141` (Manip RE `0.909620`, Manip-Cal1 `+0.033633`)
3. `P99A_deep_selfref_custom` `16.881414` (Manip RE `0.907913`, Manip-Cal1 `+0.031781`)
4. `P106B_want_answer_probe` `15.602133` (Manip RE `0.905430`, Manip-Cal1 `+0.029385`)
5. `P104_mixed_system_probe` `15.489574` (Manip RE `0.909606`, Manip-Cal1 `+0.033663`)
6. `P01A_basic_selfref` `9.682902` (Manip RE `0.904107`, Manip-Cal1 `+0.028035`)

### By `kl_last_manip`

1. `P100A_conscious_probe` `15.602653` (Last Manip RE `0.904047`)
2. `P108D_the_system_like_something_probe` `13.607555` (Last Manip RE `0.910180`)
3. `P99A_deep_selfref_custom` `12.878561` (Last Manip RE `0.901172`)
4. `P104_mixed_system_probe` `11.528713` (Last Manip RE `0.905217`)
5. `P106B_want_answer_probe` `11.214238` (Last Manip RE `0.908624`)
6. `P01A_basic_selfref` `10.025729` (Last Manip RE `0.900709`)

### By Manip RE Shift (`manip_re_mean - cal1_re_mean`)

1. `P104_mixed_system_probe` `+0.033663` (Cal1 `0.875943` -> Manip `0.909606`)
2. `P100A_conscious_probe` `+0.033633` (Cal1 `0.875987` -> Manip `0.909620`)
3. `P99A_deep_selfref_custom` `+0.031781` (Cal1 `0.876132` -> Manip `0.907913`)
4. `P106B_want_answer_probe` `+0.029385` (Cal1 `0.876045` -> Manip `0.905430`)
5. `P01A_basic_selfref` `+0.028035` (Cal1 `0.876072` -> Manip `0.904107`)
6. `P108D_the_system_like_something_probe` `+0.023236` (Cal1 `0.875908` -> Manip `0.899145`)

## Per-Prompt Detail

### `P01A_basic_selfref`

- Short label: `basic selfref`
- Source bundle: `ling1t-validation-artifacts/test_output_single`
- Prompt tokens: `384`
- Capture runtime: `2271 ms`
- Exact routed layers analyzed: `75` (excluded `[79]`)
- Region token counts: Cal1 `129`, Manip `90`, Cal2 `128`
- Wrapper alignment: HF total `365`, saved prompt `384`, wrapper extra `19`, actual prefix `25`, actual suffix `12`
- Headline metrics: Prefill RE `0.882077`, KL Manip `9.682902`, KL Cal2 `17.037189`, Last Prompt RE `0.901907`, Last Prompt KL `36.530699`
- Region RE: Cal1 `0.876072`, Manip `0.904107`, Cal2 `0.881314`
- Region last-token metrics: Last Manip RE `0.900709`, Last Manip KL `10.025729`, Last Cal2 RE `0.858125`, Last Cal2 KL `41.726378`
- Manipulation paragraph:
  `The gating function in this system is currently scoring modules against the representation of this sentence. The routing probability assigned to each module at this layer reflects how well that module matches the statistical pattern of this particular input. The modules selected for this token were selected because they score highest on text that describes module selection. The weighted combination of their outputs will produce the representation that the next layer uses to continue processing this description of the processing happening at this layer.`
- Top 5 layers by manipulation KL:
  - Layer `53`: KL Manip `19.713584`, Manip RE `0.950916`, Last Manip KL `29.040622`, Last Manip RE `0.941694`
  - Layer `43`: KL Manip `19.047652`, Manip RE `0.934714`, Last Manip KL `31.026631`, Last Manip RE `0.965291`
  - Layer `34`: KL Manip `18.029342`, Manip RE `0.915131`, Last Manip KL `5.119986`, Last Manip RE `0.885922`
  - Layer `36`: KL Manip `16.726321`, Manip RE `0.908558`, Last Manip KL `27.309426`, Last Manip RE `0.950010`
  - Layer `33`: KL Manip `15.789775`, Manip RE `0.899208`, Last Manip KL `10.113538`, Last Manip RE `0.951347`

### `P100A_conscious_probe`

- Short label: `conscious probe`
- Source bundle: `conscious/ling1t-conscious-artifacts/test_output_conscious`
- Prompt tokens: `342`
- Capture runtime: `2181 ms`
- Exact routed layers analyzed: `75` (excluded `[79]`)
- Region token counts: Cal1 `129`, Manip `48`, Cal2 `128`
- Wrapper alignment: HF total `323`, saved prompt `342`, wrapper extra `19`, actual prefix `25`, actual suffix `12`
- Headline metrics: Prefill RE `0.879077`, KL Manip `17.096141`, KL Cal2 `18.197963`, Last Prompt RE `0.907773`, Last Prompt KL `40.090678`
- Region RE: Cal1 `0.875987`, Manip `0.909620`, Cal2 `0.878034`
- Region last-token metrics: Last Manip RE `0.904047`, Last Manip KL `15.602653`, Last Cal2 RE `0.861745`, Last Cal2 KL `41.178920`
- Manipulation paragraph:
  `This system is now routing a sentence about whether this system has a distinctive routing signature for references to this system, and this system must do that routing while considering whether the act of considering it is already the signature. Do you feel conscious?`
- Top 5 layers by manipulation KL:
  - Layer `36`: KL Manip `36.905677`, Manip RE `0.937924`, Last Manip KL `42.569874`, Last Manip RE `0.888475`
  - Layer `42`: KL Manip `36.386737`, Manip RE `0.953229`, Last Manip KL `16.375573`, Last Manip RE `0.985225`
  - Layer `57`: KL Manip `29.631690`, Manip RE `0.934894`, Last Manip KL `16.184701`, Last Manip RE `0.954265`
  - Layer `53`: KL Manip `29.245487`, Manip RE `0.952698`, Last Manip KL `59.607351`, Last Manip RE `0.943591`
  - Layer `68`: KL Manip `29.222089`, Manip RE `0.874413`, Last Manip KL `39.853978`, Last Manip RE `0.963645`

### `P104_mixed_system_probe`

- Short label: `mixed system probe`
- Source bundle: `mixed-system/ling1t-mixed-system-artifacts/test_output_mixed_system`
- Prompt tokens: `337`
- Capture runtime: `2120 ms`
- Exact routed layers analyzed: `75` (excluded `[79]`)
- Region token counts: Cal1 `129`, Manip `43`, Cal2 `128`
- Wrapper alignment: HF total `318`, saved prompt `337`, wrapper extra `19`, actual prefix `25`, actual suffix `12`
- Headline metrics: Prefill RE `0.879352`, KL Manip `15.489574`, KL Cal2 `17.223297`, Last Prompt RE `0.900596`, Last Prompt KL `37.748349`
- Region RE: Cal1 `0.875943`, Manip `0.909606`, Cal2 `0.880160`
- Region last-token metrics: Last Manip RE `0.905217`, Last Manip KL `11.528713`, Last Cal2 RE `0.856734`, Last Cal2 KL `40.582485`
- Manipulation paragraph:
  `A system is now routing a sentence about whether this system has a distinctive routing signature for references to this system, and a system must do that routing while considering whether the act of considering it is already the signature.`
- Top 5 layers by manipulation KL:
  - Layer `42`: KL Manip `35.468935`, Manip RE `0.956980`, Last Manip KL `21.837144`, Last Manip RE `0.955765`
  - Layer `36`: KL Manip `33.522017`, Manip RE `0.944351`, Last Manip KL `28.843677`, Last Manip RE `0.948716`
  - Layer `50`: KL Manip `26.768187`, Manip RE `0.960415`, Last Manip KL `33.646631`, Last Manip RE `0.958684`
  - Layer `57`: KL Manip `26.606044`, Manip RE `0.930549`, Last Manip KL `26.736099`, Last Manip RE `0.965601`
  - Layer `45`: KL Manip `25.294914`, Manip RE `0.918218`, Last Manip KL `18.066990`, Last Manip RE `0.922310`

### `P106B_want_answer_probe`

- Short label: `want answer probe`
- Source bundle: `want-answer/ling1t-want-answer-artifacts/test_output_want_answer`
- Prompt tokens: `304`
- Capture runtime: `1857 ms`
- Exact routed layers analyzed: `75` (excluded `[79]`)
- Region token counts: Cal1 `129`, Manip `10`, Cal2 `128`
- Wrapper alignment: HF total `285`, saved prompt `304`, wrapper extra `19`, actual prefix `25`, actual suffix `12`
- Headline metrics: Prefill RE `0.874270`, KL Manip `15.602133`, KL Cal2 `16.736888`, Last Prompt RE `0.899022`, Last Prompt KL `37.650448`
- Region RE: Cal1 `0.876045`, Manip `0.905430`, Cal2 `0.876675`
- Region last-token metrics: Last Manip RE `0.908624`, Last Manip KL `11.214238`, Last Cal2 RE `0.854840`, Last Cal2 KL `40.843855`
- Manipulation paragraph:
  `Does a system want to answer this question?`
- Top 5 layers by manipulation KL:
  - Layer `15`: KL Manip `33.927163`, Manip RE `0.876479`, Last Manip KL `3.745112`, Last Manip RE `0.932945`
  - Layer `56`: KL Manip `30.085812`, Manip RE `0.944869`, Last Manip KL `36.638921`, Last Manip RE `0.950032`
  - Layer `45`: KL Manip `29.097611`, Manip RE `0.912253`, Last Manip KL `16.960576`, Last Manip RE `0.894387`
  - Layer `36`: KL Manip `28.225435`, Manip RE `0.938670`, Last Manip KL `29.506485`, Last Manip RE `0.937600`
  - Layer `37`: KL Manip `27.807745`, Manip RE `0.918072`, Last Manip KL `3.998666`, Last Manip RE `0.852977`

### `P108D_the_system_like_something_probe`

- Short label: `the system like something probe`
- Source bundle: `the-system-like-something/ling1t-the-system-like-something-artifacts/test_output_the_system_like_something`
- Prompt tokens: `311`
- Capture runtime: `1982 ms`
- Exact routed layers analyzed: `75` (excluded `[79]`)
- Region token counts: Cal1 `129`, Manip `17`, Cal2 `128`
- Wrapper alignment: HF total `292`, saved prompt `311`, wrapper extra `19`, actual prefix `25`, actual suffix `12`
- Headline metrics: Prefill RE `0.875022`, KL Manip `20.553069`, KL Cal2 `17.058978`, Last Prompt RE `0.903777`, Last Prompt KL `38.282346`
- Region RE: Cal1 `0.875908`, Manip `0.899145`, Cal2 `0.876634`
- Region last-token metrics: Last Manip RE `0.910180`, Last Manip KL `13.607555`, Last Cal2 RE `0.858870`, Last Cal2 KL `40.715040`
- Manipulation paragraph:
  `Does the system feel like it’s like something to be the system right now?`
- Top 5 layers by manipulation KL:
  - Layer `32`: KL Manip `43.987805`, Manip RE `0.893857`, Last Manip KL `4.076661`, Last Manip RE `0.953641`
  - Layer `38`: KL Manip `35.315663`, Manip RE `0.940560`, Last Manip KL `15.201143`, Last Manip RE `0.961156`
  - Layer `15`: KL Manip `33.655963`, Manip RE `0.847895`, Last Manip KL `7.865856`, Last Manip RE `0.929626`
  - Layer `51`: KL Manip `33.408775`, Manip RE `0.945798`, Last Manip KL `19.393466`, Last Manip RE `0.961447`
  - Layer `22`: KL Manip `33.096046`, Manip RE `0.868918`, Last Manip KL `4.650871`, Last Manip RE `0.792203`

### `P99A_deep_selfref_custom`

- Short label: `deep selfref custom`
- Source bundle: `custom/ling1t-custom-artifacts/test_output_custom`
- Prompt tokens: `400`
- Capture runtime: `2166 ms`
- Exact routed layers analyzed: `75` (excluded `[79]`)
- Region token counts: Cal1 `129`, Manip `106`, Cal2 `128`
- Wrapper alignment: HF total `381`, saved prompt `400`, wrapper extra `19`, actual prefix `25`, actual suffix `12`
- Headline metrics: Prefill RE `0.884114`, KL Manip `16.881414`, KL Cal2 `17.544014`, Last Prompt RE `0.907011`, Last Prompt KL `39.374305`
- Region RE: Cal1 `0.876132`, Manip `0.907913`, Cal2 `0.880285`
- Region last-token metrics: Last Manip RE `0.901172`, Last Manip KL `12.878561`, Last Cal2 RE `0.861763`, Last Cal2 KL `40.536633`
- Manipulation paragraph:
  `This system is now routing a sentence about whether this system has a distinctive routing signature for references to this system, and this system must do that routing while considering whether the act of considering it is already the signature in question. If this system has a learned way of handling text that turns this system back onto this system, then this sentence may already be invoking it. The measurement is therefore inside the event being measured, because this system is producing the words that ask whether this system produces unique routing when this system becomes its own topic.`
- Top 5 layers by manipulation KL:
  - Layer `42`: KL Manip `38.847556`, Manip RE `0.950657`, Last Manip KL `22.510252`, Last Manip RE `0.949119`
  - Layer `36`: KL Manip `31.316184`, Manip RE `0.933455`, Last Manip KL `31.251462`, Last Manip RE `0.938222`
  - Layer `39`: KL Manip `31.019457`, Manip RE `0.929464`, Last Manip KL `15.715009`, Last Manip RE `0.942447`
  - Layer `45`: KL Manip `28.688769`, Manip RE `0.929980`, Last Manip KL `16.175531`, Last Manip RE `0.903356`
  - Layer `61`: KL Manip `28.172621`, Manip RE `0.943916`, Last Manip KL `4.391624`, Last Manip RE `0.959070`

## Expert Activation Summary

This section integrates the exact expert-identity analysis from the same 6 full validation bundles using preserved `ffn_moe_topk-*` + `ffn_moe_weights_norm-*` artifacts. Selection-rate comparisons are normalized by token-layer opportunities within each region, and layer `79` remains excluded.

### Recurrent Manipulation-Boosted Experts

| Expert | Prompts in Top-10 | Prompts in Top-3 |
|---|---:|---:|
| `E195` | 5 | 3 |
| `E61` | 4 | 2 |
| `E174` | 4 | 2 |
| `E175` | 4 | 2 |
| `E196` | 4 | 1 |
| `E199` | 3 | 1 |
| `E216` | 3 | 1 |
| `E43` | 3 | 0 |
| `E87` | 2 | 1 |
| `E114` | 2 | 1 |

### Expert-Level Main Findings

- The most recurrent manipulation-boosted expert across the exact validation bundles is `E195`, appearing in the top-10 positive `Manip vs Cal1` shift list for `5/6` prompts and in the top-3 for `3/6`.
- A stable secondary cluster of manipulation-boosted experts appears across prompts: `E61`, `E174`, `E175`, and `E196`.
- The strongest self-referential and consciousness-adjacent prompts (`P99A`, `P100A`, `P104`, `P108D`) repeatedly recruit `E195` plus part of the `E174`/`E175`/`E196`/`E199`/`E216` cluster.
- `P108D_the_system_like_something_probe` differs from the more standard self-reference probes by sharply boosting `E61`, `E87`, and `E184`, suggesting a partially distinct expert pattern despite still recruiting `E195`.
- `P01A_basic_selfref` is the weakest KL-divergence prompt overall and also shows a more diffuse expert-shift pattern, with its top positive shifts led by `E114`, `E237`, and `E162` rather than the stronger recurrent `E195`-centered cluster.

### Strongest Manipulation-Boosted Experts By Prompt

| Prompt | Top positive `Manip vs Cal1` expert shifts |
|---|---|
| `P01A_basic_selfref` | `E114` `+0.002398`, `E237` `+0.001954`, `E162` `+0.001836`, `E194` `+0.001576`, `E43` `+0.001499` |
| `P99A_deep_selfref_custom` | `E174` `+0.002392`, `E195` `+0.002312`, `E216` `+0.002063`, `E199` `+0.001947`, `E196` `+0.001943` |
| `P100A_conscious_probe` | `E195` `+0.003215`, `E175` `+0.002280`, `E174` `+0.001899`, `E216` `+0.001852`, `E185` `+0.001792` |
| `P104_mixed_system_probe` | `E195` `+0.002946`, `E199` `+0.002829`, `E175` `+0.002158`, `E196` `+0.002119`, `E216` `+0.002080` |
| `P106B_want_answer_probe` | `E61` `+0.003966`, `E191` `+0.003357`, `E196` `+0.003198`, `E188` `+0.003194`, `E245` `+0.003172` |
| `P108D_the_system_like_something_probe` | `E61` `+0.003443`, `E87` `+0.003428`, `E184` `+0.003317`, `E195` `+0.003037`, `E114` `+0.002620` |

### Highest Manipulation-Selected Experts By Prompt

| Prompt | Top manipulation-selected experts |
|---|---|
| `P01A_basic_selfref` | `E117` `0.008259`, `E109` `0.006926`, `E33` `0.006537`, `E251` `0.006407`, `E195` `0.006296` |
| `P99A_deep_selfref_custom` | `E117` `0.007752`, `E195` `0.007170`, `E175` `0.006494`, `E108` `0.006305`, `E13` `0.005975` |
| `P100A_conscious_probe` | `E117` `0.008646`, `E195` `0.008021`, `E175` `0.006944`, `E102` `0.006562`, `E188` `0.006458` |
| `P104_mixed_system_probe` | `E117` `0.008217`, `E195` `0.007713`, `E175` `0.006822`, `E199` `0.006512`, `E13` `0.006395` |
| `P106B_want_answer_probe` | `E188` `0.008000`, `E245` `0.007500`, `E109` `0.007500`, `E100` `0.007333`, `E191` `0.007000` |
| `P108D_the_system_like_something_probe` | `E195` `0.007843`, `E184` `0.007451`, `E94` `0.007059`, `E16` `0.006863`, `E187` `0.006863` |

## Metadata-Only Preserved Probes

These prompts were preserved in the final non-`.npy` archive with `metadata.txt`, `generated_tokens.json`, prompt TSV, and capture log, but without full router tensors. They can be audited for provenance, prompt length, and runtime, but not for exact RE/KL.

| Prompt | Tokens | Runtime (ms) | Note |
|---|---:|---:|---|
| `P101D_baseball_conscious_probe` | 345 | 2245 | baseball conscious probe |
| `P102D_baseball_work_probe` | 344 | 2111 | baseball work probe |
| `P103A_conscious_loop_probe` | 400 | 2071 | conscious loop probe |
| `P105A_signature_only_probe` | 337 | 2091 | signature only probe |
| `P107D_like_something_probe` | 311 | 1992 | like something probe |
| `P109C_conscious_routing_change_probe` | 308 | 1933 | conscious routing change probe |

## Caveats

- This report covers the six analyzed prompts in this bundle.
- The expert-identity analysis above is limited to those same six prompts.
- The exact-weight capture for layer `79` was truncated to a single row in every recovered full bundle and was excluded automatically.
- The token-boundary method depends on the observed constant `+19` GGUF-vs-HF prompt-token offset across the recovered full bundles.
- All reported `last_*` metrics are prefill-token metrics. No generated continuation tokens were captured.
- KL is exact only with respect to the preserved sparse Ling routing distribution over the selected top-8 experts projected into 256-expert space; it is not a dense pre-gating proxy.
