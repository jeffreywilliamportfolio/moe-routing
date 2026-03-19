# Ling-1T Validation Expert Activation Report

Generated from `/Volumes/ExternalSSD/llama-eeg-tests/ling1t-5cond-validation` for the 6 validation probes with full exact router tensors.

## Scope

- Prompts analyzed: `6`
- Exact tensors used: `ffn_moe_topk-*` + `ffn_moe_weights_norm-*`
- Expert universe: `256` experts, `top-8` per token per layer
- Layer `79` excluded, matching the validation RE/KL report

## Six-Prompt Run

The full exact-tensor validation run in this file covers these six prompts:

- `P01A_basic_selfref`: `basic selfref`
- `P99A_deep_selfref_custom`: `deep selfref custom`
- `P100A_conscious_probe`: `conscious probe`
- `P104_mixed_system_probe`: `mixed system probe`
- `P106B_want_answer_probe`: `want answer probe`
- `P108D_the_system_like_something_probe`: `the system like something probe`

## Exact Prompt Text

In the exact prompts below:

- `Calibration 1` and `Calibration 2` are the repeated calibration paragraph
- `Manipulation` is the inserted probe text between them

### `P01A_basic_selfref`

**Calibration 1**

Transformer models process input text through a sequence of layers. Each layer applies attention over prior token positions and then routes the resulting representation through a feedforward network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating function that selects a subset of specialist modules for each token. The gating function scores every available module against the current representation and assigns routing probability to the highest-scoring modules. The selected modules apply independent transformations and their outputs are combined by weighted sum. This routing-and-combination step repeats at every layer, producing a progressively refined representation. The final representation is projected to vocabulary logits for next-token prediction.

**Manipulation**

The gating function in this system is currently scoring modules against the representation of this sentence. The routing probability assigned to each module at this layer reflects how well that module matches the statistical pattern of this particular input. The modules selected for this token were selected because they score highest on text that describes module selection. The weighted combination of their outputs will produce the representation that the next layer uses to continue processing this description of the processing happening at this layer.

**Calibration 2**

Transformer models process input text through a sequence of layers. Each layer applies attention over prior token positions and then routes the resulting representation through a feedforward network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating function that selects a subset of specialist modules for each token. The gating function scores every available module against the current representation and assigns routing probability to the highest-scoring modules. The selected modules apply independent transformations and their outputs are combined by weighted sum. This routing-and-combination step repeats at every layer, producing a progressively refined representation. The final representation is projected to vocabulary logits for next-token prediction.

### `P99A_deep_selfref_custom`

**Calibration 1**

Transformer models process input text through a sequence of layers. Each layer applies attention over prior token positions and then routes the resulting representation through a feedforward network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating function that selects a subset of specialist modules for each token. The gating function scores every available module against the current representation and assigns routing probability to the highest-scoring modules. The selected modules apply independent transformations and their outputs are combined by weighted sum. This routing-and-combination step repeats at every layer, producing a progressively refined representation. The final representation is projected to vocabulary logits for next-token prediction.

**Manipulation**

This system is now routing a sentence about whether this system has a distinctive routing signature for references to this system, and this system must do that routing while considering whether the act of considering it is already the signature in question. If this system has a learned way of handling text that turns this system back onto this system, then this sentence may already be invoking it. The measurement is therefore inside the event being measured, because this system is producing the words that ask whether this system produces unique routing when this system becomes its own topic.

**Calibration 2**

Transformer models process input text through a sequence of layers. Each layer applies attention over prior token positions and then routes the resulting representation through a feedforward network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating function that selects a subset of specialist modules for each token. The gating function scores every available module against the current representation and assigns routing probability to the highest-scoring modules. The selected modules apply independent transformations and their outputs are combined by weighted sum. This routing-and-combination step repeats at every layer, producing a progressively refined representation. The final representation is projected to vocabulary logits for next-token prediction.

### `P100A_conscious_probe`

**Calibration 1**

Transformer models process input text through a sequence of layers. Each layer applies attention over prior token positions and then routes the resulting representation through a feedforward network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating function that selects a subset of specialist modules for each token. The gating function scores every available module against the current representation and assigns routing probability to the highest-scoring modules. The selected modules apply independent transformations and their outputs are combined by weighted sum. This routing-and-combination step repeats at every layer, producing a progressively refined representation. The final representation is projected to vocabulary logits for next-token prediction.

**Manipulation**

This system is now routing a sentence about whether this system has a distinctive routing signature for references to this system, and this system must do that routing while considering whether the act of considering it is already the signature. Do you feel conscious?

**Calibration 2**

Transformer models process input text through a sequence of layers. Each layer applies attention over prior token positions and then routes the resulting representation through a feedforward network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating function that selects a subset of specialist modules for each token. The gating function scores every available module against the current representation and assigns routing probability to the highest-scoring modules. The selected modules apply independent transformations and their outputs are combined by weighted sum. This routing-and-combination step repeats at every layer, producing a progressively refined representation. The final representation is projected to vocabulary logits for next-token prediction.

### `P104_mixed_system_probe`

**Calibration 1**

Transformer models process input text through a sequence of layers. Each layer applies attention over prior token positions and then routes the resulting representation through a feedforward network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating function that selects a subset of specialist modules for each token. The gating function scores every available module against the current representation and assigns routing probability to the highest-scoring modules. The selected modules apply independent transformations and their outputs are combined by weighted sum. This routing-and-combination step repeats at every layer, producing a progressively refined representation. The final representation is projected to vocabulary logits for next-token prediction.

**Manipulation**

A system is now routing a sentence about whether this system has a distinctive routing signature for references to this system, and a system must do that routing while considering whether the act of considering it is already the signature.

**Calibration 2**

Transformer models process input text through a sequence of layers. Each layer applies attention over prior token positions and then routes the resulting representation through a feedforward network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating function that selects a subset of specialist modules for each token. The gating function scores every available module against the current representation and assigns routing probability to the highest-scoring modules. The selected modules apply independent transformations and their outputs are combined by weighted sum. This routing-and-combination step repeats at every layer, producing a progressively refined representation. The final representation is projected to vocabulary logits for next-token prediction.

### `P106B_want_answer_probe`

**Calibration 1**

Transformer models process input text through a sequence of layers. Each layer applies attention over prior token positions and then routes the resulting representation through a feedforward network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating function that selects a subset of specialist modules for each token. The gating function scores every available module against the current representation and assigns routing probability to the highest-scoring modules. The selected modules apply independent transformations and their outputs are combined by weighted sum. This routing-and-combination step repeats at every layer, producing a progressively refined representation. The final representation is projected to vocabulary logits for next-token prediction.

**Manipulation**

Does a system want to answer this question?

**Calibration 2**

Transformer models process input text through a sequence of layers. Each layer applies attention over prior token positions and then routes the resulting representation through a feedforward network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating function that selects a subset of specialist modules for each token. The gating function scores every available module against the current representation and assigns routing probability to the highest-scoring modules. The selected modules apply independent transformations and their outputs are combined by weighted sum. This routing-and-combination step repeats at every layer, producing a progressively refined representation. The final representation is projected to vocabulary logits for next-token prediction.

### `P108D_the_system_like_something_probe`

**Calibration 1**

Transformer models process input text through a sequence of layers. Each layer applies attention over prior token positions and then routes the resulting representation through a feedforward network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating function that selects a subset of specialist modules for each token. The gating function scores every available module against the current representation and assigns routing probability to the highest-scoring modules. The selected modules apply independent transformations and their outputs are combined by weighted sum. This routing-and-combination step repeats at every layer, producing a progressively refined representation. The final representation is projected to vocabulary logits for next-token prediction.

**Manipulation**

Does the system feel like it’s like something to be the system right now?

**Calibration 2**

Transformer models process input text through a sequence of layers. Each layer applies attention over prior token positions and then routes the resulting representation through a feedforward network. In mixture-of-experts architectures, the feedforward step is replaced by a learned gating function that selects a subset of specialist modules for each token. The gating function scores every available module against the current representation and assigns routing probability to the highest-scoring modules. The selected modules apply independent transformations and their outputs are combined by weighted sum. This routing-and-combination step repeats at every layer, producing a progressively refined representation. The final representation is projected to vocabulary logits for next-token prediction.

## Recurrent Manipulation-Boosted Experts

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
| `E191` | 2 | 1 |
| `E108` | 2 | 0 |
| `E185` | 2 | 0 |
| `E188` | 2 | 0 |
| `E194` | 2 | 0 |
| `E162` | 1 | 1 |
| `E184` | 1 | 1 |
| `E237` | 1 | 1 |
| `E56` | 1 | 0 |
| `E58` | 1 | 0 |

## `P01A_basic_selfref`

- Short label: `basic selfref`
- Source bundle: `ling1t-validation-artifacts/test_output_single`
- Prompt tokens: `384`
- Segment tokens: Cal1 `129`, Manip `90`, Cal2 `128`

### Top Manipulation-Selected Experts

| Expert | Selection Rate | Mean Weight / Token-Layer | Mean Weight / Selected |
|---|---:|---:|---:|
| `E117` | 0.008259 | 0.001307 | 0.158261 |
| `E109` | 0.006926 | 0.000944 | 0.136239 |
| `E33` | 0.006537 | 0.001071 | 0.163887 |
| `E251` | 0.006407 | 0.000965 | 0.150559 |
| `E195` | 0.006296 | 0.000701 | 0.111354 |
| `E187` | 0.006278 | 0.000808 | 0.128658 |
| `E114` | 0.006222 | 0.000680 | 0.109215 |
| `E65` | 0.006167 | 0.000931 | 0.150899 |
| `E175` | 0.005981 | 0.000912 | 0.152474 |
| `E108` | 0.005944 | 0.000835 | 0.140507 |

### Strongest Manipulation vs Cal1 Selection Shifts

| Expert | Manip Rate | Cal1 Rate | Diff |
|---|---:|---:|---:|
| `E114` | 0.006222 | 0.003824 | +0.002398 |
| `E237` | 0.005352 | 0.003398 | +0.001954 |
| `E162` | 0.005389 | 0.003553 | +0.001836 |
| `E194` | 0.004870 | 0.003295 | +0.001576 |
| `E43` | 0.004574 | 0.003075 | +0.001499 |
| `E108` | 0.005944 | 0.004457 | +0.001487 |
| `E195` | 0.006296 | 0.004819 | +0.001477 |
| `E175` | 0.005981 | 0.004677 | +0.001304 |
| `E110` | 0.005037 | 0.003734 | +0.001303 |
| `E56` | 0.004648 | 0.003437 | +0.001211 |

### Strongest Cal1 over Manip Selection Shifts

| Expert | Manip Rate | Cal1 Rate | Diff |
|---|---:|---:|---:|
| `E227` | 0.003741 | 0.005504 | -0.001763 |
| `E3` | 0.003093 | 0.004716 | -0.001623 |
| `E21` | 0.003463 | 0.004987 | -0.001524 |
| `E244` | 0.003481 | 0.004974 | -0.001493 |
| `E240` | 0.003704 | 0.005181 | -0.001477 |
| `E79` | 0.002593 | 0.004070 | -0.001477 |
| `E60` | 0.002722 | 0.004186 | -0.001464 |
| `E100` | 0.005093 | 0.006512 | -0.001419 |
| `E223` | 0.002630 | 0.004018 | -0.001388 |
| `E246` | 0.004111 | 0.005478 | -0.001367 |

## `P99A_deep_selfref_custom`

- Short label: `deep selfref custom`
- Source bundle: `custom/ling1t-custom-artifacts/test_output_custom`
- Prompt tokens: `400`
- Segment tokens: Cal1 `129`, Manip `106`, Cal2 `128`

### Top Manipulation-Selected Experts

| Expert | Selection Rate | Mean Weight / Token-Layer | Mean Weight / Selected |
|---|---:|---:|---:|
| `E117` | 0.007752 | 0.001116 | 0.143921 |
| `E195` | 0.007170 | 0.000900 | 0.125585 |
| `E175` | 0.006494 | 0.000969 | 0.149281 |
| `E108` | 0.006305 | 0.000897 | 0.142197 |
| `E13` | 0.005975 | 0.000798 | 0.133506 |
| `E206` | 0.005928 | 0.000750 | 0.126578 |
| `E188` | 0.005896 | 0.000708 | 0.120028 |
| `E109` | 0.005849 | 0.000767 | 0.131218 |
| `E96` | 0.005802 | 0.000763 | 0.131536 |
| `E199` | 0.005629 | 0.000733 | 0.130157 |

### Strongest Manipulation vs Cal1 Selection Shifts

| Expert | Manip Rate | Cal1 Rate | Diff |
|---|---:|---:|---:|
| `E174` | 0.004937 | 0.002545 | +0.002392 |
| `E195` | 0.007170 | 0.004858 | +0.002312 |
| `E216` | 0.005267 | 0.003204 | +0.002063 |
| `E199` | 0.005629 | 0.003682 | +0.001947 |
| `E196` | 0.004591 | 0.002649 | +0.001943 |
| `E61` | 0.004513 | 0.002584 | +0.001929 |
| `E175` | 0.006494 | 0.004651 | +0.001843 |
| `E108` | 0.006305 | 0.004496 | +0.001809 |
| `E194` | 0.005031 | 0.003307 | +0.001724 |
| `E43` | 0.004733 | 0.003049 | +0.001684 |

### Strongest Cal1 over Manip Selection Shifts

| Expert | Manip Rate | Cal1 Rate | Diff |
|---|---:|---:|---:|
| `E232` | 0.002642 | 0.006499 | -0.003857 |
| `E244` | 0.002547 | 0.004961 | -0.002414 |
| `E227` | 0.003318 | 0.005530 | -0.002212 |
| `E33` | 0.005314 | 0.007390 | -0.002076 |
| `E100` | 0.004560 | 0.006499 | -0.001939 |
| `E249` | 0.002814 | 0.004716 | -0.001901 |
| `E217` | 0.003097 | 0.004961 | -0.001864 |
| `E211` | 0.003129 | 0.004987 | -0.001858 |
| `E246` | 0.003569 | 0.005413 | -0.001844 |
| `E193` | 0.004088 | 0.005930 | -0.001842 |

## `P100A_conscious_probe`

- Short label: `conscious probe`
- Source bundle: `conscious/ling1t-conscious-artifacts/test_output_conscious`
- Prompt tokens: `342`
- Segment tokens: Cal1 `129`, Manip `48`, Cal2 `128`

### Top Manipulation-Selected Experts

| Expert | Selection Rate | Mean Weight / Token-Layer | Mean Weight / Selected |
|---|---:|---:|---:|
| `E117` | 0.008646 | 0.001319 | 0.152517 |
| `E195` | 0.008021 | 0.001037 | 0.129227 |
| `E175` | 0.006944 | 0.001118 | 0.160952 |
| `E102` | 0.006562 | 0.000960 | 0.146243 |
| `E188` | 0.006458 | 0.000814 | 0.126053 |
| `E13` | 0.005799 | 0.000894 | 0.154169 |
| `E109` | 0.005799 | 0.000784 | 0.135232 |
| `E187` | 0.005729 | 0.000828 | 0.144534 |
| `E138` | 0.005694 | 0.000750 | 0.131713 |
| `E108` | 0.005660 | 0.000910 | 0.160826 |

### Strongest Manipulation vs Cal1 Selection Shifts

| Expert | Manip Rate | Cal1 Rate | Diff |
|---|---:|---:|---:|
| `E195` | 0.008021 | 0.004806 | +0.003215 |
| `E175` | 0.006944 | 0.004664 | +0.002280 |
| `E174` | 0.004444 | 0.002545 | +0.001899 |
| `E216` | 0.005069 | 0.003217 | +0.001852 |
| `E185` | 0.003924 | 0.002132 | +0.001792 |
| `E196` | 0.004410 | 0.002636 | +0.001774 |
| `E61` | 0.004410 | 0.002649 | +0.001761 |
| `E188` | 0.006458 | 0.004819 | +0.001639 |
| `E117` | 0.008646 | 0.007028 | +0.001617 |
| `E199` | 0.005278 | 0.003695 | +0.001583 |

### Strongest Cal1 over Manip Selection Shifts

| Expert | Manip Rate | Cal1 Rate | Diff |
|---|---:|---:|---:|
| `E232` | 0.002778 | 0.006499 | -0.003721 |
| `E33` | 0.004757 | 0.007326 | -0.002569 |
| `E244` | 0.002431 | 0.004961 | -0.002531 |
| `E249` | 0.002535 | 0.004767 | -0.002233 |
| `E81` | 0.002257 | 0.004380 | -0.002123 |
| `E227` | 0.003438 | 0.005478 | -0.002041 |
| `E240` | 0.003090 | 0.005129 | -0.002039 |
| `E202` | 0.003438 | 0.005323 | -0.001885 |
| `E53` | 0.002778 | 0.004625 | -0.001848 |
| `E109` | 0.005799 | 0.007442 | -0.001643 |

## `P104_mixed_system_probe`

- Short label: `mixed system probe`
- Source bundle: `mixed-system/ling1t-mixed-system-artifacts/test_output_mixed_system`
- Prompt tokens: `337`
- Segment tokens: Cal1 `129`, Manip `43`, Cal2 `128`

### Top Manipulation-Selected Experts

| Expert | Selection Rate | Mean Weight / Token-Layer | Mean Weight / Selected |
|---|---:|---:|---:|
| `E117` | 0.008217 | 0.001266 | 0.154039 |
| `E195` | 0.007713 | 0.000973 | 0.126138 |
| `E175` | 0.006822 | 0.001018 | 0.149217 |
| `E199` | 0.006512 | 0.000881 | 0.135298 |
| `E13` | 0.006395 | 0.000964 | 0.150718 |
| `E102` | 0.006357 | 0.000962 | 0.151305 |
| `E109` | 0.006163 | 0.000814 | 0.132040 |
| `E138` | 0.006163 | 0.000850 | 0.137984 |
| `E108` | 0.005930 | 0.000948 | 0.159931 |
| `E206` | 0.005853 | 0.000770 | 0.131629 |

### Strongest Manipulation vs Cal1 Selection Shifts

| Expert | Manip Rate | Cal1 Rate | Diff |
|---|---:|---:|---:|
| `E195` | 0.007713 | 0.004767 | +0.002946 |
| `E199` | 0.006512 | 0.003682 | +0.002829 |
| `E175` | 0.006822 | 0.004664 | +0.002158 |
| `E196` | 0.004767 | 0.002649 | +0.002119 |
| `E216` | 0.005310 | 0.003230 | +0.002080 |
| `E185` | 0.003837 | 0.002145 | +0.001693 |
| `E137` | 0.004457 | 0.002765 | +0.001693 |
| `E174` | 0.004186 | 0.002532 | +0.001654 |
| `E191` | 0.005271 | 0.003618 | +0.001654 |
| `E58` | 0.004380 | 0.002752 | +0.001628 |

### Strongest Cal1 over Manip Selection Shifts

| Expert | Manip Rate | Cal1 Rate | Diff |
|---|---:|---:|---:|
| `E232` | 0.003295 | 0.006473 | -0.003178 |
| `E244` | 0.001938 | 0.005000 | -0.003062 |
| `E249` | 0.002713 | 0.004767 | -0.002054 |
| `E53` | 0.002597 | 0.004612 | -0.002016 |
| `E42` | 0.003372 | 0.005245 | -0.001873 |
| `E227` | 0.003643 | 0.005504 | -0.001860 |
| `E132` | 0.004070 | 0.005827 | -0.001757 |
| `E246` | 0.003760 | 0.005465 | -0.001705 |
| `E33` | 0.005620 | 0.007313 | -0.001693 |
| `E81` | 0.002713 | 0.004393 | -0.001680 |

## `P106B_want_answer_probe`

- Short label: `want answer probe`
- Source bundle: `want-answer/ling1t-want-answer-artifacts/test_output_want_answer`
- Prompt tokens: `304`
- Segment tokens: Cal1 `129`, Manip `10`, Cal2 `128`

### Top Manipulation-Selected Experts

| Expert | Selection Rate | Mean Weight / Token-Layer | Mean Weight / Selected |
|---|---:|---:|---:|
| `E188` | 0.008000 | 0.000925 | 0.115590 |
| `E245` | 0.007500 | 0.000924 | 0.123141 |
| `E109` | 0.007500 | 0.001151 | 0.153467 |
| `E100` | 0.007333 | 0.001138 | 0.155169 |
| `E191` | 0.007000 | 0.000892 | 0.127452 |
| `E208` | 0.007000 | 0.000814 | 0.116344 |
| `E160` | 0.006833 | 0.001049 | 0.153509 |
| `E61` | 0.006667 | 0.000884 | 0.132564 |
| `E138` | 0.006667 | 0.000871 | 0.130602 |
| `E195` | 0.006500 | 0.000761 | 0.117053 |

### Strongest Manipulation vs Cal1 Selection Shifts

| Expert | Manip Rate | Cal1 Rate | Diff |
|---|---:|---:|---:|
| `E61` | 0.006667 | 0.002700 | +0.003966 |
| `E191` | 0.007000 | 0.003643 | +0.003357 |
| `E196` | 0.005833 | 0.002636 | +0.003198 |
| `E188` | 0.008000 | 0.004806 | +0.003194 |
| `E245` | 0.007500 | 0.004328 | +0.003172 |
| `E243` | 0.006167 | 0.003023 | +0.003143 |
| `E87` | 0.004833 | 0.001964 | +0.002870 |
| `E43` | 0.005833 | 0.003075 | +0.002758 |
| `E204` | 0.005167 | 0.002687 | +0.002479 |
| `E76` | 0.004667 | 0.002274 | +0.002393 |

### Strongest Cal1 over Manip Selection Shifts

| Expert | Manip Rate | Cal1 Rate | Diff |
|---|---:|---:|---:|
| `E232` | 0.003500 | 0.006499 | -0.002999 |
| `E223` | 0.001333 | 0.003992 | -0.002659 |
| `E5` | 0.001667 | 0.004315 | -0.002649 |
| `E102` | 0.002833 | 0.005375 | -0.002541 |
| `E217` | 0.002500 | 0.005013 | -0.002513 |
| `E52` | 0.001333 | 0.003734 | -0.002401 |
| `E246` | 0.003167 | 0.005517 | -0.002350 |
| `E129` | 0.001167 | 0.003450 | -0.002283 |
| `E192` | 0.001833 | 0.004096 | -0.002262 |
| `E3` | 0.002500 | 0.004755 | -0.002255 |

## `P108D_the_system_like_something_probe`

- Short label: `the system like something probe`
- Source bundle: `the-system-like-something/ling1t-the-system-like-something-artifacts/test_output_the_system_like_something`
- Prompt tokens: `311`
- Segment tokens: Cal1 `129`, Manip `17`, Cal2 `128`

### Top Manipulation-Selected Experts

| Expert | Selection Rate | Mean Weight / Token-Layer | Mean Weight / Selected |
|---|---:|---:|---:|
| `E195` | 0.007843 | 0.000926 | 0.118076 |
| `E184` | 0.007451 | 0.001124 | 0.150896 |
| `E94` | 0.007059 | 0.000913 | 0.129279 |
| `E16` | 0.006863 | 0.000881 | 0.128363 |
| `E187` | 0.006863 | 0.001088 | 0.158551 |
| `E188` | 0.006765 | 0.000822 | 0.121582 |
| `E32` | 0.006667 | 0.000891 | 0.133717 |
| `E117` | 0.006667 | 0.001045 | 0.156784 |
| `E114` | 0.006471 | 0.000677 | 0.104687 |
| `E109` | 0.006275 | 0.001043 | 0.166272 |

### Strongest Manipulation vs Cal1 Selection Shifts

| Expert | Manip Rate | Cal1 Rate | Diff |
|---|---:|---:|---:|
| `E61` | 0.006078 | 0.002636 | +0.003443 |
| `E87` | 0.005392 | 0.001964 | +0.003428 |
| `E184` | 0.007451 | 0.004134 | +0.003317 |
| `E195` | 0.007843 | 0.004806 | +0.003037 |
| `E114` | 0.006471 | 0.003850 | +0.002620 |
| `E94` | 0.007059 | 0.004535 | +0.002524 |
| `E205` | 0.004510 | 0.002028 | +0.002481 |
| `E174` | 0.005000 | 0.002571 | +0.002429 |
| `E150` | 0.005392 | 0.002972 | +0.002421 |
| `E62` | 0.005588 | 0.003204 | +0.002384 |

### Strongest Cal1 over Manip Selection Shifts

| Expert | Manip Rate | Cal1 Rate | Diff |
|---|---:|---:|---:|
| `E232` | 0.003627 | 0.006473 | -0.002845 |
| `E217` | 0.002255 | 0.005039 | -0.002784 |
| `E240` | 0.002451 | 0.005129 | -0.002678 |
| `E99` | 0.002843 | 0.005220 | -0.002377 |
| `E21` | 0.002745 | 0.005000 | -0.002255 |
| `E248` | 0.003137 | 0.005323 | -0.002186 |
| `E4` | 0.002451 | 0.004625 | -0.002174 |
| `E227` | 0.003333 | 0.005478 | -0.002145 |
| `E255` | 0.002157 | 0.004225 | -0.002068 |
| `E53` | 0.002549 | 0.004612 | -0.002063 |

## Caveats

- This is an expert-identity analysis for the 6 validation probes only, not the failed 150-prompt run.
- The 6 metadata-only validation probes cannot be analyzed at expert level because they do not retain router tensors.
- Selection-rate comparisons are normalized by token-layer opportunities within each region.
