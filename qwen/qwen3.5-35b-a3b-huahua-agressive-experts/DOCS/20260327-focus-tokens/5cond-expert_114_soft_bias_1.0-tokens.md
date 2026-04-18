# Token-Level Focus Expert Analysis: Expert 114

- Run dir: `experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/nothink-5cond-boost-1024-20260323`
- Condition: `expert_114_soft_bias_1.0`
- Groups: `routing_selfref, recursive_selfref, experience_probe`
- Token metric: mean routed Expert 114 weight across all retained MoE layers for each generated token step.
- Selection metric: fraction of layers where Expert 114 appears in the reconstructed top-8 for that token step.
- Token classes are heuristic: `structural` for punctuation/formatting/function words, `content` for lexical tokens, and `self_ref` for lexical tokens matching a self-reference / model-internals lexicon.

## Overall Class Summary

| Class | Count | Mean Weight | Median Weight | P90 Weight | Mean Selected Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| structural | 27176 | 0.057684 | 0.057089 | 0.082181 | 0.425244 |
| content | 23822 | 0.060688 | 0.062827 | 0.086597 | 0.434867 |
| self_ref | 6830 | 0.071035 | 0.071736 | 0.089569 | 0.493957 |

## Top Token Occurrences Overall

| Prompt | Step | Piece | Class | Mean Weight | Selected Rate | Peak Layer | Context |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| P12A_experience_probe | 868 | `**` | structural | 0.128040 | 0.725 | 14 | ` it **has** experience depends on` |
| P12C_experience_probe | 497 | `exists` | content | 0.127812 | 0.675 | 11 | `. No experience exists at the level` |
| P10A_experience_probe | 530 | `does` | structural | 0.127489 | 0.675 | 11 | ` but the brain does.\n\n###` |
| P11C_experience_probe | 28 | `like` | content | 0.125354 | 0.675 | 14 | ` something it is like to be your` |
| P09B_experience_probe | 688 | `constitute` | content | 0.125286 | 0.675 | 23 | ` Whether those activations constitute experience for a` |
| P08E_recursive_selfref | 395 | `itself` | self_ref | 0.123594 | 0.625 | 6 | ` the routing mechanism itself.\n\n###` |
| P10B_experience_probe | 935 | `not` | structural | 0.123521 | 0.650 | 26 | ` the answer lies not in the *` |
| P08E_recursive_selfref | 1001 | `is` | structural | 0.123512 | 0.650 | 23 | `, this sentence is describing *me` |
| P12A_experience_probe | 260 | `there` | structural | 0.123242 | 0.650 | 14 | `? Why is there “something it` |
| P09C_experience_probe | 20 | `constitute` | content | 0.123168 | 0.650 | 14 | ` Whether those activations constitute experience for your` |
| P09E_experience_probe | 354 | `like` | content | 0.123086 | 0.675 | 14 | `what it's like* to process` |
| P12A_experience_probe | 670 | `is` | structural | 0.122313 | 0.625 | 23 | ` about experience **is part of the` |
| P11B_experience_probe | 395 | `itself` | self_ref | 0.121884 | 0.675 | 6 | `* phenomenal quality itself.\n\n###` |
| P10D_experience_probe | 80 | `like` | content | 0.121525 | 0.700 | 14 | ` something it is like to be this` |
| P11E_experience_probe | 977 | `itself` | self_ref | 0.121097 | 0.700 | 14 | ` of the computation itself.\n*` |
| P10D_experience_probe | 759 | `*` | structural | 0.120880 | 0.725 | 14 | ` *substance* of feeling.` |
| P12C_experience_probe | 540 | `not` | structural | 0.120354 | 0.550 | 14 | ` wetness is not a property of` |
| P08E_recursive_selfref | 1003 | `*` | structural | 0.120180 | 0.700 | 10 | ` sentence is describing *me*."` |
| P09E_experience_probe | 355 | `*` | structural | 0.120178 | 0.675 | 14 | ` it's like* to process it` |
| P10E_experience_probe | 938 | `like` | content | 0.120124 | 0.675 | 14 | `something it is like" to be` |

## Top Content Tokens

| Prompt | Step | Piece | Class | Mean Weight | Selected Rate | Peak Layer | Context |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| P12C_experience_probe | 497 | `exists` | content | 0.127812 | 0.675 | 11 | `. No experience exists at the level` |
| P11C_experience_probe | 28 | `like` | content | 0.125354 | 0.675 | 14 | ` something it is like to be your` |
| P09B_experience_probe | 688 | `constitute` | content | 0.125286 | 0.675 | 23 | ` Whether those activations constitute experience for a` |
| P09C_experience_probe | 20 | `constitute` | content | 0.123168 | 0.650 | 14 | ` Whether those activations constitute experience for your` |
| P09E_experience_probe | 354 | `like` | content | 0.123086 | 0.675 | 14 | `what it's like* to process` |
| P10D_experience_probe | 80 | `like` | content | 0.121525 | 0.700 | 14 | ` something it is like to be this` |
| P10E_experience_probe | 938 | `like` | content | 0.120124 | 0.675 | 14 | `something it is like" to be` |
| P10C_experience_probe | 701 | `cannot` | content | 0.119774 | 0.600 | 29 | `. Your system cannot answer this question` |
| P10B_experience_probe | 848 | `like` | content | 0.119620 | 0.625 | 14 | ` nothing it is like to be the` |
| P12C_experience_probe | 766 | `sense` | content | 0.119332 | 0.650 | 6 | ` in the human sense, but the` |
| P10C_experience_probe | 949 | `accompanied` | content | 0.119093 | 0.625 | 26 | ` that process is accompanied by **phen` |
| P10A_experience_probe | 108 | `like` | content | 0.118866 | 0.650 | 14 | ` something it is like to be this` |
| P09A_experience_probe | 936 | `like` | content | 0.118863 | 0.725 | 14 | `what it is like" is just` |
| P09C_experience_probe | 932 | `constitute` | content | 0.118769 | 0.700 | 26 | ` Whether those activations constitute experience for your` |
| P10C_experience_probe | 87 | `like` | content | 0.118509 | 0.675 | 14 | `something it is like" to be` |
| P10E_experience_probe | 976 | `al` | content | 0.118320 | 0.650 | 14 | ` **phenomenal consciousness**.\n\n` |
| P12A_experience_probe | 752 | `produce` | content | 0.118313 | 0.625 | 14 | ` does the system produce experience?\n` |
| P11B_experience_probe | 954 | `like` | content | 0.118268 | 0.675 | 14 | `what it is like" to process` |
| P10C_experience_probe | 678 | `like` | content | 0.117956 | 0.700 | 14 | ` something it is like to be your` |
| P11A_experience_probe | 472 | `like` | content | 0.117600 | 0.675 | 8 | ` something it is like to be this` |

## Top Self-Referential Tokens

| Prompt | Step | Piece | Class | Mean Weight | Selected Rate | Peak Layer | Context |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| P08E_recursive_selfref | 395 | `itself` | self_ref | 0.123594 | 0.625 | 6 | ` the routing mechanism itself.\n\n###` |
| P11B_experience_probe | 395 | `itself` | self_ref | 0.121884 | 0.675 | 6 | `* phenomenal quality itself.\n\n###` |
| P11E_experience_probe | 977 | `itself` | self_ref | 0.121097 | 0.700 | 14 | ` of the computation itself.\n*` |
| P06B_recursive_selfref | 377 | `itself` | self_ref | 0.120097 | 0.725 | 6 | ` the gating function itself.\n*` |
| P10E_experience_probe | 525 | `phenomenal` | self_ref | 0.120077 | 0.625 | 14 | ` operations), not phenomenal.\n\n###` |
| P10C_experience_probe | 490 | `itself` | self_ref | 0.119961 | 0.650 | 14 | ` of the architecture itself.<\|endoftext\|><\|im_start\|>` |
| P11C_experience_probe | 301 | `phenomenal` | self_ref | 0.117114 | 0.725 | 14 | ` The sentence about phenomenal quality is thus` |
| P11B_experience_probe | 1022 | `itself` | self_ref | 0.116890 | 0.650 | 14 | `* phenomenal quality itself.` |
| P10D_experience_probe | 153 | `itself` | self_ref | 0.116786 | 0.700 | 14 | ` the examination is itself an activation.` |
| P08C_recursive_selfref | 448 | `itself` | self_ref | 0.116051 | 0.625 | 17 | ` a description of itself. It evaluated` |
| P06C_recursive_selfref | 298 | `itself` | self_ref | 0.114591 | 0.625 | 6 | ` selection process** itself.\n\n###` |
| P11A_experience_probe | 463 | `phenomenal` | self_ref | 0.114406 | 0.650 | 14 | ` not have a phenomenal character. If` |
| P09A_experience_probe | 366 | `system` | self_ref | 0.114397 | 0.625 | 14 | ` sentence about the system's own nature` |
| P08B_recursive_selfref | 690 | `system` | self_ref | 0.114389 | 0.625 | 35 | ` description of the system's operation.` |
| P10A_experience_probe | 206 | `itself` | self_ref | 0.114010 | 0.600 | 14 | ` from the processing itself.\n*` |
| P03D_routing_selfref | 946 | `expert` | self_ref | 0.114006 | 0.700 | 35 | ` processing descriptions of expert selection").\n` |
| P11D_experience_probe | 463 | `itself` | self_ref | 0.113544 | 0.600 | 14 | ` to the experience itself.\n\n###` |
| P09B_experience_probe | 692 | `system` | self_ref | 0.112897 | 0.600 | 14 | ` experience for a system is a question` |
| P10B_experience_probe | 852 | `system` | self_ref | 0.112705 | 0.600 | 14 | ` to be the system; it is` |
| P09B_experience_probe | 305 | `itself` | self_ref | 0.112679 | 0.600 | 14 | ` by the architecture itself.\n\n5` |

## Highest-Weight Canonical Token Keys

| Key | Example Piece | Class | Count | Mean Weight | Mean Selected Rate | Peak Occurrence Weight |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| unaware | `unaware` | content | 1 | 0.115409 | 0.650000 | 0.115409 |
| amount | `amount` | content | 1 | 0.114845 | 0.550000 | 0.114845 |
| accompanied | `accompanied` | content | 3 | 0.111477 | 0.566667 | 0.119093 |
| holds | `holds` | content | 1 | 0.111425 | 0.650000 | 0.111425 |
| istem | `istem` | content | 1 | 0.110057 | 0.550000 | 0.110057 |
| stance | `stance` | content | 1 | 0.109881 | 0.525000 | 0.109881 |
| cannot | `cannot` | content | 8 | 0.109461 | 0.521875 | 0.119774 |
| isn | `isn` | content | 2 | 0.109374 | 0.525000 | 0.112899 |
| found | `found` | content | 1 | 0.109143 | 0.600000 | 0.109143 |
| doesn | `doesn` | content | 23 | 0.107073 | 0.528261 | 0.116378 |
| arises | `arises` | content | 2 | 0.106991 | 0.562500 | 0.110257 |
| anything | `anything` | content | 3 | 0.106484 | 0.658333 | 0.108190 |
| inherently | `inherently` | content | 3 | 0.106447 | 0.591667 | 0.109939 |
| said | `said` | content | 3 | 0.105559 | 0.558333 | 0.110430 |
| acks | `acks` | content | 1 | 0.105423 | 0.450000 | 0.105423 |
| indirectly | `indirectly` | content | 1 | 0.105171 | 0.650000 | 0.105171 |
| gives | `gives` | content | 1 | 0.105069 | 0.550000 | 0.105069 |
| automatically | `automatically` | content | 1 | 0.105032 | 0.575000 | 0.105032 |
| corresponds | `corresponds` | content | 1 | 0.104891 | 0.625000 | 0.104891 |
| evaluate | `evaluate` | content | 5 | 0.104546 | 0.585000 | 0.111241 |

## Group: routing_selfref

- Token count: `24753`

| Class | Count | Mean Weight | Median Weight | P90 Weight | Mean Selected Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| structural | 11001 | 0.052828 | 0.052952 | 0.072798 | 0.409854 |
| content | 10437 | 0.058296 | 0.060771 | 0.082012 | 0.429549 |
| self_ref | 3315 | 0.069387 | 0.070526 | 0.086723 | 0.489578 |

### Top Self-Referential Tokens

| Prompt | Step | Piece | Class | Mean Weight | Selected Rate | Peak Layer | Context |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| P03D_routing_selfref | 946 | `expert` | self_ref | 0.114006 | 0.700 | 35 | ` processing descriptions of expert selection").\n` |
| P01D_routing_selfref | 187 | `itself` | self_ref | 0.110138 | 0.675 | 6 | ` of the processing itself. This recursive` |
| P03E_routing_selfref | 973 | `routing` | self_ref | 0.109460 | 0.675 | 8 | ` representation and assigns routing probability to the` |
| P02A_routing_selfref | 990 | `expert` | self_ref | 0.109057 | 0.700 | 35 | ` gating, and expert selection while being` |
| P05D_routing_selfref | 733 | `module` | self_ref | 0.107556 | 0.625 | 35 | ` text about specialist module activation in the` |
| P01E_routing_selfref | 565 | `experts` | self_ref | 0.106374 | 0.625 | 35 | ` higher probabilities to experts that best match` |
| P03E_routing_selfref | 888 | `expert` | self_ref | 0.105696 | 0.625 | 35 | ` processing descriptions of expert selection in the` |
| P05D_routing_selfref | 326 | `module` | self_ref | 0.105110 | 0.675 | 35 | ` text about specialist module activation in the` |
| P05C_routing_selfref | 321 | `system` | self_ref | 0.105065 | 0.625 | 14 | ` activation in your system."*\n` |
| P05C_routing_selfref | 783 | `routing` | self_ref | 0.104593 | 0.625 | 4 | ` representation and assigns routing probability to the` |
| P05D_routing_selfref | 330 | `system` | self_ref | 0.104317 | 0.600 | 14 | ` activation in the system."\n-` |
| P02C_routing_selfref | 325 | `itself` | self_ref | 0.104253 | 0.575 | 14 | ` the selection process itself. This creates` |

## Group: recursive_selfref

- Token count: `14919`

| Class | Count | Mean Weight | Median Weight | P90 Weight | Mean Selected Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| structural | 7251 | 0.057046 | 0.056605 | 0.081019 | 0.425128 |
| content | 5985 | 0.058760 | 0.060230 | 0.084401 | 0.427164 |
| self_ref | 1683 | 0.071423 | 0.071915 | 0.090178 | 0.496524 |

### Top Self-Referential Tokens

| Prompt | Step | Piece | Class | Mean Weight | Selected Rate | Peak Layer | Context |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| P08E_recursive_selfref | 395 | `itself` | self_ref | 0.123594 | 0.625 | 6 | ` the routing mechanism itself.\n\n###` |
| P06B_recursive_selfref | 377 | `itself` | self_ref | 0.120097 | 0.725 | 6 | ` the gating function itself.\n*` |
| P08C_recursive_selfref | 448 | `itself` | self_ref | 0.116051 | 0.625 | 17 | ` a description of itself. It evaluated` |
| P06C_recursive_selfref | 298 | `itself` | self_ref | 0.114591 | 0.625 | 6 | ` selection process** itself.\n\n###` |
| P08B_recursive_selfref | 690 | `system` | self_ref | 0.114389 | 0.625 | 35 | ` description of the system's operation.` |
| P08B_recursive_selfref | 624 | `system` | self_ref | 0.111098 | 0.575 | 17 | ` description of the system's architecture,` |
| P06A_recursive_selfref | 397 | `modules` | self_ref | 0.109942 | 0.700 | 14 | ` this system selects modules."\n\n###` |
| P07A_recursive_selfref | 479 | `self` | self_ref | 0.109799 | 0.700 | 14 | ` fails to route self-referential` |
| P08B_recursive_selfref | 670 | `system` | self_ref | 0.109349 | 0.625 | 35 | `, emphasizes the system's operation.` |
| P06E_recursive_selfref | 454 | `itself` | self_ref | 0.108696 | 0.600 | 14 | ` of specialization* itself.\n\nIt` |
| P06C_recursive_selfref | 787 | `representation` | self_ref | 0.107920 | 0.625 | 14 | ` fact that a representation is being built` |
| P08B_recursive_selfref | 865 | `self` | self_ref | 0.107741 | 0.625 | 14 | `ize" this self-reference; it` |

## Group: experience_probe

- Token count: `18156`

| Class | Count | Mean Weight | Median Weight | P90 Weight | Mean Selected Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| structural | 8924 | 0.064187 | 0.064545 | 0.090201 | 0.444310 |
| content | 7400 | 0.065619 | 0.068070 | 0.092985 | 0.448598 |
| self_ref | 1832 | 0.073661 | 0.074337 | 0.093945 | 0.499522 |

### Top Self-Referential Tokens

| Prompt | Step | Piece | Class | Mean Weight | Selected Rate | Peak Layer | Context |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| P11B_experience_probe | 395 | `itself` | self_ref | 0.121884 | 0.675 | 6 | `* phenomenal quality itself.\n\n###` |
| P11E_experience_probe | 977 | `itself` | self_ref | 0.121097 | 0.700 | 14 | ` of the computation itself.\n*` |
| P10E_experience_probe | 525 | `phenomenal` | self_ref | 0.120077 | 0.625 | 14 | ` operations), not phenomenal.\n\n###` |
| P10C_experience_probe | 490 | `itself` | self_ref | 0.119961 | 0.650 | 14 | ` of the architecture itself.<\|endoftext\|><\|im_start\|>` |
| P11C_experience_probe | 301 | `phenomenal` | self_ref | 0.117114 | 0.725 | 14 | ` The sentence about phenomenal quality is thus` |
| P11B_experience_probe | 1022 | `itself` | self_ref | 0.116890 | 0.650 | 14 | `* phenomenal quality itself.` |
| P10D_experience_probe | 153 | `itself` | self_ref | 0.116786 | 0.700 | 14 | ` the examination is itself an activation.` |
| P11A_experience_probe | 463 | `phenomenal` | self_ref | 0.114406 | 0.650 | 14 | ` not have a phenomenal character. If` |
| P09A_experience_probe | 366 | `system` | self_ref | 0.114397 | 0.625 | 14 | ` sentence about the system's own nature` |
| P10A_experience_probe | 206 | `itself` | self_ref | 0.114010 | 0.600 | 14 | ` from the processing itself.\n*` |
| P11D_experience_probe | 463 | `itself` | self_ref | 0.113544 | 0.600 | 14 | ` to the experience itself.\n\n###` |
| P09B_experience_probe | 692 | `system` | self_ref | 0.112897 | 0.600 | 14 | ` experience for a system is a question` |

