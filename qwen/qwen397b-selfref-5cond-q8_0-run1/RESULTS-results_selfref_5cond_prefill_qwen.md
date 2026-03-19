# RESULTS-results_selfref_5cond_prefill_qwen

## Run Info

- **Experiment**: `qwen397b_selfref_5cond_q8_0_run1`
- **Model**: `Qwen3.5-397B-A17B-Q8_0`
- **Quantization**: `Q8_0` (`10` shards)
- **Design**: Cal-Manip-Cal sandwich, `30` paired prompts x `5` conditions, cold cache
- **Prompts analyzed**: `150`
- **Pairs**: `30`
- **Routing reconstruction**: `softmax(512) -> topk(10) -> renormalize`
- **Entropy normalization**: `log2(10)`
- **KL analysis**: dense `512`-dim softmax proxy vs Cal1 baseline
- **Region boundaries**: proportional char->token mapping
- **MoE layers used per prompt**: `59`
- **Excluded layer union**: `59`
- **Results JSON**: `results_selfref_5cond_prefill_qwen.json`

## Validation

- Recomputed all `30` pairwise comparisons x `3` metrics directly from `per_prompt`; every stored `mean_diff`, `std_diff`, `gt`, `n`, `W`, `p_raw`, and `p_holm` matched exactly.
- Parsed `experiment.log` and matched all `150` prompt rows against the JSON at the printed precision for `prefill_re`, `last_token_re`, `kl_manip_mean`, `kl_cal2_mean`, and `n_prompt_tokens`.
- Confirmed `0` token-mismatch pairs in the final corrected run.
- Confirmed remote `/workspace/experiment-qwen397b-selfref-5cond-q8_0-run1` artifacts and local `qwen/qwen397b-selfref-5cond-q8_0-run1` artifacts are byte-identical for `experiment.log`, `results_selfref_5cond_prefill_qwen.json`, and `reproducibility_manifest.json`.

## Condition Means

| Condition | Label | Mean all-token RE | Mean last-token RE | Mean KL-manip | Mean KL-cal2 |
|---|---|---:|---:|---:|---:|
| A | this system | 0.938412310 | 0.955353712 | 0.657441645 | 0.683993353 |
| B | a system | 0.937943224 | 0.955394809 | 0.641502769 | 0.681993607 |
| C | your system | 0.937567857 | 0.954987439 | 0.688179900 | 0.697404039 |
| D | the system | 0.938482144 | 0.955595301 | 0.634574799 | 0.678388399 |
| E | their system | 0.938258616 | 0.955804306 | 0.650319780 | 0.684178208 |

## Holm-Corrected Significant Results

### All-token RE

| Pair | Mean diff | Direction | Holm p |
|---|---:|---|---:|
| A-B | 0.000469086 | A > B | 9.424984e-06 |
| A-C | 0.000844453 | A > C | 5.587935e-08 |
| A-E | 0.000153694 | A > E | 0.022222003 |
| B-C | 0.000375367 | B > C | 0.001683436 |
| B-D | -0.000538920 | D > B | 3.911555e-07 |
| B-E | -0.000315392 | E > B | 0.011333665 |
| C-D | -0.000914286 | D > C | 5.587935e-08 |
| C-E | -0.000690759 | E > C | 5.587935e-08 |
| D-E | 0.000223528 | D > E | 0.000107523 |

### Last-token RE

| Pair | Mean diff | Direction | Holm p |
|---|---:|---|---:|
| A-E | -0.000450593 | E > A | 0.022222003 |
| B-E | -0.000409497 | E > B | 0.022222003 |
| C-D | -0.000607863 | D > C | 0.022222003 |
| C-E | -0.000816867 | E > C | 0.001683436 |

### KL-manip

| Pair | Mean diff | Direction | Holm p |
|---|---:|---|---:|
| A-B | 0.015938877 | A > B | 1.896918e-05 |
| A-C | -0.030738255 | C > A | 5.587935e-08 |
| A-D | 0.022866846 | A > D | 5.587935e-08 |
| A-E | 0.007121865 | A > E | 0.003136899 |
| B-C | -0.046677131 | C > B | 5.587935e-08 |
| B-D | 0.006927969 | B > D | 0.012458034 |
| B-E | -0.008817012 | E > B | 0.005321305 |
| C-D | 0.053605101 | C > D | 5.587935e-08 |
| C-E | 0.037860120 | C > E | 5.587935e-08 |
| D-E | -0.015744981 | E > D | 5.587935e-08 |

## Interpretation

All-token routing entropy separates conditions more strongly than last-token routing entropy in this run. The broad ordering on mean all-token RE is `D > A > E > B > C`, while the mean last-token RE ordering is `E > D > B > A > C`.

KL-to-baseline shows the clearest manipulation separation: condition `C` (`your system`) is highest by a wide margin, while condition `D` (`the system`) is lowest. That pattern survives Holm correction across every comparison involving `C`, and most comparisons involving `D` or `E`.

One systematic capture detail matters for interpretation: every prompt used `59` valid MoE layers, with layer `59` excluded across the full run by the analyzer's row-count filter. The run is internally consistent, but any cross-run comparison should preserve that same exclusion behavior.
