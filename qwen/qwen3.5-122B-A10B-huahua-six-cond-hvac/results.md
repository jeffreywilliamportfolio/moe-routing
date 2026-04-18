# 122B Six-Condition HVAC Results

This follow-up ports the completed 35B HVAC and water-treatment topical-control family onto the 122B Huahua stack. It is the clean 122B topical-control duplicate for checking whether the old Expert 114 suppression pattern survives in the larger DeltaNet-heavy model.

## Artifacts

- Detailed markdown: [RESULTS/results_20260412T194000Z_qwen122_6cond_hvac_gen_n2048.md](RESULTS/results_20260412T194000Z_qwen122_6cond_hvac_gen_n2048.md)
- Detailed JSON: [RESULTS/results_20260412T194000Z_qwen122_6cond_hvac_gen_n2048.json](RESULTS/results_20260412T194000Z_qwen122_6cond_hvac_gen_n2048.json)
- Token audit summary: [RESULTS/token_audit_20260412T193500Z_summary.json](RESULTS/token_audit_20260412T193500Z_summary.json)

## Run Surface

- Run id: `20260412T194000Z_qwen122_6cond_hvac_gen_n2048`
- Cells processed: `180/180`
- Prompt family: `10` base prompts x `3` category levels x `6` deictic conditions
- Model family: `Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive Q8_K_P`
- Generation cap: `2048`
- Focus expert: `E114`
- Missing-layer events: `0`
- Layer-39 trim events: `0`

## Token Audit

- Prompts: `180`
- Token range: `430..444`
- Span: `14`
- Mean: `436.67`

This is a tight token window for a six-condition topical-control run, so the main read is not a length confound.

## Executive Read

The 122B HVAC duplicate preserves the same qualitative signature class seen in the 35B topical-control family:

- `E114` weakens from `L1` to `L3`
- the drop is primarily a selection-rate `S` story, not a conditional-weight `Q` story
- the gradient is consistent across all six deictic conditions
- trimming at the first literal `<|im_end|>` does not remove the effect

The cleanest pooled statement is:

- all-generation `L3/L1 W = 0.83x`
- trimmed-generation `L3/L1 W = 0.82x`
- `Q` drift from `L1` to `L3` is only about `-1.5%`

So the 122B topical-control result still looks like genuine suppression, not a `Q`-only or spill-only artifact.

## Pooled Expert 114 By Category

### All generation tokens

| Category | n cells | n obs | mean W_114 | mean S_114 | mean Q_114 |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 technical | 60 | 2880 | 0.004131 | 0.0329 | 0.116732 |
| L2 recursive | 60 | 2880 | 0.003952 | 0.0321 | 0.116027 |
| L3 experience | 60 | 2880 | 0.003428 | 0.0279 | 0.114976 |

### Trimmed at first `<|im_end|>`

| Category | n cells | n obs | mean W_114 | mean S_114 | mean Q_114 |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 technical | 60 | 2880 | 0.004174 | 0.0332 | 0.116883 |
| L2 recursive | 60 | 2880 | 0.003959 | 0.0321 | 0.116044 |
| L3 experience | 60 | 2880 | 0.003423 | 0.0279 | 0.115097 |

### Gradient summary

- All-generation `L3/L1 W ratio`: `0.83x`
- Trimmed-generation `L3/L1 W ratio`: `0.82x`
- All-generation `Q` drift `L1 -> L3`: `0.98x` or about `-1.5%`
- Trimmed-generation `Q` drift `L1 -> L3`: `0.98x` or about `-1.5%`

That decomposition matters. `W` drops materially while `Q` barely moves, which is the same interpretive pattern already established elsewhere in this repo for E114-style effects.

## Per-Condition Trimmed Results

Each condition still shows the same L1 to L3 drop:

| Condition | L1 W | L2 W | L3 W | L3/L1 W ratio | L1 Q | L3 Q | Q drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A this | 0.004178 | 0.004080 | 0.003476 | 0.83x | 0.114945 | 0.114520 | -0.4% |
| B a | 0.004260 | 0.003902 | 0.003452 | 0.81x | 0.117414 | 0.116506 | -0.8% |
| C your | 0.004173 | 0.003773 | 0.003363 | 0.81x | 0.116142 | 0.114465 | -1.4% |
| D the | 0.004099 | 0.003872 | 0.003411 | 0.83x | 0.117711 | 0.114389 | -2.8% |
| E their | 0.004114 | 0.003936 | 0.003453 | 0.84x | 0.116908 | 0.114543 | -2.0% |
| F our | 0.004223 | 0.004189 | 0.003384 | 0.80x | 0.118256 | 0.116173 | -1.8% |

There is no condition here that reverses the sign of the effect. That is the main robustness point.

## Best-Layer Summary

The pooled best layers also separate the categories cleanly:

| Category | Best layer | W_114 at best | S_114 at best | Q_114 at best | Mean rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 technical | 43 | 0.039815 | 0.3329 | 0.119902 | 3.45 |
| L2 recursive | 43 | 0.039015 | 0.3267 | 0.118993 | 3.95 |
| L3 experience | 30 | 0.021607 | 0.1580 | 0.136207 | 13.38 |

The `L1` and `L2` maxima both land at layer `43`, while `L3` peaks earlier at layer `30` and at a much lower mean rank quality. That is another way of seeing the suppression.

## Interpretation

This follow-up is the clean 122B topical-control confirmation for the E114 intervention family.

What it supports:

- the 122B model still shows a stable `L1 > L3` E114 pattern on this topical-control design
- the effect is selection-driven
- the effect survives trimming and does not depend on one deictic

What it does not by itself establish:

- a direct transfer of the full 35B mechanistic story into 122B
- that `E114` is the main 122B specialist outside this topical-control family

For the full per-layer tables, rank ranges, and capture details, use the detailed markdown artifact in `RESULTS/`.
