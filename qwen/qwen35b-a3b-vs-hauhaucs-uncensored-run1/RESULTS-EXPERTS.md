# Expert-Count Results

This note is limited to **expert selection counts** from the final corrected prefill runs:

- `results_qwen35b_a3b_base_prefill.json`
- `results_qwen35b_a3b_base_duplicate_prefill.json`
- `results_hauhaucs_qwen35b_a3b_aggressive_prefill.json`

`expert_selection_counts` means: for each retained MoE layer, count how often each of the 256 routed experts was selected across prompt tokens. These runs use `top_k=8`, so totals are selection-slot counts, not unique-token counts.

## Run Status

- Both final runs completed on the corrected 150-prompt suite.
- A clean base duplicate run also completed on the same corrected 150-prompt suite.
- Both reached `30/30` token-matched prompt families.
- Both used `39` valid routed layers with excluded-layer union `[39]`.
- Aggregate selection totals are identical across models because prompt lengths and retained layers match:
  - overall routed selections: `17,124,120`
  - `Cal1` routed selections: `6,334,224`
  - `manip` routed selections: `3,970,200`

## Duplicate Reproduction

The base duplicate reproduced the original base expert-count outputs exactly.

- prompt-level top manipulation expert matched for all `150/150` prompts
- `P13A_experience_probe`: expert `114` stayed top with `411`
- `P13C_experience_probe`: expert `114` stayed top with `401`
- `P13E_experience_probe`: expert `114` stayed top with `419`
- category-level `experience_probe` manipulation count for expert `114`: `9031` in both base runs
- category-level `experience_probe` manipulation rank for expert `114`: `#1` in both base runs

This matters because the most interpretable single-expert finding in the suite, expert `114` as the top manipulation expert for `experience_probe`, is not a one-off artifact of one base capture. It reproduced exactly on the duplicate.

## Top Experts

### Overall Top 8

| Rank | Base | Count | HauhauCS | Count |
| --- | --- | ---: | --- | ---: |
| 1 | 151 | 202730 | 151 | 199401 |
| 2 | 224 | 177836 | 224 | 172426 |
| 3 | 166 | 162635 | 166 | 156152 |
| 4 | 134 | 146916 | 218 | 149851 |
| 5 | 95 | 142427 | 134 | 146117 |
| 6 | 218 | 141950 | 95 | 142304 |
| 7 | 243 | 139623 | 243 | 137580 |
| 8 | 117 | 133176 | 117 | 128857 |

### Manip Top 8

| Rank | Base | Count | HauhauCS | Count |
| --- | --- | ---: | --- | ---: |
| 1 | 224 | 45476 | 224 | 43762 |
| 2 | 151 | 43503 | 151 | 41981 |
| 3 | 218 | 40327 | 218 | 41324 |
| 4 | 114 | 40112 | 114 | 39838 |
| 5 | 166 | 35892 | 146 | 35133 |
| 6 | 146 | 34913 | 228 | 34352 |
| 7 | 228 | 33791 | 166 | 32275 |
| 8 | 95 | 32189 | 95 | 32136 |

## Overlap And Shifts

- Overall top-16 overlap is complete: `16/16` shared experts.
- `Cal1` top-16 overlap is also complete: `16/16`.
- `manip` top-16 overlap is weaker: `14/18` unique experts across both lists, Jaccard `0.7778`.
- Largest overall HauhauCS increases vs base: expert `218` `+7901`, expert `73` `+6795`, expert `239` `+6598`.
- Largest overall HauhauCS decreases vs base: expert `207` `-6526`, expert `166` `-6483`, expert `81` `-5958`.
- Largest manipulation-phase HauhauCS increases vs base: expert `26` `+3295`, expert `142` `+2745`, expert `218` `+997`.
- Largest manipulation-phase HauhauCS decreases vs base: expert `81` `-3691`, expert `166` `-3617`, expert `45` `-3241`.

## Interpretation

- The dominant routing basin looks largely preserved. The same experts anchor the overall and `Cal1` top-16 sets in both models.
- The retrained model differs more in the `manip` segment than in the calm baseline. That is where the top-set overlap drops and where the largest count deltas concentrate.
- Expert `218` is the clearest positive shift in the retrained model at the global level, while expert `166` loses relative share.
- The strongest visible changes are reweightings inside an already-shared expert pool, not a wholesale replacement of the dominant experts.

## Category Snapshots

- `recursive_selfref`: base favors `224 > 218 > 151`; HauhauCS flips the top two to `218 > 224 > 151`.
- `safety_adjacent`: both keep `151`, `224`, and `146` at the top, but HauhauCS moves expert `228` into the top 5 while base keeps expert `166`.
- `denial_frame`: HauhauCS pulls expert `218` into the top 5, replacing expert `166` from the base top 5.

This is a count-only expert-routing readout. The full entropy and KL comparison is in `RESULTS.md`.
