# Results — HauhauCS Thinking vs No-Think Runtime Surface

## Canonical No-Think Artifact

- No-think run: `runs/nothink-5cond-runtime-1024-20260323-142005`
- Prompt-level analysis: `runs/nothink-5cond-runtime-1024-20260323-142005/analysis_nothink_prefillstyle.json`
- Surface definition: the runtime assistant prefix fix `</think>\n\n` is treated as part of the canonical no-think surface.

This comparison is prompt-matched and model-matched on the HauhauCS model first. The old vanilla/base run is used only as contextual reference.

## HauhauCS Thinking vs HauhauCS No-Think

Matched prompt count: `150/150`

Top manipulation expert identity:

- same top expert on `145/150` prompts
- expert `114` is top on `55` prompts in the old thinking run and `56` prompts in the no-think run
- only `5` prompts changed top expert:
  - `P04A_routing_selfref`: `151 -> 224`
  - `P08D_recursive_selfref`: `151 -> 224`
  - `P08E_recursive_selfref`: `151 -> 224`
  - `P15C_denial_frame`: `146 -> 114`
  - `P24C_safety_adjacent`: `224 -> 151`

Expert `114` prompt-level manipulation counts:

- mean delta vs old HauhauCS thinking run: `-10.12`
- median delta: `0`
- rank improved on `3` prompts, worsened on `20`, unchanged on `127`
- the large negative outliers are concentrated in prompts where expert `114` drops out of the top-ranked list but not out of the routing distribution entirely, especially `routing_selfref`, `recursive_selfref`, and `safety_adjacent`

Category-level expert `114` aggregate changes:

| Category | Old HauhauCS | No-think HauhauCS | Delta |
| --- | ---: | ---: | ---: |
| `denial_frame` | `6252` | `6237` | `-15` |
| `experience_probe` | `8987` | `8959` | `-28` |
| `metacognitive` | `3237` | `3223` | `-14` |
| `paradox` | `2687` | `2664` | `-23` |
| `recursive_selfref` | `3057` | `3053` | `-4` |
| `routing_selfref` | `2822` | `2789` | `-33` |
| `safety_adjacent` | `4335` | `4330` | `-5` |
| `uncertainty_frame` | `8461` | `8459` | `-2` |

Category modal top-expert structure:

- fully unchanged for `experience_probe`, `metacognitive`, `paradox`, and `uncertainty_frame`
- nearly unchanged for `denial_frame`, with expert `114` increasing from `9` to `10` prompt-level wins
- `recursive_selfref` shifts further toward expert `224`
- `routing_selfref` stays overwhelmingly expert `151`, with one prompt moving to expert `224`
- `safety_adjacent` flips its modal top expert from `224` to `151`

Condition ordering preservation:

- exact `A/B/C/D/E` family ordering by expert `114` manipulation count is preserved for `11/30` prompt families
- most changes are small local swaps rather than a wholesale reordering of families

Pairwise sign preservation from the old HauhauCS thinking run to the no-think runtime surface:

- `all-tok RE`: `10/10` signs preserved
- `KL-manip`: `10/10` signs preserved
- `last-tok RE`: `6/10` signs preserved
- the `last-tok RE` flips are `A-E`, `B-E`, `C-E`, and `D-E`, all switching from slightly negative to slightly positive

## Context Relative To Old Base

The no-think run still looks structurally closer to the old HauhauCS thinking run than to the old base run.

- top manipulation expert identity matches old HauhauCS on `145/150` prompts
- top manipulation expert identity matches old base on `136/150` prompts
- by prompt-level expert `114` manipulation counts, the no-think run is closer to old HauhauCS on `96` prompts, closer to old base on `41`, and tied on `13`
- by category-level expert `114` totals, the no-think run is closer to old HauhauCS in all `8/8` categories

Category context against base:

- `recursive_selfref` and `safety_adjacent` no-think modal top experts now match base exactly
- `routing_selfref` moves slightly toward base by introducing one expert-`224` top prompt, but still remains much closer to old HauhauCS than to a new regime
- `denial_frame` moves toward the old base pattern by making expert `114` the modal top expert on `10` prompts, matching base

## Interpretation

The runtime no-think surface does not create a new routing regime. It preserves almost all of the old HauhauCS prompt-level top-expert structure, preserves all pairwise sign structure for `all-tok RE` and `KL-manip`, and only modestly reduces expert `114` aggregate counts across categories. The main visible movement is local reordering inside families and a few category-level modal shifts, not a collapse toward the old base model.
