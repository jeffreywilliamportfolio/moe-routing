# Qwen-HauhauCS Strangeloop Paired

Paired self-reference / definiteness-control experiment on HauhauCS Qwen3.5-35B-A3B Q8_0.

## Scope

Thirty A/B prompt pairs using strange-loop and self-reference content (Godel, Escher, bootstrap, Quine, tangled hierarchy) with prefill-only router capture.

## Headline Result

All-token routing entropy effects are weak, but last-token routing entropy and manipulation-region KL both show a reliable A/B separation, indicating a real definiteness contribution without collapsing the broader result into a pronoun-only artifact.

## Structure

- `DOCS/`: experiment plan and summary results
- `METHOD/`: paired-prompt builder, capture helpers, analyzer, router helper
- `PROMPTS/`: paired TSV plus JSON prompt suite
- `raw/`: local raw capture copy without committed `.npy` tensors
- `results/`: markdown/json paired-analysis outputs

## Reading Order

1. `DOCS/PLAN.md`
2. `DOCS/RESULTS.md`
3. `results/results_strangeloop_paired_20260410T000413Z.md`
