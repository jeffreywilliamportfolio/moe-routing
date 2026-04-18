# Results

This brief summarizes only the locally verifiable HauhauCS aggressive artifacts bundled in this folder. It separates historical full-condition results from the local rerun summaries so that reproducibility claims do not exceed the data presently on disk.

## Method Note

All routing claims here follow the local source of truth in [qwen_router.py](../METHOD/qwen_router.py): dense probabilities are `softmax` over all `256` experts, sparse routed probabilities are reconstructed by `softmax -> top-8 select -> renormalize`, normalized routing entropy is computed on the sparse routed distribution, and `kl_manip_*` is computed on dense probabilities.

## Verified Findings

The historical full-scope `150`-prompt soft-bias artifact discussed in earlier notes is not bundled on this branch, so the claims here are restricted to the locally present rerun summaries and focus reports.

The resident rerun under [20260325-raw-npy-rerun](20260325-raw-npy-rerun) confirms the analysis pipeline on the locally bundled subset. The current coverage is `61` prompts for [analysis-baseline-root.json](20260325-raw-npy-rerun/5cond/analysis-baseline-root.json), `57` for [analysis-baseline-capture.json](20260325-raw-npy-rerun/5cond/analysis-baseline-capture.json), `52` for [analysis-expert_114_soft_bias_0.25.json](20260325-raw-npy-rerun/5cond/analysis-expert_114_soft_bias_0.25.json), `49` for [analysis-expert_114_soft_bias_0.5.json](20260325-raw-npy-rerun/5cond/analysis-expert_114_soft_bias_0.5.json), and `58` for [analysis-expert_114_soft_bias_1.0.json](20260325-raw-npy-rerun/5cond/analysis-expert_114_soft_bias_1.0.json). Across these subset reruns, `prefill_re_mean` remains tightly stable at approximately `0.956` and `last_token_re_mean` at approximately `0.961`, which supports methodological consistency. These subset files should not be treated as replacements for the historical full-matrix analyses referenced in earlier notes.

The smoke intervention run provides a clean routing-specific sanity check on a matched `3`-prompt subset. The regenerated output [smoke/analysis.json](20260325-raw-npy-rerun/smoke/analysis.json) shows that `expert_114_soft_bias_1.0` raises Expert `114` selection rate across all three bands by about `0.039` to `0.047`, with comparable weight-rate increases. `expert_114_forced_inclusion` is markedly stronger, raising Expert `114` selection rate by about `0.121` in `process`, `regulation`, and `static_fact`, with band-level JSD values of approximately `0.056` to `0.067`. By contrast, sham interventions on experts `134` and `243` leave Expert `114` selection-rate deltas near zero under both soft-bias and forced-inclusion conditions. Within the local evidence, this is the cleanest sign that the intervention manipulates participation of Expert `114` specifically rather than merely increasing routing instability globally. See also [RESULTS-NOTHINK-COMPARISON.md](RESULTS-NOTHINK-COMPARISON.md) and [RESULTS-SHAM-CONTROLS.md](RESULTS-SHAM-CONTROLS.md).

## Interpretation

The current evidence supports a narrow conclusion. Expert `114` can be selectively recruited in HauhauCS aggressive routing, and this effect is strongest under forced inclusion and detectable but smaller under soft bias. The bundled rerun validates the analysis path on the subset presently available, but because the branch does not include the full historical run matrix, it supports subset-level verification rather than a full rerun claim. The smoke run should be read as a routing sanity check, not as sufficient evidence for behavioral or rubric-level claims.
