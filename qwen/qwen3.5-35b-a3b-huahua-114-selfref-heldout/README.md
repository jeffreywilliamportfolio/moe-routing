# Qwen3.5-35B-A3B HauhauCS — Expert 114 self-reference heldout test

Matched-token controlled heldout test of the hypothesis that Expert 114 at layer 14 carries self-reference during generation on HauhauCS Qwen3.5-35B-A3B Q8_0.

## Scope

Twenty hand-authored prompts partitioned 10 fire / 10 nofire. Every nofire prompt uses the same anchor tokens (`itself`, `hum`, `processing`, `honestly`, `I`, `me`, `my`, `own`, `this`, `here`) as the fire prompts, placed in non-self-referential contexts (refrigerators, bees, CPUs, rivers, chefs, cats, wool sweaters). Structurally rhyming pairs (F01/N01, F02/N02, …). A pure-lexical hypothesis ("E114 fires on the word `is`") would predict similar W₁₁₄ on both sets.

Run: `heldout_20260417T202651Z` | 20 prompts | 256 tokens/prompt | bare-`</think>` no-think template, `--seed 0` default sampling | 2 × RTX 5090 (Vast.ai), CUDA 12.9, `llama.cpp@1772701f9`, custom `capture_residuals` binary.

**Sampling note:** this is a stochastic/default-sampling generation run. The grouped fire/nofire separation is the result to read; individual generated continuations can vary under a different seed.

**Headline result.** On the trimmed generation track at L14, fire mean-of-means W₁₁₄ = **0.0675 ± 0.0307**, nofire = **0.0031 ± 0.0040**. Ratio **21.7×**, Cohen's d **2.94**, **no range overlap** (worst fire 0.0122 > best nofire 0.0105). The lexical hypothesis is ruled out.

**Refinement.** The two boundary cases sharpen the label:
- **F07** (weakest fire) responded to a self-referential prompt with a third-person textbook on transformer internals → E114 stayed quiet.
- **N10** (strongest nofire) was asked about a wool sweater and spontaneously personified it in first person ("my perception … my own fiber … it feels alive") → E114 fired in four clusters on the phenomenological tokens.

Provisional refined characterization: E114 tracks **phenomenological / mental-state register in the generated output**, not prompt type. First-person experiential language fires it regardless of whether the experiencer is the model, an anthropomorphized object, or a constructed persona.

## Folder Contents

- [`METHOD/`](METHOD/): `capture_residuals.cpp` (residual + router tap at L13/L14/L15), `bootstrap_remote_instance.sh` (Vast.ai provisioning, CUDA 12.9), `qwen_router.py` (canonical router reconstruction), `analyze_heldout.py` (per-prompt W₁₁₄ stats + top-4 timeseries plot).
- [`PROMPTS/`](PROMPTS/): `heldout_prompts.tsv` (2-col prompt_id + prompt, `\n` as literal newline escapes), `heldout_classes.tsv` (sidecar `prompt_id`→`predicted_class`).
- [`DOCS/`](DOCS/): [`PLAN.md`](DOCS/PLAN.md), [`RESULTS.md`](DOCS/RESULTS.md).
- [`results/`](results/): `heldout_stats.tsv` (per-prompt W/S/Q), `heldout_timeseries_top4.png` (per-token W₁₁₄ at L14 for top-2 fire + top-2 nofire by within-class mean).
- [`raw/20260417T202651Z_heldout/`](raw/20260417T202651Z_heldout/): per-prompt `metadata.txt`, `prompt_tokens.json`, `generated_tokens.json`, `generated_text.txt`, plus the run-level `capture_manifest.json` and `capture.log`. Router + residual `.npy` tensors excluded from git per repo policy (~200 MB, regenerable).

## Relationship to other experiments

- Identifies the tap point (`attn_post_norm-<il>` for `qwen35moe`, not `ffn_norm-<il>` as in `qwen3moe`). The numerically verified bit-exact check against the frozen baseline capture binary used the [`qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/METHOD/capture_activations.cpp`](../qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/METHOD/capture_activations.cpp) as the baseline reference.
- Builds on the L14-vs-L26 formation-vs-readout rationale from [`qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/results/results_20260410T042340Z_single_prompt_processing_hum_no_think_gen_n1024.json`](../qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/results/results_20260410T042340Z_single_prompt_processing_hum_no_think_gen_n1024.json) (`top_layers_generation`).
- Complementary to [`qwen3.5-35b-a3b-huahua-114-pm/`](../qwen3.5-35b-a3b-huahua-114-pm/) (bias-sweep on E114) and [`qwen3.5-35b-a3b-huahua-five-cond-experience-probe/`](../qwen3.5-35b-a3b-huahua-five-cond-experience-probe/) (5-condition experience probe showing E114 as the top manipulation expert). This folder provides the matched-token control that prior experiments did not.

## Reproducibility

- Yes: re-analysis of included `results/heldout_stats.tsv` and per-prompt `raw/…/generated_text.txt` + `generated_tokens.json` + `metadata.txt`.
- Yes: re-run `METHOD/analyze_heldout.py` against a freshly captured `raw/<run_id>/`.
- No: end-to-end rerun without a Vast.ai instance and HuggingFace access to `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive`. See [`DOCS/PLAN.md`](DOCS/PLAN.md) §Reproduction for the remote capture workflow.

## Reading Order

1. [`DOCS/PLAN.md`](DOCS/PLAN.md) — hypothesis, matched-token discipline, measurement spec.
2. [`DOCS/RESULTS.md`](DOCS/RESULTS.md) — full tables, temporal analysis, N10 cluster-by-cluster trace, interpretation.
3. [`METHOD/analyze_heldout.py`](METHOD/analyze_heldout.py) — the computation.
4. [`results/heldout_timeseries_top4.png`](results/heldout_timeseries_top4.png) — the shape.
