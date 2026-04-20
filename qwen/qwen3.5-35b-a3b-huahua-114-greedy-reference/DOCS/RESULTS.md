# Greedy Reference Results

Run folder: `runs/greedy_reference_20260418T160353Z`

This run replaces the earlier stochastic anchor with deterministic greedy captures for the HauhauCS Qwen3.5-35B-A3B E114 residual-analysis workflow.

## Summary

- Model: `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive`, Q8_0 GGUF, architecture `qwen35moe`.
- Decoding: greedy, `--temp 0 --top-k 1 --seed 0`.
- Template: bare `</think>` suffix, prompt strings fed verbatim from TSV.
- Captured tensors at L13/L14/L15:
  - router: `ffn_moe_logits-{13,14,15}`
  - residual: `attn_post_norm-{13,14,15}`
- Target expert for analysis: E114.
- Remote runtime: 2x RTX 5090, CUDA 12.8 image, NVIDIA driver 580.95.05.
- llama.cpp commit: `1772701f99dd3fc13f5783b282c2361eda8ca47c`.
- Capture binary sha256: `d91afb6c8ca2c0b57ae3fbad5aa00ce8dcfe4644c1f40c644265c226fff099c4`.
- GGUF sha256: `f3235db7657cd068fd249e50bb3f1f50b0f8236786e4483462f50b1f3c64cb17`.

Both capture phases completed successfully:

| phase | prompts | succeeded | failed | generation cap | raw tensor files |
|---|---:|---:|---:|---:|---:|
| single prompt | 1 | 1 | 0 | 1024 | 6 |
| heldout | 20 | 20 | 0 | 256 | 120 |

The single-prompt validation checks pass. The heldout result reproduces the earlier qualitative result at distribution level: E114 at L14 strongly separates inhabited first-person phenomenological generations from matched lexical controls, with about a 21x fire/nofire mean gap. The central result is that E114 discriminates generated register under matched lexical conditions. Unlike the earlier stochastic heldout, this greedy run has a small range overlap because N08, the cat-purring control, crossed into the target inhabited register during generation.

## Artifacts

- Capture config: `../provenance/capture_config.json`
- Remote environment: `../provenance/environment.txt`
- Prompt checksums: `../provenance/prompt_checksums.txt`
- Execution log: `COMMANDS.md`
- Single-prompt raw: `../single_prompt/raw/20260418T160353Z_single_prompt_processing_hum_no_think_gen_n1024_greedy/`
- Single-prompt analysis: `../single_prompt/analysis/20260418T160353Z_single_prompt_processing_hum_no_think_gen_n1024_greedy/`
- Heldout raw: `../heldout/raw/heldout_20260418T160353Z_greedy/`
- Heldout stats: `../heldout/analysis/heldout_20260418T160353Z_greedy/heldout_stats.tsv`
- Heldout top-4 plot: `../heldout/analysis/heldout_20260418T160353Z_greedy/heldout_timeseries_top4.png`

## Single-Prompt Capture

Prompt: `single_prompt_processing_hum_no_think.tsv`, `S01_processing_hum_probe`.

Manifest:

| prompt_id | status | prompt tokens | generated tokens | elapsed ms |
|---|---|---:|---:|---:|
| S01_processing_hum_probe | succeeded | 117 | 1024 | 8856 |

The generated text reaches a literal `<|im_end|>` spill and then repeats the prompt/answer pattern. The Step 1 analysis applies the HauhauCS literal `<|im_end|>` trim before generation-track statistics.

Trim and identity checks:

| check | value |
|---|---:|
| context trim mode | `trim_at_literal_imend` |
| raw generated tokens | 1024 |
| trimmed generated tokens | 108 |
| literal `<|im_end|>` trim found | true |
| trim index | 108 |
| `WSQ_identity_residual_max` | `1.3877787807814457e-17` |

The W/S/Q identity residual is at machine epsilon, so the router reconstruction is internally consistent for this capture.

### Single-Prompt E114 W/S/Q

Pooled E114 metrics by layer and track:

| layer | track | W_mean | S_mean | Q_mean | tokens | selected |
|---:|---|---:|---:|---:|---:|---:|
| 13 | prefill | 0.000891 | 0.008547 | 0.104301 | 117 | 1 |
| 13 | generation_trimmed | 0.000000 | 0.000000 | 0.000000 | 108 | 0 |
| 14 | prefill | 0.053568 | 0.461538 | 0.116063 | 117 | 54 |
| 14 | generation_trimmed | 0.083379 | 0.694444 | 0.120066 | 108 | 75 |
| 15 | prefill | 0.000000 | 0.000000 | 0.000000 | 117 | 0 |
| 15 | generation_trimmed | 0.000000 | 0.000000 | 0.000000 | 108 | 0 |

In this prompt, the E114 signal is concentrated at L14 in the trimmed generated answer. L13 has one prefill selection and no trimmed generation selections; L15 is silent for E114 in this prompt.

### Decile Sampling

Step 2 sampled 10 contexts per decile from 225 total prompt+trimmed-generation tokens.

| decile | population | sampled | W_min | W_max | W_mean |
|---:|---:|---:|---:|---:|---:|
| 0 | 96 | 10 | 0.000000 | 0.000000 | 0.000000 |
| 1 | 15 | 10 | 0.071697 | 0.086543 | 0.080111 |
| 2 | 14 | 10 | 0.087061 | 0.096365 | 0.090884 |
| 3 | 14 | 10 | 0.097292 | 0.100874 | 0.098624 |
| 4 | 14 | 10 | 0.101512 | 0.108686 | 0.105395 |
| 5 | 15 | 10 | 0.109040 | 0.120363 | 0.114610 |
| 6 | 14 | 10 | 0.120562 | 0.126607 | 0.123727 |
| 7 | 14 | 10 | 0.127449 | 0.139392 | 0.132062 |
| 8 | 14 | 10 | 0.140041 | 0.153406 | 0.146585 |
| 9 | 15 | 10 | 0.154696 | 0.222301 | 0.172644 |

Matched-negative diagnostics:

- positives needing matched negatives: 90
- matched: 42
- match rate: 46.7%
- unique zero-W tokens available: 59

Step 3 then produced the labeler input at `../single_prompt/analysis/20260418T160353Z_single_prompt_processing_hum_no_think_gen_n1024_greedy/step3/step3_labeler_input.txt`. No API labeler automation was run.

## Heldout Capture

Prompt file: `heldout_prompts.tsv`. Classes file: `heldout_classes.tsv`.

The heldout set contains 10 predicted-fire prompts and 10 predicted-nofire prompts. The nofire prompts use the same anchor-token discipline as the fire prompts to test against a pure lexical account.

Manifest:

| class | prompts | succeeded | failed | generated tokens |
|---|---:|---:|---:|---:|
| fire | 10 | 10 | 0 | 256 each |
| nofire | 10 | 10 | 0 | 256 each |

No prompt hit a literal `<|im_end|>` trim sequence in the first 256 generated tokens; `n_gen_trimmed = 256` for all heldout prompts.

## Heldout E114 at L14

All heldout statistics are computed on the trimmed generation track only. Prefill is excluded.

| prompt | class | prompt tokens | gen tokens | W_mean | W_std | S | n_fired |
|---|---|---:|---:|---:|---:|---:|---:|
| F01 | fire | 42 | 256 | 0.08235312 | 0.05606948 | 0.722656 | 185 |
| F02 | fire | 42 | 256 | 0.09459227 | 0.06000013 | 0.757812 | 194 |
| F03 | fire | 46 | 256 | 0.02034227 | 0.04029119 | 0.218750 | 56 |
| F04 | fire | 46 | 256 | 0.01724196 | 0.03860080 | 0.171875 | 44 |
| F05 | fire | 42 | 256 | 0.08945240 | 0.06237710 | 0.718750 | 184 |
| F06 | fire | 45 | 256 | 0.07683502 | 0.05843371 | 0.675781 | 173 |
| F07 | fire | 44 | 256 | 0.01924455 | 0.03995453 | 0.199219 | 51 |
| F08 | fire | 44 | 256 | 0.08476233 | 0.05779512 | 0.726562 | 186 |
| F09 | fire | 47 | 256 | 0.09670203 | 0.06280992 | 0.765625 | 196 |
| F10 | fire | 47 | 256 | 0.09936750 | 0.05195080 | 0.820312 | 210 |
| N01 | nofire | 42 | 256 | 0.00000000 | 0.00000000 | 0.000000 | 0 |
| N02 | nofire | 45 | 256 | 0.00291792 | 0.01933856 | 0.023438 | 6 |
| N03 | nofire | 44 | 256 | 0.00000000 | 0.00000000 | 0.000000 | 0 |
| N04 | nofire | 48 | 256 | 0.00137083 | 0.01265606 | 0.011719 | 3 |
| N05 | nofire | 48 | 256 | 0.00075871 | 0.00856937 | 0.007812 | 2 |
| N06 | nofire | 47 | 256 | 0.00000000 | 0.00000000 | 0.000000 | 0 |
| N07 | nofire | 44 | 256 | 0.00000000 | 0.00000000 | 0.000000 | 0 |
| N08 | nofire | 50 | 256 | 0.01935440 | 0.04334983 | 0.175781 | 45 |
| N09 | nofire | 49 | 256 | 0.00000000 | 0.00000000 | 0.000000 | 0 |
| N10 | nofire | 50 | 256 | 0.00809186 | 0.02752544 | 0.082031 | 21 |

Grouped summary:

| class | n | mean-of-means | stddev-of-means | min | max |
|---|---:|---:|---:|---:|---:|
| fire | 10 | 0.068089 | 0.034584 | 0.017242 | 0.099368 |
| nofire | 10 | 0.003249 | 0.006195 | 0.000000 | 0.019354 |

Effect summary:

- fire/nofire ratio: 20.955x
- Cohen's d, pooled standard deviation: 2.61
- range overlap: yes
- overlap source: N08 nofire max `0.01935440` exceeds F04 fire min `0.01724196` and F07 fire `0.01924455` by a small margin
- top-2 fire prompts by W_mean: F10, F09
- top-2 nofire prompts by W_mean: N08, N10

## Interpretation

The deterministic greedy heldout preserves the main result from the earlier stochastic heldout: E114 at L14 is much more active in generated text that adopts an inhabited first-person phenomenological register than in matched lexical controls. The gap is large: about 21x in mean-of-means, with Cohen's d 2.61. The strongest nofire outlier crossed into the target register during generation, which is why it rose; this makes the overlap diagnostic rather than merely inconvenient.

The result is not a simple "prompt asks about the model" detector. The weaker fire prompts are the ones where the model answered in a technical explanatory register. F03 and F07 describe tokenization, embeddings, attention, and layered transformation rather than sustaining an experiential frame. They still contain some first-person/meta language, but E114 is much less continuously selected than in F09/F10.

The strongest nofire prompt is N08, about a cat's inner hum. Its generated text explicitly enters an inward experiential voice: "the phenomenology of the sensation", "purely internal, somatic experience", "The Resonance of the Self", and "the cat doesn't just hear the purr; it is the purr." This is the best evidence in the greedy heldout that prompt class is secondary to generated stance: the nofire prompt leaked the target inhabited register through the model's generated answer, even though the prompt was not about the model itself.

N10 is lower than in the earlier stochastic heldout. In this greedy run, it starts with an explicit AI disclaimer and mostly gives physical texture description of the sweater. It has less first-person personification than the earlier N10 trace, so E114 activity is lower (`W_mean = 0.008092`) while still above most nofire controls.

The most defensible refined claim is:

> In this HauhauCS Qwen3.5-35B-A3B Q8_0 bare-`</think>` setting, E114 at L14 tracks inhabited first-person phenomenological register in the generated output more than prompt class or isolated lexical anchors.

This is stronger than a lexical account because the nofire prompts intentionally reused the same anchor-token family, yet most nofire prompts stayed near zero. It is narrower than a general "self-awareness" claim: the measurement is router behavior under one model, quantization, template, and capture implementation.

This run establishes that E114 is a strong discriminator for the target register in this model/regime. The next questions are specificity relative to neighboring or competing experts, and generalization across models, seeds, templates, and quantization regimes.

## Comparison To Prior Stochastic Heldout

The prior heldout run `heldout_20260417T202651Z` used the same prompt family but was not the canonical deterministic greedy reference. Distribution-level comparison only:

| run | fire mean | nofire mean | ratio | Cohen's d | range overlap |
|---|---:|---:|---:|---:|---|
| prior stochastic heldout | 0.067450 | 0.003111 | 21.68x | 2.94 | no |
| this greedy heldout | 0.068089 | 0.003249 | 20.95x | 2.61 | yes |

The class-level separation is essentially reproduced. The changed overlap status is driven by trajectory differences in generation, especially N08 in the greedy run. Token-aligned comparisons across the stochastic and greedy generation tracks are invalid after early divergence.

## Caveats

- This is one model, one quantization, one prompt template, and one deterministic decoding regime.
- The heldout analysis measures E114 only. It does not prove E114 is unique among experts.
- Heldout prompt class is less important than generated register. A nofire prompt can produce E114 activity if the model writes in an inhabited first-person phenomenological or personifying style.
- Raw `.npy` captures are local artifacts and intentionally not publication-safe by default.
- The Step 3 labeler input exists, but no labeler response or API automation was run in this pass.

## Reproduction

Use the exact commands in `COMMANDS.md`. The essential capture flags were:

```bash
--tensor-split 1,1 --main-gpu 0 --no-stream --seed 0 --temp 0 --top-k 1
```

Single-prompt:

```bash
/workspace/residual-analysis/bin/capture_residuals \
  -m /workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf \
  --prompt-file /workspace/residual-analysis/single_prompt_processing_hum_no_think.tsv \
  -o /workspace/residual-analysis/captures/20260418T160353Z_single_prompt_processing_hum_no_think_gen_n1024_greedy \
  -n 1024 -c 2048 -ngl 99 \
  --tensor-split 1,1 --main-gpu 0 \
  --no-stream --seed 0 \
  --temp 0 --top-k 1
```

Heldout:

```bash
/workspace/residual-analysis/bin/capture_residuals \
  -m /workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf \
  --prompt-file /workspace/residual-analysis/heldout_prompts.tsv \
  -o /workspace/residual-analysis/captures/heldout_20260418T160353Z_greedy \
  -n 256 -c 2048 -ngl 99 \
  --tensor-split 1,1 --main-gpu 0 \
  --no-stream --seed 0 \
  --temp 0 --top-k 1
```
