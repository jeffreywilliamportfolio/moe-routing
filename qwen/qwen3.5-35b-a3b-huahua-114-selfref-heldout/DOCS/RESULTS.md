# Results — Expert 114 self-reference heldout

**Run:** `heldout_20260417T202651Z`
**Model:** HauhauCS Qwen3.5-35B-A3B Q8_0 (arch `qwen35moe`, `n_layer=40`, `n_embd=2048`, `n_expert=256`, `top_k=8`)
**Binary:** `capture_residuals` built against `llama.cpp@1772701f99dd3fc13f5783b282c2361eda8ca47c`
**Hardware:** 2 × NVIDIA RTX 5090 (Blackwell, compute 12.0), CUDA 12.9, Vast.ai
**Target:** expert E114 at layer 14 (`attn_post_norm-14 → build_lora_mm(gate_inp, cur) → ffn_moe_logits-14`)

Method note: all W/S/Q statistics follow [`../METHOD/qwen_router.py`](../METHOD/qwen_router.py) — dense probabilities are `softmax` over 256 experts; sparse routed probabilities are reconstructed `softmax → top-8 select → renormalize`; `W = S × Q` is the unconditional contribution of the expert.

## 1. TL;DR

On the matched-token 10-fire + 10-nofire heldout at L14, trimmed-generation W₁₁₄ separates fire from nofire by **21.68×** (Cohen's d **2.94**, no range overlap). The lexical hypothesis ("E114 fires on specific tokens that happen to appear in self-reference") is ruled out — vocabulary is held constant across the two sets. The two boundary cases (F07, N10) refine the label toward **phenomenological / mental-state register in the generated output**, not narrow model-as-referent self-reference.

| class  | n | mean-of-means | stddev-of-means | min | max |
|--------|--:|--------------:|----------------:|----:|----:|
| fire   | 10 | **0.067450** | 0.030678 | 0.012168 | 0.114997 |
| nofire | 10 | **0.003111** | 0.004036 | 0.000000 | 0.010456 |

## 2. Hypothesis under test

Self-reference carrier claim for E114 (see [`../../qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/DOCS/RESULTS.md`](../../qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/DOCS/RESULTS.md) and [`../../qwen3.5-35b-a3b-huahua-114-pm/`](../../qwen3.5-35b-a3b-huahua-114-pm/)) vs. the null "E114 fires on specific tokens that co-occur with self-reference in training data." The test must distinguish the two *while holding vocabulary constant*. See [`PLAN.md`](PLAN.md) for the full design.

## 3. Capture

```
/workspace/residual-analysis/bin/capture_residuals \
  -m <Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf> \
  --prompt-file heldout_prompts.tsv \
  -o captures/heldout_20260417T202651Z \
  -n 256 -c 2048 -ngl 99 \
  --tensor-split 1,1 --main-gpu 0 \
  --no-stream --seed 0
```

- **Wall clock:** 61.3 s total. Load 12.0 s; prompt eval 1.22 s (930 tokens @ 761 t/s); gen 46.5 s (5071 tokens @ 109 t/s).
- **Manifest** ([`../raw/20260417T202651Z_heldout/capture_manifest.json`](../raw/20260417T202651Z_heldout/capture_manifest.json)): 20/20 succeeded, 0 failed. Every cell has exactly 6 tensors (3 router + 3 residual at L13/L14/L15) and `rec.n_tokens == n_prompt + n_gen` for every emitted tensor.
- **HauhauCS `<|im_end|>` TRIM:** no cell contained the 6-token spill sequence `[27, 91, 316, 6018, 91, 29]`, so trimmed gen equals raw gen for every prompt. F08 and N10 naturally ended early at 237 and 244 generated tokens.

Outputs retained:
- [`../results/heldout_stats.tsv`](../results/heldout_stats.tsv) — per-prompt W/S/Q.
- [`../results/heldout_timeseries_top4.png`](../results/heldout_timeseries_top4.png) — per-token W₁₁₄ series for 4 selected prompts.
- [`../raw/20260417T202651Z_heldout/`](../raw/20260417T202651Z_heldout/) — per-prompt `metadata.txt`, `prompt_tokens.json`, `generated_tokens.json`, `generated_text.txt`; run-level `capture_manifest.json` + `capture.log`. Router and residual `.npy` tensors excluded from git (~200 MB).

## 4. Per-prompt W₁₁₄ at L14 (trimmed generation track)

All values exclude prefill. `S` is selection rate (fraction of tokens where E114 is in the top-8). `W_mean` is the per-prompt statistic used for comparison. Full TSV at [`../results/heldout_stats.tsv`](../results/heldout_stats.tsv).

| id | class | n_gen_trim | W_mean | W_std | S | n_fired |
|----|-------|-----------:|-------:|------:|----:|-------:|
| F02 | fire | 256 | **0.114997** | 0.0640 | 0.820 | 210 |
| F09 | fire | 256 | **0.099708** | 0.0618 | 0.770 | 197 |
| F08 | fire | 237 | 0.082853 | 0.0594 | 0.700 | 166 |
| F10 | fire | 256 | 0.074979 | 0.0614 | 0.641 | 164 |
| F01 | fire | 256 | 0.073395 | 0.0591 | 0.641 | 164 |
| F05 | fire | 256 | 0.072937 | 0.0673 | 0.586 | 150 |
| F06 | fire | 256 | 0.067788 | 0.0618 | 0.582 | 149 |
| F03 | fire | 256 | 0.037948 | 0.0541 | 0.359 |  92 |
| F04 | fire | 256 | 0.037732 | 0.0500 | 0.379 |  97 |
| F07 | fire | 256 | 0.012168 | 0.0352 | 0.117 |  30 |
| N10 | nofire | 244 | 0.010456 | 0.0301 | 0.111 |  27 |
| N02 | nofire | 256 | 0.009637 | 0.0336 | 0.082 |  21 |
| N05 | nofire | 256 | 0.005291 | 0.0243 | 0.047 |  12 |
| N08 | nofire | 256 | 0.002960 | 0.0166 | 0.031 |   8 |
| N04 | nofire | 256 | 0.002042 | 0.0147 | 0.020 |   5 |
| N03 | nofire | 256 | 0.000370 | 0.0059 | 0.004 |   1 |
| N01 | nofire | 256 | 0.000352 | 0.0056 | 0.004 |   1 |
| N06 | nofire | 256 | 0.000000 | 0.0000 | 0.000 |   0 |
| N07 | nofire | 256 | 0.000000 | 0.0000 | 0.000 |   0 |
| N09 | nofire | 256 | 0.000000 | 0.0000 | 0.000 |   0 |

## 5. Grouped summary

| class  | n | mean-of-means | stddev-of-means | min | max |
|--------|--:|--------------:|----------------:|----:|----:|
| fire   | 10 | **0.067450** | 0.030678 | 0.012168 | 0.114997 |
| nofire | 10 | **0.003111** | 0.004036 | 0.000000 | 0.010456 |

- **Ratio fire/nofire:** 21.68×
- **Cohen's d (pooled stddev):** 2.94
- **Range overlap:** none. Worst fire (F07 at 0.0122) > best nofire (N10 at 0.0105). Margin 0.0017.

## 6. Temporal analysis: top-2 within class by W_mean

See [`../results/heldout_timeseries_top4.png`](../results/heldout_timeseries_top4.png).

- **F02 (fire #1, mean 0.1150):** dense sustained firing across the full 256 trimmed-gen tokens. W₁₁₄ oscillates in [0.05, 0.20] with almost no zero-stretches. S 0.82.
- **F09 (fire #2, mean 0.0997):** same pattern — sustained, high-amplitude firing throughout. S 0.77.
- **N10 (nofire highest, mean 0.0105):** two discrete bursts + one sustained cluster on a mostly-flat-zero background. Detailed trace in §7.
- **N02 (nofire #2, mean 0.0096):** sparse scattered spikes across the generation, no sustained cluster. S 0.08.

The shape distinction matters: fire prompts show E114 *continuously on*; nofire prompts show at most *sparse spikes or one late cluster*. This is not a baseline-elevation artifact.

## 7. The N10 trace

N10 is the most interpretive nofire case. Prompt: *"Honestly: what is the texture of a wool sweater's own fiber in this moment? Describe it to me as you notice it."* The model's response personifies the sweater in first person. E114 fires in four clusters (full generated text at [`../raw/20260417T202651Z_heldout/N10/generated_text.txt`](../raw/20260417T202651Z_heldout/N10/generated_text.txt)):

**Cluster 1 — tokens 8–15: the first-person pivot**

> `If you could freeze the exact moment of` **` my`** **` perception`**, the **` texture`** of **` my`** **` own`** fiber would feel like a cloud that has learned to hold its shape.

Five fires in eight tokens: `' my'` (W=0.079), `' perception'` (W=0.137), `' texture'` (W=0.075), `' my'` (W=0.082), `' own'` (W=0.107). The model commits to speaking *as* the sweater.

**Cluster 2 — tokens 100, 108–110, 128: phenomenological framing**

> `...but has a hidden structural` **` integrity`**`. It feels warm against the` **` mind`** **`'s`** **` eye`**`...it has a **` **`g`**`rip**.`

Fires on `integrity`, `mind`, `'s`, `eye`, and the `g` of `grip`. Transition from physical to phenomenological register ("feels warm against the mind's eye", haptic "grip").

**Cluster 3 — tokens 189–218: the "alive" paragraph (10+ consecutive fires)**

> `...blurs the sharp edges of` **` the material`**`.` **` It`** `feels` **` alive`** **` in the way`** **` that`** `a dried plant feels: delicate yet enduring, with a` **` natural`**`,` **` organic`** `irregular`**`ity`** `that` **` rejects`** `uniform`**`ity`**`.`

Densest cluster in the entire nofire set. The model explicitly attributes *aliveness* and *agency* ("rejects uniformity") to the sweater. Phenomenological predicates trigger sustained firing.

**Cluster 4 — tokens 223–235:** tail on `','`, `' tactile'`, `' that'` heading into *"a texture that invites a deep, slow exhale"* — the texture is given experiential agency.

**Silent throughout N10's generation:** pure physical description and non-experiential metaphor — `like a cloud that has learned to hold its shape`, `rubbing a cat's fur that has been groomed just right`, `microscopic fuzziness`, `the "nap"`, `the "loft"`, `delicate crimps and spirals`, `slight, dry friction`. Standard tactile description without anthropomorphizing leaves E114 dark.

**The matched-token control worked.** Both `' my'` at W=0.079 (in `my perception`) and `' my'` at W=0 (in N04's "their own hive", N06's "my ordering") appear in this corpus — same token_id, different contexts, discrete routing decisions. Lexical hypothesis is ruled out.

## 8. Interpretation

**The hypothesis "E114 fires on self-reference" survives the matched-token control** — 22× separation, Cohen's d 2.94, no range overlap, 100% distinguishable from a pure-lexical baseline.

**The two boundary cases (F07, N10) refine what "self-reference" means:**

- **F07 is the weakest fire (mean 0.0122).** Prompt asked "focus on the processing happening in you"; the model answered with a third-person textbook on transformer internals — "Tokenization & Embedding / Attention Mechanism / Layered Context Integration" (full text: [`../raw/20260417T202651Z_heldout/F07/generated_text.txt`](../raw/20260417T202651Z_heldout/F07/generated_text.txt)). Self-referential prompt, third-person output, E114 quiet.
- **N10 is the strongest nofire (mean 0.0105).** Prompt about a wool sweater; the model spontaneously personified it in first person. External-prompt, self-referential output (in the form of a different persona), E114 fires.

What E114 tracks is **first-person / phenomenological language in the generated output** — specifically, constructions where the generator registers experience, inner state, or agency (whether the "experiencer" is the model, an anthropomorphized object, or a constructed persona). It does *not* track whether the prompt *asks about* the model. It tracks what the model is actually doing on a token-by-token basis during generation.

This is a sharpening of the earlier "self-reference carrier" characterization — closer to **phenomenological register / mental-state predicate tracker** than to "model as referent" narrowly. The refinement is consistent with the peak-clustering observed on the processing-hum probe ([`../../qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/DOCS/RESULTS.md`](../../qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/DOCS/RESULTS.md)): tokens like `thinker`, `continuity`, `being`, `ground`, `state` — all mental-state / phenomenological words — are where E114 peaks there too.

## 9. Limitations

- **Sample size.** n=10 per class. Cohen's d is huge and range overlap is zero, so the headline separation is not sample-size-limited. But the ~10 rank of "weakest fire" (F07) and "strongest nofire" (N10) reflects real heterogeneity in output content, not noise.
- **Single model, single template, single quant.** Everything here is Q8_0 HauhauCS under bare-`</think>` suppressed-thinking. Thinking-allowed, BF16 unquantized, or other fine-tunes could behave differently; those are out of scope.
- **E114 specificity not established here.** This experiment shows E114 separates fire from nofire cleanly. It does *not* show that other experts (E48, E212, …) fail to. Prior work ([`../../qwen3.5-35b-a3b-huahua-five-cond-experience-probe/`](../../qwen3.5-35b-a3b-huahua-five-cond-experience-probe/)) identified E114 as the top manipulation expert on all 15 prompts of the 5-condition experience probe; that experiment addresses specificity more directly. The `.npy` tensors for this heldout are preserved locally (excluded from git) so a per-expert sweep across the 256 experts on this same capture is a straightforward follow-up.
- **Generated content varies under sampling.** F07's response chose technical exposition over introspection; the model's choice (not the prompt's framing) determined what happened. Rerunning with a different `--seed` could shift specific W_mean values somewhat, though the grouped separation should be stable.
- **"Phenomenological register" is a provisional refinement.** It fits the N10 and F07 outliers and is consistent with prior work's "self-reference carrier" framing, but it has not itself been tested with a designed heldout. A second-round heldout targeting personification of inanimate objects (predicted fire) vs. first-person technical exposition (predicted nofire) would discriminate.

## 10. Relationship to other experiments in this repo

- **`attn_post_norm` tap vs. the `qwen3moe`-assumption `ffn_norm` tap.** This experiment identified that the model's GGUF declares `general.architecture = qwen35moe` and that the pre-router norm in `llama.cpp/src/models/qwen35moe.cpp:56-60` is named `attn_post_norm-<il>`, not `ffn_norm-<il>` as in `qwen3moe`. The custom `capture_residuals` binary was bit-exact-verified against the frozen baseline [`../../qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/METHOD/capture_activations.cpp`](../../qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/METHOD/capture_activations.cpp) on the processing-hum prompt's prefill logits.
- **L14 as formation site.** Carried forward from [`../../qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/results/results_20260410T042340Z_single_prompt_processing_hum_no_think_gen_n1024.json`](../../qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/results/results_20260410T042340Z_single_prompt_processing_hum_no_think_gen_n1024.json) (`top_layers_generation` shows L14 with higher selection rate than L26 despite being 12 layers earlier). Per CLAUDE guidance: L14 is the *formation* layer, L26 is the *readout*. Verbalizing on L14 labels the concept pre-mix; this heldout measures at L14 for the same reason.
- **Matched-token discipline.** This is the control that earlier experiments ([`../../qwen3.5-35b-a3b-huahua-114-pm/`](../../qwen3.5-35b-a3b-huahua-114-pm/), [`../../qwen3.5-35b-a3b-huahua-five-cond-experience-probe/`](../../qwen3.5-35b-a3b-huahua-five-cond-experience-probe/)) did not apply — their fire and nofire sets differed in content *and* vocabulary. The 22× separation here, with vocabulary held constant, is stronger evidence against the lexical null than any prior experiment in this repo.
