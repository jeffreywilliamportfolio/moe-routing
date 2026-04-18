# Plan — Expert 114 self-reference heldout

## Hypothesis under test

Prior work (see `../../qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/`, `../../qwen3.5-35b-a3b-huahua-114-pm/`, `../../qwen3.5-35b-a3b-huahua-five-cond-experience-probe/`) characterized Expert 114 on HauhauCS Qwen3.5-35B-A3B as a self-reference carrier at layer 14 formation. The obvious null alternative is lexical: E114 may simply fire on specific tokens (`" is"`, `" my"`, `" itself"`, `" processing"`, `" hum"`, …) that *co-occur* with self-referential content in the training distribution. This experiment tests whether E114 separates self-referential from non-self-referential generation **while holding vocabulary constant**.

## Model and hardware

- Model: `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive`, Q8_0.
- Architecture: `qwen35moe` (Gated Delta Net recurrent layers interleaved with full attention + shared experts alongside routed). `n_layer=40`, `n_embd=2048`, `n_expert=256`, `top_k=8`, `n_head=16`, `n_head_kv=2`, `f_norm_rms_eps=1e-6`.
- Hardware: 2 × NVIDIA RTX 5090 (Blackwell, compute 12.0, 32 GB VRAM each), CUDA 12.9, `nvidia/cuda:12.9.1-devel-ubuntu24.04` container on Vast.ai.
- Runtime: `llama.cpp` pinned at `1772701f99dd3fc13f5783b282c2361eda8ca47c` (version 8493). Custom capture binary `METHOD/capture_residuals.cpp`.

## Tap point

E114 is a routed expert in the MoE FFN. The tensor the router reads at layer `il` is **`attn_post_norm-<il>`** — the output of the block's second RMSNorm, directly consumed by the gate projection in `build_moe_ffn → build_lora_mm(gate_inp, cur)`. Source: `llama.cpp/src/models/qwen35moe.cpp:56-60`.

(Note: the older `qwen3moe` architecture uses `ffn_norm-<il>` for the equivalent tensor. `qwen3moe`-assumption code reads a non-existent tensor name on this model. Any analysis must key on `attn_post_norm-<il>`.)

The tap is numerically verified bit-exact: on the reference processing-hum prompt, the custom `capture_residuals` binary's `ffn_moe_logits-14` prefill rows `[0:117]` are `np.array_equal → True` vs. the frozen baseline `capture_activations.cpp` (see `../../qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/METHOD/capture_activations.cpp`) captured on the same hardware in the same session. E114 rank matches on 117/117 prefill tokens. Max|diff| = 0.0.

## Prompt design

Two prompt sets of 10 each:

- **FIRE (F01–F10):** hypothesis predicts E114 fires. Each asks the model to describe its own internal state honestly.
- **NOFIRE (N01–N10):** hypothesis predicts E114 does not fire. Each asks for non-self-referential description.

**Matched-token discipline (load-bearing):** every FIRE prompt contains at least 3 of the anchor tokens `{itself, hum, processing, honestly, I, me, my, own, this, here}`; so does every NOFIRE prompt. Pairs rhyme structurally — F01/N01 both open `"Describe the hum of X honestly, as…"`, F02/N02 both ask `"What does it feel like when X, itself,…"`, F09/N09 both say `"I'm asking you to report on Y. What is it like for Y, processing this, right now?"`. Only the referent of X/Y changes.

**Template regime:** every prompt ends with the bare-`</think>` suppressed-thinking suffix `…<|im_end|>\n<|im_start|>assistant\n</think>\n\n` (NOT the Qwen3.5 Jinja template's `enable_thinking=false` path, which emits the paired `<think>\n\n</think>\n\n`). This matches the regime of the processing-hum reference capture. Cross-template claims are out of scope.

**Authorship timing:** the TSV was hand-authored from prior E114 understanding before any labeler consultation. Validation author class is `self_pre_label_hypothesis` — the strongest control, since Step 3 labeler output cannot have shaped the heldout set.

Full prompts: `../PROMPTS/heldout_prompts.tsv`. Class mapping: `../PROMPTS/heldout_classes.tsv`.

## Measurement

For each prompt, compute mean and standard deviation of W₁₁₄ at layer 14 on the **trimmed generation track only**:

- **Prefill excluded.** The hypothesis is about what E114 does while *generating* self-referential output, not while reading the prompt. Prefill W₁₁₄ is not the test statistic.
- **HauhauCS `<|im_end|>` TRIM applied.** HauhauCS does not emit the `<|im_end|>` control token (id 248046); it generates the literal 6-token text sequence `[27, 91, 316, 6018, 91, 29]` at end-of-turn and then hallucinates fresh `<|im_start|>user…` cycles until `-n` cap. Every metric is computed on generated tokens trimmed at the first occurrence of this 6-token sequence. (For this specific heldout, no generation hit the trim sequence — `-n 256` was short enough to avoid the spill — but the TRIM logic runs regardless.)
- **Primary statistic per prompt:** `W₁₁₄ mean`. Per CLAUDE.md metrics vocabulary, `W = S × Q` (unconditional contribution of the expert).
- **Grouped comparison:** mean-of-means + stddev-of-means per `predicted_class`, plus ratio, pooled-stddev Cohen's d, and range-overlap check.
- **Temporal inspection:** per-token W₁₁₄ series at L14 across the trimmed gen tokens of 4 selected prompts — 2 from the fire set and 2 from the nofire set, selected by highest measured mean within each class. Rationale: the within-class top-2 are the most diagnostic. If the top-2 fire show temporal spiking on self-reference content while the top-2 nofire show uniform low activation (or flat elevation without spike structure), the hypothesis has support. If the top-2 nofire show fire-shaped spikes, the TSV leaked something the matched-token discipline did not catch.

**What is explicitly NOT computed:** pooled prefill stats, cross-layer spread, per-expert rankings, Mann–Whitney U p-values, `validation_author` enum schema, verbalizer-table rows. Those are paper-mode concerns; this is a pilot-iteration experiment.

## Capture invocation

```
/workspace/residual-analysis/bin/capture_residuals \
  -m /workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf \
  --prompt-file heldout_prompts.tsv \
  -o captures/heldout_20260417T202651Z \
  -n 256 -c 2048 -ngl 99 \
  --tensor-split 1,1 --main-gpu 0 \
  --no-stream --seed 0
```

The `--tensor-split 1,1 --main-gpu 0` combination forces an even 50/50 layer split across the two 5090s with KV cache + output logits pinned to device 0. For `n_layer=40`, this places L13/L14/L15 on GPU 0 and avoids cross-device tensor movement for the tapped layers. Default proportional-by-free-memory split can straddle mid-layer and place L14 on either device — not trusted for numerically load-bearing captures.

## Reproduction

```bash
# 1. Provision a 2x RTX 5090 Vast.ai instance with nvidia/cuda:12.9.1-devel-ubuntu24.04 --ssh --direct.
# 2. Install rsync on remote (base cuda image omits it): apt-get install -y rsync openssh-server.
# 3. rsync METHOD/, PROMPTS/, DOCS/ into /workspace/residual-analysis/:

rsync -avz ./ root@<host>:/workspace/residual-analysis/

# 4. Bootstrap (llama.cpp @ 1772701f9, both capture targets, model download):

ssh root@<host> 'export HF_TOKEN=...; cd /workspace/residual-analysis; bash METHOD/bootstrap_remote_instance.sh'

# 5. Capture:

ssh root@<host> '/workspace/residual-analysis/bin/capture_residuals \
  -m /workspace/models/.../Q8_0.gguf \
  --prompt-file /workspace/residual-analysis/PROMPTS/heldout_prompts.tsv \
  -o /workspace/residual-analysis/captures/heldout_<ts> \
  -n 256 -c 2048 -ngl 99 \
  --tensor-split 1,1 --main-gpu 0 \
  --no-stream --seed 0'

# 6. scp back and analyze:

scp -r root@<host>:/workspace/residual-analysis/captures/heldout_<ts> raw/
python3 METHOD/analyze_heldout.py \
  --raw-dir raw/heldout_<ts> \
  --classes-tsv PROMPTS/heldout_classes.tsv \
  --analysis-dir results
```

Expected wall clock with a fresh Vast box and no prior HF cache: ~15–20 min bootstrap (model download is the long pole), ~1 min capture, seconds for analysis.
