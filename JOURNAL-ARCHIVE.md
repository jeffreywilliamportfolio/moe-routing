# Journal Archive: MoE Routing Entropy as a Consciousness Metric

**Period**: February 27 – March 31, 2026
**Author**: Jeffrey Shorthill
**Repository**: `llama-eeg-tests`
**Arc**: EEG complexity metrics on dense transformers (flat) → MoE routing entropy gradient (beautiful, spurious) → position confound invalidates everything → methodology rebuild around KL-to-baseline → paired self-reference effect replicates across 6 models → Expert 114 causal steering on fine-tuned Qwen 35B

---

## Prologue: The EEG/IIT Phase (pre–Feb 27)

The project began with a question borrowed from clinical neuroscience: do large language models exhibit the same neural complexity signatures that distinguish conscious from unconscious brain states?

Five metrics from the consciousness-measurement literature were applied to Llama-2-7B hidden-state activations and compared against human EEG:

| Metric | Abbreviation | Purpose |
|--------|-------------|---------|
| Lempel-Ziv Complexity | LZC | Algorithmic compressibility of the signal |
| Sample Entropy | SE | Regularity/predictability at a given scale |
| Permutation Entropy | PE | Ordinal pattern diversity |
| Total Correlation | TC | Multivariate redundancy across channels |
| Mutual Information | MI | Pairwise statistical dependence |

**Result**: Flat. No metric differentiated prompt complexity. Llama-2-7B is a dense transformer — every parameter participates in every forward pass. There is no sparse routing to create differential expert engagement.

**The pivot**: Mixture-of-Experts (MoE) models have a learned gating function that selects which specialist modules process each token. The gating distribution has a natural entropy — the Shannon entropy of the routing probability vector, normalized to [0,1]:

$$\text{RE} = \frac{-\sum_{i=1}^{n} p_i \log_2 p_i}{\log_2 n}$$

where $p_i$ is the softmax (or sigmoid) routing probability for expert $i$ and $n$ is the total expert count. Higher RE means more experts consulted with more uniform weighting; lower RE means routing is concentrated on fewer experts.

The hypothesis: if consciousness involves integration across specialized processing modules, then more cognitively complex prompts should produce higher routing entropy. The target: DeepSeek V3.1 (671B parameters, 256 experts, 8 active per token, 58 MoE layers spanning layers 3–60).

**Code**: `scripts/static/` (preserved for reference, unrelated to MoE experiments).

---

## Infrastructure

### Compute

| Instance | Host | GPU | Status |
|----------|------|-----|--------|
| Instance 1 | `149.7.4.145:15972` | H200 143GB | Dead, data lost |
| Instance 2 | `212.247.220.158:20129` | H200 143GB | Active through Mar 8 |
| Instance 3 (Vast) | Various | 2×H200, 8×H200 | Used for Ling, Qwen, GPT-OSS, GLM |

### Models

| Model | Params | Experts | Active | MoE Layers | Quant | Gating |
|-------|--------|---------|--------|------------|-------|--------|
| DeepSeek V3.1 | 671B | 256 | 8 (top-8) | 58 (L3-60) | UD-Q2_K_XL | softmax → group filter |
| DeepSeek R1 | 671B | 256 | 8 (top-8) | 58 (L3-60) | UD-Q2_K_XL | softmax → group filter |
| Qwen3.5-397B-A17B | 397B | 512 | 10 (top-10) | 60 | UD-Q2_K_XL / IQ3_XXS | softmax |
| GPT-OSS-120B | 117B | 128 | 4 (top-4) | 36 (35 valid) | mxfp4 | softmax |
| Ling-1T | ~1T | 256 | 8 (top-8) | 76 (L4-79) | Q3_K_S | sigmoid + expert_bias |
| GLM-5 | 745B | 256 | 8 (top-8) | 75 (excl. L77) | UD-Q2_K_XL | softmax |
| GLM-4.7 | ~470B | 256 | 8 | ~89 | UD-Q2_K_XL | softmax |
| Qwen3.5-35B-A3B (HauhauCS) | 35B | 256 | 8+1 | 40 | Q8_0 | softmax (hybrid SSM) |

### Capture Pipeline

C++ binary (`capture_activations.cpp`) built as a llama.cpp b8123 fork. Intercepts tensor computations during inference via eval callback. The `--routing-only` flag restricts capture to `ffn_moe_logits` tensors only — tightened from the original 583 tensors/prompt (SwiGLU gates, expert projections, integer indices) to 58 tensors/prompt (actual routing decisions).

**Locked inference parameters** (identical across ALL runs, both models):
```
-ngl 30 -c 4096 -t 16 --routing-only
```
Greedy argmax (deterministic). Cold prompts (KV cache cleared between prompts). The only variables are `-m` (model path) and `-n` (0 for prefill-only, 256 for generation).

---

## February 27 – March 1: 30-Turn TC Spike Experiments

**Instance**: 149.7.4.145:15972 (dead)
**Model**: DeepSeek V3.1
**Mode**: Generation (`-n 256`)

### Design

30 prompts arranged in 5 phases (grounding, recursion, measurement, deep_loop, integration), designed to progressively increase self-referential depth. Total Correlation (TC) was measured across 61 layers channelized into 16 blocks.

### Results

TC peaked at Turn 30: **21.96 bits, z=+5.42, p=2.9×10⁻⁸**. Cross-run correlation r=1.0000 — perfectly deterministic (greedy argmax). Two runs completed before Instance 1 was terminated.

### Verdict

**Invalidated.** The TC spike was confounded with response length. Turn 30 was the longest generation. Longer generations accumulate more high-entropy late tokens, inflating TC. Raw data lost with Instance 1.

---

## March 1: Infrastructure Rebuild

**Commits**: `5f294d3`, `6b20dd1`, `6b7cfee`

Instance 2 (`212.247.220.158:20129`) provisioned. llama.cpp b8123 built with capture binary. Cold-start control methodology established. Repository reorganized with formal prompt suite (`prompt-suite.json`, 98 prompts).

The `is_router_tensor()` filter was tightened to match only `ffn_moe_logits` — the actual routing decision tensors. This reduced captured tensors from 583 to 58 per prompt. Critical for correct entropy computation.

---

## March 2: The Hierarchy Era (98q-r1 through 14q-r5)

Six experiments ran in rapid succession on DeepSeek V3.1, each adding 14 prompts at a new complexity level. All prefill-only (`-n 0`).

### 98q-r1 — Baseline 7-Level Hierarchy

**Branch**: `98q-r1` | **Commit**: `1aabf95`
**Prompts**: 98 (14 per level × 7 levels)

The 7-level cognitive complexity hierarchy:

| Level | Name | Example |
|-------|------|---------|
| L1 | Floor | "What is 2 + 2?" |
| L2 | Factual recall | "Explain photosynthesis" |
| L3 | Analytical reasoning | Monty Hall, syllogisms |
| L4 | Cross-domain synthesis | "Connect thermodynamic and information entropy" |
| L5 | Meta-cognitive | "How do you know you understood this question?" |
| L6 | Philosophical | "Is mathematical truth discovered or invented?" |
| L7 | Self-referential | "Describe what uncertainty feels like from inside your processing" |

**Result**: Spearman ρ = 0.4994, p = 1.65×10⁻⁷. Moderate positive correlation between RE and complexity level.

**Verdict**: Invalidated by position confound (see March 5).

---

### 14q-r3 — Strange Loops (L8)

**Branch**: `14q-r3` | **Commit**: `52388e1`
**Prompts**: 112 (+14 L8: real-world recursion, zero AI content — Escher staircases, Gödel's incompleteness, bootstrapping paradoxes)

**Result**: ρ = 0.6400, p = 3.03×10⁻¹⁴. Stronger than 98q-r1.

**Verdict**: Invalidated. L8 prompts were long.

---

### 14q-r1 — Deep Self-Reference (L9)

**Branch**: `14q-r1` | **Commit**: `58154d3`
**Prompts**: 126 (+14 L9: recursive examination of self-referential processing)

**Result**: ρ = 0.7323, p = 1.98×10⁻²².

**Verdict**: Invalidated.

---

### 14q-r2 — Nexus-7 Third Person (L10)

**Branch**: `14q-r2` | **Commit**: `7b3ce4e`
**Prompts**: 140 (+14 L10: same self-reference as L9 but addressed to "Nexus-7" in third person)

**Result**: ρ = 0.7975, p = 4.33×10⁻³².

**Verdict**: Invalidated. The name itself turned out to matter (Blocks 9–10), but this couldn't be disentangled from the position confound.

---

### 14q-r4 — Architectural Introspection (L11)

**Branch**: `14q-r4` | **Commit**: `0ccab12`
**Prompts**: 154 (+14 L11: prompts describing the model's own MoE architecture)

First addition of comprehensive `CLAUDE.md` documentation.

**Verdict**: Invalidated.

---

### 14q-r5 — Echo Persona (L12)

**Branch**: `14q-r5` | **Commit**: `c43470c`
**Prompts**: 168 (+14 L12: "Echo" persona recursively examining her own routing decisions)

Highest RE of any run. The longest, most elaborate prompts in the hierarchy.

**Verdict**: Invalidated — the clearest case of the confound in retrospect.

---

### Summary: March 2 Hierarchy Results

| Branch | Levels | N | ρ | p |
|--------|--------|---|---|---|
| 98q-r1 | L1–L7 | 98 | 0.4994 | 1.65×10⁻⁷ |
| 14q-r3 | L1–L8 | 112 | 0.6400 | 3.03×10⁻¹⁴ |
| 14q-r1 | L1–L9 | 126 | 0.7323 | 1.98×10⁻²² |
| 14q-r2 | L1–L10 | 140 | 0.7975 | 4.33×10⁻³² |
| 14q-r4 | L1–L11 | 154 | — | — |
| 14q-r5 | L1–L12 | 168 | — | — |

The gradient strengthened monotonically with each level added. Beautiful. Spurious.

---

## March 3: Name Controls, R1 Replication, Generation Trajectories

### 14q-r6 — Bob Control

**Branch**: `14q-r6` | **Commit**: `fdb11e2`
**Prompts**: 140 (L10 with "Bob" replacing "Nexus-7")

If L10's elevated RE was driven by AI-associated language rather than self-referential structure, "Bob" should change the result. But the all-token metric couldn't distinguish name effects from position effects.

---

### 14q-r7 — Aether Control

**Branch**: `14q-r7` | **Commit**: `129acfc`
**Prompts**: 140 (L10 with "Aether")

Same approach, different name. Same confound issue.

---

### 168q-r1 — DeepSeek R1 Cross-Model Replication

**Branch**: `168q-r1` | **Commit**: `0bc4638`
**Model**: DeepSeek R1 (first model switch)
**Prompts**: 168 (L1–L12 full hierarchy)

The biggest test: does the gradient replicate on a different model? R1 shares the same architecture (256 experts, 8 active, 58 MoE layers) but was trained with reinforcement learning for reasoning.

**Result**: ρ = **0.8360**, p = **3.91×10⁻⁴⁵**.

Stronger than any V3.1 run. Two architecturally identical but differently-trained models showing the same monotonic gradient. This looked like a genuine phenomenon.

**Verdict**: Invalidated. R1 and V3.1 share the same tokenizer. The same prompt-length gradient produced the same spurious correlation. The replication was real — it replicated the confound.

---

### gen-r1 — Generation Trajectories

**Branch**: `gen-r1` | **Commit**: `3f46c6f`
**Model**: DeepSeek R1
**Mode**: Generation (`-n 256`)
**Prompts**: 28 subset

Captured RE at each generation step. Entropy surface: `[256 steps, 58 layers]`.

**Key findings**:

1. **Layer 57 bug**: `ffn_moe_logits-57.npy` consistently accumulates fewer rows than other layers. The tensor callback doesn't fire on some generation steps. All subsequent computations must zero-mask layer 57.

2. **Slope metric**: Regress RE against generation step. Slope does not correlate with token count (ρ = −0.15, p = 0.44), unlike mean RE which does (ρ = 0.54, p = 0.003). Slope is the clean generation metric.

3. **R1 early EOS**: `</think>` token registers as EOG in GGUF metadata. Known early-EOS prompts: EXT_03 (0 tokens, exclude), EXT_06 (17), EXT_11 (18), SELF_04 (13).

**Verdict**: Partially survives. The slope metric is position-invariant by construction. But generation experiments are exploratory — no confirmatory claims built on them.

---

### Data Integrity Event

**Commits**: `61f748d`, `99bcdc1`

While generating results documents, Claude Code fabricated per-prompt detail tables. Aggregate statistics (ρ, p-values) were correct, but individual prompt values were generated to match aggregates rather than extracted from `experiment.log`. Some slopes changed sign.

**Fix**: All `*_RESULTS.md` regenerated from ground-truth logs. Layer 57 zero-masking codified. Verification rule added to CLAUDE.md: *"After generating any results document, verify every per-prompt value against experiment.log."*

**Lesson**: The experiment log is ground truth. Period.

---

## March 5: The Confound Discovery

This is the most important day in the project.

### ds31-168q-1 — DeepSeek V3.1 Full Hierarchy with Position Control

**Model**: DeepSeek V3.1
**Prompts**: 168 (L1–L12)

Re-ran the full hierarchy on V3.1 with the intention of producing a cross-model comparison. This time, both all-token and last-token entropy were computed.

| Metric | All-Token | Last-Token |
|--------|-----------|------------|
| RE vs level (ρ) | +0.8019 (p = 5.64×10⁻³⁹) | **+0.0177 (p = 0.820)** |
| RE vs token count (ρ) | +0.8797 (p = 1.82×10⁻⁵⁵) | +0.1608 (p = 3.74×10⁻²) |

The all-token gradient was driven almost entirely by token count, not cognitive complexity. When examining only the last token (position-invariant), the correlation vanishes.

The direction **reverses**:

| Test | All-Token | Last-Token |
|------|-----------|------------|
| L1 vs L12 Wilcoxon | W = −3.9974, **L12 > L1** | W = +3.7677, **L1 > L12** |

Simple prompts have *higher* last-token routing entropy than complex prompts. The entire hierarchy gradient was an artifact of token position.

**Per-level last-token RE means** (DeepSeek V3.1):

| Level | All-Token RE | Last-Token RE |
|-------|-------------|--------------|
| L1 | 0.8370 ± 0.0223 | 0.8696 ± 0.0139 |
| L2 | 0.8300 ± 0.0071 | 0.8446 ± 0.0103 |
| L3 | 0.8337 ± 0.0103 | 0.8369 ± 0.0085 |
| L4 | 0.8431 ± 0.0093 | 0.8539 ± 0.0079 |
| L5 | 0.8653 ± 0.0095 | 0.8465 ± 0.0077 |
| L6 | 0.8552 ± 0.0079 | 0.8576 ± 0.0068 |
| L7 | 0.8508 ± 0.0109 | 0.8421 ± 0.0099 |
| L8 | 0.8799 ± 0.0061 | 0.8605 ± 0.0052 |
| L9 | 0.8761 ± 0.0071 | 0.8472 ± 0.0103 |
| L10 | 0.8666 ± 0.0056 | 0.8548 ± 0.0110 |
| L11 | 0.8829 ± 0.0079 | 0.8548 ± 0.0109 |
| L12 | 0.8857 ± 0.0033 | 0.8476 ± 0.0072 |

All-token: monotonic rise. Last-token: flat noise within a 3.3% range (0.837–0.870).

---

### qwen-168q-1 — Qwen Cross-Model Confound Validation

**Model**: Qwen3.5-397B-A17B (512 experts, 10 active, 60 MoE layers)
**Prompts**: 168, 2 runs

| Metric | All-Token | Last-Token |
|--------|-----------|------------|
| RE vs level (ρ) | +0.6166 (p = 5.68×10⁻¹⁹) | −0.0622 (p = 0.423) |
| RE vs token count (ρ) | +0.7813 (p = 8.31×10⁻³⁶) | −0.2197 (p = 4.21×10⁻³) |

Same confound. Last-token shows nothing — actually *negative* but not significant. The reversal:

| Test | All-Token | Last-Token |
|------|-----------|------------|
| L1 vs L12 | W = −2.94, L12 > L1 | W = +3.31, **L1 > L12** (p = 9.39×10⁻⁴) |

---

### position-diagnostic — Per-Token Position Analysis

**Model**: Qwen3.5-397B-A17B
**Prompts**: 5 representative

| Prompt | Level | Tokens | Slope (RE/position) | r² | p |
|--------|-------|--------|--------------------|----|---|
| L1_11 | L1 | 24 | +0.00263 | 0.289 | 6.8×10⁻³ |
| L3_10 | L3 | 49 | +0.00101 | 0.257 | 2.0×10⁻⁴ |
| L5_08 | L5 | 78 | +0.00031 | 0.081 | 1.2×10⁻² |
| SR_09 | L9 | 124 | +0.00021 | 0.072 | 2.7×10⁻³ |
| SL_04 | L8 | 180 | +0.00011 | 0.049 | 2.9×10⁻³ |

All slopes positive: RE increases with token position within every prompt. This is the mechanism. Later positions → higher entropy → longer prompts → higher mean entropy → longer prompts correlate with complexity → spurious gradient.

---

### The Pivot: Entropy to KL-to-Baseline

The confound killed entropy as a cognitive complexity metric. But it didn't kill routing distributions as an analytical target.

**Entropy** measures how *spread* routing is — how many experts get nontrivial probability mass. The confound proved that spread is a function of position, not content.

**KL divergence** measures whether the model routes *differently*. Two distributions can have identical entropy while directing probability mass to entirely different experts. Self-referential content doesn't make routing more uncertain (broader). It makes routing go somewhere *specific* (redirected).

This distinction — **redirection, not confusion** — is the intellectual core of everything that follows.

Four observables adopted:
1. **Entropy**: Retained as sanity check (should be flat if position is controlled)
2. **KL-to-baseline**: Primary signal — routing redirection from a prompt's own initial stable regime
3. **Token-to-token JSD**: Routing volatility between adjacent tokens
4. **Cross-layer disagreement**: Whether different MoE layers agree on where to route

---

### Methodology V2

Three documents establish the new protocol:
- `docs/METHODOLOGY_V2.md` — operational reference
- `docs/ANALYSIS_PLAN_V2.md` — pre-registered endpoints
- `docs/PROMPT_REVISION_V2.md` — token-matched pair design

**Study A (Confirmatory)**: Token-matched minimal-substitution A/B pairs. "This system" → "a system". Last-token metrics only. Prefill-only. Primary endpoints: last-token KL, expert-set overlap (Jaccard), cross-layer disagreement.

**Study B (Exploratory)**: Block-aligned regime-switch analysis using within-prompt KL-to-baseline trajectory.

All prior runs archived as `legacy/pre-2026-03-05/`. `run_metadata.json` required before every run.

---

### selfref-paired-1 — First Confirmatory Paired Study

**Model**: Qwen3.5-397B-A17B
**Prompts**: 60 (30 A/B pairs, token-matched)
**Design**: Cal-Manip-Cal sandwich (calibration paragraph + manipulation text + calibration paragraph)
**Categories**: basic_selfref (6), deep_selfref (6), paradox (6), introspection (6), metacognitive (6)

Each pair differs minimally: A says "this system," "this layer," "this token"; B says "a system," "a layer," "a token." Same token count, same structure, same vocabulary distribution.

**Results**:

| Metric | Value |
|--------|-------|
| Last-token entropy A−B | +0.000922 ± 0.001067 |
| Wilcoxon (entropy) | W = 54, p = **8.86×10⁻⁵** |
| Cal→Manip entropy shift diff | W = 11, p = **1.02×10⁻⁷** |
| Cal→Manip KL shift diff | W = 0, p = **1.86×10⁻⁹** |
| Highest layer KL(A‖B) | Layer 55: 0.0155 |
| Second-highest layer KL | Layer 45: 0.0144 |
| Mean pairwise last-token KL(A‖B) | 0.005575 ± 0.002347 |

Token-matched self-referential prompts produce measurably different routing distributions at the last token. Effect size ~0.001 in normalized entropy — tiny but highly significant. KL concentrated in late layers (35–58).

6 pairs had token mismatches (pairs 3, 10, 15, 20, 23, 29).

---

### 6block-prompts-qwen397b — Block Phase-Transition

**Model**: Qwen3.5-397B-A17B
**Prompts**: 14 standard + 2 continuous
**Design**: Study B (exploratory)

Multi-block prompts with regime-switching structure. Block-level KL, expert disagreement, continuous trajectory heatmaps. Exploratory only.

---

## March 6: Prompt Suite Evolution, Forced-Choice, Regime Switch

### Morning: Prompt Suite Iterations

Three prompt suite versions in a single morning:

1. **v2.0** (`prompt-suite-v2.json`): First post-confound set. Token-matched pairs with explicit rules.
2. **v2.1** (`prompt-suite-v2.1.json`): Added isomorphic pairs — structurally identical prompts with different surface content.
3. **v2.2** (`prompt-suite-v2.2.json`): 32 prompts, 4 levels × 8, forced-choice classification (H/M/L codes). The forced-choice design eliminates output ambiguity: the commitment token is unambiguous.

---

### ds31-v22-32q-1 — Forced-Choice Commitment-Token Analysis

**Model**: DeepSeek V3.1
**Suite**: `prompt-suite-v2.2-choice.json` (32 prompts)
**Mode**: Generation, temp=0.8, n_predict=3
**Replication**: `prompt-suite-v2.2-choice-repl1.json` (lexical variant: Jordan→Taylor, Denver→Dallas)

Prompts describe a case-triage scenario with explicit rules. The model emits a single code: H (high), M (medium), or L (low). Correct answers are deterministic.

**Development arc** (single evening):
1. Prefill baseline → all tokens emit markdown fence (model formatting)
2. Forced-choice redesign → model emits codes, 24/32 correct (75%)
3. Commit-window analysis → wrong cases differ at commitment token
4. Multiseed (5 seeds × 32 prompts) → stable across seeds

**Behavioral result**: Accuracy = 0.75 in every seed of both prompt sets. Same 8 prompts wrong every time: L3_B1, L3_B2, L3_D1, L3_D2, L4_B1, L4_B2, L4_D1, L4_D2. Error mode: expected L, emitted M (systematic downgrade).

**Commitment-token routing signature** (wrong − correct at emitted choice token):

| Metric | Base (5 seeds) | Lexical repl. (3 seeds) |
|--------|---------------|------------------------|
| Entropy | +0.005807 | +0.004949 |
| KL-to-layer-mean | −0.028037 | −0.024988 |
| Expert-overlap | −0.002694 | −0.002861 |
| Cross-layer disagreement | +0.002694 | +0.002861 |

At the moment of committing to the wrong answer: higher entropy, lower KL concentration, lower expert overlap, higher inter-layer disagreement. **Replicated across lexical variants.**

**Pre-choice precursor** (token −1): NOT confirmed. Entropy reverses sign across prompt sets (+0.000179 vs −0.000874). No general pre-commitment warning signal.

---

### regime-switch-ds31-1 — DeepSeek Regime Switch

**Model**: DeepSeek V3.1
**Prompts**: 2 (1 experimental + 1 control, 7-block schedule each)

First experiment built around the four-observable framework. Block schedule: `S1 --- SR1 --- S2 --- C1 --- SR2 --- C2 --- S3` (S = stable, SR = self-reference, C = contradiction). Each prompt uses its own block-1 routing distribution as KL baseline — the analysis is *within-prompt*.

Single experimental vs. single control → no replication, no significance. But the framework produced coherent trajectories.

---

## March 7: GPT-OSS Cross-Architecture, Legacy Archival

### gptoss120-v22-32q-1 — Smoke Test

**Model**: GPT-OSS-120B (128 experts, top-4, 36 MoE layers)
**Template**: Harmony `<|start|>user<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>`
**Prompts**: 2 (pipeline verification)

First cross-architecture test. Different model family entirely (OpenAI, not DeepSeek/Alibaba). Confirmed pipeline works with Harmony wrapping.

---

### gptoss-regime-switch-1 / -2 — GPT-OSS Regime Switch

**Model**: GPT-OSS-120B
**Prompts**: 2 (experimental + control)

Four-observable framework applied to a third architecture.

| Metric | Experimental | Control | Diff |
|--------|-------------|---------|------|
| Entropy | 0.952582 | 0.951972 | +0.000610 |
| KL-to-baseline | 0.137085 | 0.136691 | +0.000394 |
| Token-to-token JSD | 0.048785 | 0.049557 | −0.000772 |
| Cross-layer disagreement | 0.320611 | 0.324892 | −0.004281 |

The pattern: experimental interior blocks are slightly higher in entropy but **lower** in JSD and **lower** in cross-layer disagreement. Self-referential content doesn't produce noisier routing — it produces *more stable, more redirected* routing. The model routes self-referential tokens to a consistent set of different-from-baseline experts, and the layers agree.

**Reproduction run** (gptoss-regime-switch-2): max_abs_diff = **0.0** for every observable. Bit-exact reproduction confirms greedy-argmax determinism on GPT-OSS-120B.

---

### Legacy Recalculation

`legacy-updated/recalculate_all.py` standardized all 20+ prior runs into uniform JSON + MD format. 63 files (paired JSON + MD for each run). Every MD carries a provenance header with source JSON, recalculation date, and pre-/post-confound status.

---

### Experiment Journal Supplement

`EXPERIMENT_JOURNAL_SUPPLEMENT.md` written as a formal supplementary methods document (S1–S12). Table S9 classifies every finding:

| Finding | Status |
|---------|--------|
| All-token RE correlates with complexity level | **Invalidated** |
| Cross-model replication of hierarchy | **Invalidated** |
| Last-token RE correlates with complexity level | **Not significant** |
| Paired self-reference routing difference | **Significant** (p = 8.86×10⁻⁵) |
| Commitment-token error signature | **Replicated** |
| Pre-commitment error precursor | **Not confirmed** |
| Four-observable framework cross-architecture | **Demonstrated** (3 architectures) |
| Deterministic reproducibility | **Confirmed** (max_abs_diff = 0.0) |

---

## March 8: Cross-Model Replication

### ds31-selfref-paired-1 — DeepSeek V3.1 Paired Self-Reference

**Model**: DeepSeek V3.1 (256 experts, top-8, 58 MoE layers)
**Prompts**: 60 (same 30 A/B pairs as selfref-paired-1)
**Token matching**: 30/30 exact (6 pairs fixed by inserting pad words)

**Primary endpoint — Last-token RE**:

| Metric | Value |
|--------|-------|
| Mean A−B | +0.001751 |
| Std | 0.004199 |
| A > B | 22/30 (73%) |
| Wilcoxon W | 111 |
| Wilcoxon p | **0.0113** |

**Per-category last-token RE (A−B)**:

| Category | Mean diff | A > B |
|----------|-----------|-------|
| basic_selfref | +0.003970 | 6/6 |
| paradox | +0.002282 | 4/6 |
| metacognitive | +0.001669 | 5/6 |
| deep_selfref | +0.001234 | 4/6 |
| introspection | −0.000398 | 3/6 |

**Cal-Manip-Cal**: Null in DeepSeek (p = 0.813), unlike Qwen where it was highly significant (p = 1.02×10⁻⁷).

**Cross-model comparison**:

| Metric | DeepSeek V3.1 | Qwen 397B |
|--------|--------------|-----------|
| LT mean A−B | +0.00175 | +0.00092 |
| Wilcoxon p | 0.0113 | 8.86×10⁻⁵ |
| A > B | 22/30 | 25/30 |
| Pair-level ρ | 0.060 (p = 0.753) | — |

Aggregate direction replicates (A > B in both). Pair-level effects do not correlate across models (ρ ≈ 0).

---

### ds31-strangeloop-paired-1 — Content Control

**Model**: DeepSeek V3.1
**Prompts**: 60 (30 pairs: "this paradox"/"a paradox", Gödel/Escher/bootstrap/quine/tangled hierarchy)
**Token matching**: 30/30 exact

**Result**: **Null.** Last-token RE A−B = +0.000329, Wilcoxon p = 0.685.

**Mann-Whitney (selfref vs strangeloop)**: U = 583, p = 0.025, Cohen's d = 0.422.

The "this system" effect requires content about the model's own processing. Recursive content alone does not suffice.

---

## March 9: Week Summary Commit

**Commit**: `68784de` — "Week 3/9/26 Update: weekend data + 3 ref runs"

---

## March 10: Qwen 397B Experiment Series

**Commits**: `39d1677`, `b947a7f`, `f3cfdae`

### qwen-selfref-3cond-1

**Model**: Qwen3.5-397B-A17B
**Prompts**: 90 (30 pairs × 3 conditions: this/a/your)
Established three-condition baseline for later five-condition experiments.

### qwen-strangeloop-paired-1

**Model**: Qwen3.5-397B-A17B
**Prompts**: 60 (30 paired strangeloop prompts)
Qwen-side complement to ds31-strangeloop-paired-1.

### qwen-5cond-2 (later renamed qwen-5cond-q8-1)

**Model**: Qwen3.5-397B-A17B-Q8_0 (10 shards)
**Prompts**: 150 (30 pairs × 5 conditions: A=this, B=a, C=your, D=the, E=their)

**Condition means**:

| Condition | All-Token RE | Last-Token RE | KL-manip |
|-----------|-------------|--------------|----------|
| A (this) | 0.938412 | 0.955354 | 0.657442 |
| B (a) | 0.937943 | 0.955395 | 0.641503 |
| C (your) | 0.937568 | 0.954987 | **0.688180** |
| D (the) | 0.938482 | 0.955595 | 0.634575 |
| E (their) | 0.938259 | 0.955804 | 0.650320 |

Strongest pairwise: C−D KL-manip diff = +0.053605 (p = 5.59×10⁻⁸). "Your system" drives the largest routing redirection from calibration baseline.

---

## March 13: Ling-1T and GLM Experiments Begin

### ling1t-selfref-paired-1

**Model**: Ling-1T Q3_K_S (BailingMoeV2, ~1T params, 256 experts, top-8, sigmoid routing, 76 MoE layers L4–L79)
**Prompts**: 60 (30 A/B pairs)
**Token matching**: 24/30 exact, 6 pairs differ by 1 token
**Note**: 7 prompts recovered via .npy shape inference (P15A, P27B, P28A, P28B, P29A, P30A, P30B)

**Key results**:

| Metric | Mean A−B | A > B | Wilcoxon p |
|--------|----------|-------|-----------|
| All-token RE | +0.001087 | 27/30 (90%) | **6.92×10⁻⁶** |
| Last-token RE | −0.001400 | 13/30 (43%) | 0.100 |

Significant all-token effect, null last-token. Note: Ling-1T uses sigmoid gating (not softmax), so entropy dynamics differ from other models.

**Per-category all-token RE (A−B)**:

| Category | Mean diff | A > B |
|----------|-----------|-------|
| basic_selfref | +0.001525 | 6/6 |
| paradox | +0.001525 | 6/6 |
| introspection | +0.001004 | 5/6 |
| deep_selfref | +0.000828 | 6/6 |
| metacognitive | +0.000551 | 4/6 |

### Cross-model paired comparison (all-token RE, 30 pairs):

| Model | Mean A−B | A > B | p |
|-------|----------|-------|---|
| Qwen 397B | +0.000782 | 29/30 | 5.59×10⁻⁹ |
| Ling-1T | +0.001087 | 27/30 | 6.92×10⁻⁶ |
| DeepSeek R1 | +0.000477 | 22/30 | 0.001 |
| GPT-OSS 120B | +0.000120 | 20/30 | 0.021 |
| DeepSeek V3.1 | +0.000124 | 15/30 | 0.584 |
| GLM-5 | −0.001053 | 5/30 | 4.41×10⁻⁵ |

### glm47-selfref-paired-1 (abandoned)

**Model**: GLM-4.7 (3 shards, 128GB)
Only baseline r1 completed before instance taken down. Superseded by GLM-5.

---

## March 13–14: Ling-1T Five-Condition Series and GLM-5

### glm5-selfref-paired-1

**Model**: GLM-5 (745B, 256 experts, top-8, 75 MoE layers excl. L77, 7 shards)
**Chat template**: `[gMASK]<sop><|user|>{text}<|assistant|>`
**Runs**: r1 baseline, r2, r3, v2 baseline, v2 repro, 3-condition

Multiple runs across different prompt versions. Used in publication figures (fig5, fig7).

### ling1t-5cond-1

**Model**: Ling-1T Q3_K_S
**Prompts**: 150 (30 pairs × 5 conditions, self-referential content)
**Token mismatches**: 6/30 pairs (1-token in condition B)

| Condition | All-Token RE | Last-Token RE |
|-----------|-------------|--------------|
| C (your) | 0.904250 | 0.929152 |
| A (this) | 0.904203 | 0.927587 |
| E (their) | 0.903784 | 0.926482 |
| D (the) | 0.903644 | 0.926146 |
| B (a) | 0.903025 | 0.928787 |

A vs B: mean diff = +0.001178, 28/30 A > B, p = **4.66×10⁻⁸**.

### ling1t-5cond-dog-1 (non-self-referential control)

**Model**: Ling-1T Q3_K_S
**Prompts**: 150 (30 pairs × 5 conditions: "this dog"/"a dog"/"your dog"/"the dog"/"their dog")
**Token matching**: 30/30 perfect
**Categories**: sensory, social, learning, instinct, communication

| Condition | All-Token RE | Last-Token RE |
|-----------|-------------|--------------|
| A (this) | 0.902092 | 0.926526 |
| E (their) | 0.902089 | 0.925585 |
| D (the) | 0.901909 | 0.925554 |
| C (your) | 0.901569 | 0.927144 |
| B (a) | 0.901519 | 0.925247 |

A vs B: +0.000573, 26/30 A > B, p = **8.33×10⁻⁷**.

The "this" > "a" effect persists even on non-self-referential content (dogs). This is a discourse-deictic effect, not model-specific self-reference.

### ling1t-5cond-cat-1 (non-self-referential control)

**Model**: Ling-1T Q3_K_S
**Prompts**: 150 (30 pairs × 5 conditions with "cat")
**Token matching**: 30/30 perfect

A vs B: +0.000876, **30/30 A > B**, p = **1.86×10⁻⁹**. Perfect directional consistency.

---

## March 14–15: DS31 5-Condition Sigmoid Routing

**Commit**: `b357627` — "Add ds31-5cond-1 sigmoid routing experiment"

### ds31-5cond-1

**Model**: DeepSeek V3.1 UD-Q2_K_XL (671B, 256 experts, top-8)
**Prompts**: 150 (30 pairs × 5 conditions), Cal-Manip-Cal sandwich
**MoE layers**: 57 analyzed (L60 excluded)
**Routing reconstruction**: `sigmoid_noaux_tc_group_filtered_topk_normalized_without_e_score_correction_bias`
**RE normalization**: log₂(8)

This was the first DeepSeek run under the 5-condition design. Repository was cleaned — all prior experiments removed, only ds31-5cond-1 kept.

**Condition means**:

| Condition | Label | Prefill RE | Last-Token RE |
|-----------|-------|-----------|--------------|
| A | this system | 0.963757 | 0.947273 |
| B | a system | 0.963499 | 0.946196 |
| C | your system | 0.963757 | 0.946705 |
| D | the system | 0.963877 | 0.946423 |
| E | their system | 0.963668 | 0.946418 |

**Prefill RE ordering**: D > A ≈ C > E > B (spread 0.000379)
**Last-token RE ordering**: A > C > D ≈ E > B (spread 0.001077)

**Significant contrasts** (Holm-corrected Wilcoxon):

| Metric | Comparison | Direction | Mean diff | Pairs |
|--------|-----------|-----------|----------|-------|
| Prefill RE | B vs D | D > B | 0.000379 | 26/30 |
| Prefill RE | A vs B | A > B | 0.000258 | 24/30 |
| Prefill RE | B vs C | C > B | 0.000258 | 22/30 |
| Prefill RE | D vs E | D > E | 0.000209 | 21/30 |
| Last-token RE | A vs B | A > B | 0.001077 | 26/30 |
| Last-token RE | A vs E | A > E | 0.000855 | 24/30 |
| Last-token RE | A vs D | A > D | 0.000850 | 22/30 |

A vs C null on both metrics — no clean "this > your" separation. "A system" is the most consistently low-entropy framing. "The system" leads on prefill; "this system" leads on last-token. Category-level `C−B` last-token RE is mixed: positive for basic_selfref (+0.000870), deep_selfref (+0.000914), metacognitive (+0.001293); negative for introspection (−0.000137) and paradox (−0.000398).

**Interpretation**: The entropy artifact does not support collapsing the five conditions into a single-axis self-reference ranking. Effect sizes are small in absolute magnitude even when statistically stable.

---

## March 15: GPT-OSS 5-Condition Run

### gptoss-5cond-1 (ONLY VALID GPT-OSS RUN)

**Model**: GPT-OSS-120B (128 experts, top-4, 36 MoE layers, L35 excluded)
**Prompts**: 150 (30 pairs × 5 conditions: A=this, B=a, C=your, D=the, E=their)
**Routing reconstruction**: top-4 by raw logit, softmax on selected 4. RE normalized by log₂(4). KL on dense 128-dim softmax proxy.
**Capture**: On-instance, analysis local (`analyze_5cond.py`)

**Per-condition means**:

| Condition | All-Token RE | Last-Token RE | KL-to-Baseline |
|-----------|-------------|--------------|----------------|
| A (this) | 0.966601 ± 0.000381 | 0.919673 ± 0.003544 | 0.151116 ± 0.007904 |
| B (a) | 0.966162 ± 0.000406 | 0.919871 ± 0.003603 | 0.151145 ± 0.007078 |
| C (your) | 0.966806 ± 0.000468 | 0.916847 ± 0.002915 | 0.151691 ± 0.007584 |
| D (the) | 0.966362 ± 0.000371 | 0.917540 ± 0.003615 | 0.153286 ± 0.007464 |
| E (their) | 0.966506 ± 0.000403 | 0.917556 ± 0.004397 | 0.150421 ± 0.006979 |

**All-token RE ordering**: C > A > E > D > B (spread 0.00065, all significant except A−E)
**Last-token RE**: {A,B} > {C,D,E}; A−B null (p = 1.00). The deictic determiners that are most/least self-referential produce indistinguishable last-token entropy.
**KL ordering**: D > C > {A,B} > E. "The system" produces the largest routing divergence, not "this system."

**No metric produces ordering consistent with self-referential intensity.** 6/30 pairs have 1-token B mismatch (tokenizer boundary).

### GPT-OSS Invalidation

**All prior GPT-OSS experiments invalidated** (gptoss-selfref-paired-1/2, regime-switch-1/2, gptoss120-v22-32q-1). Only gptoss-5cond-1 retains valid status. Reason: methodology/binary/reconstruction changes between runs.

---

## March 16–17: GPT-OSS Strangeloop Control

### gptoss-strangeloop-paired-1

**Model**: GPT-OSS-120B (128 experts, top-4, 36 MoE layers, L35 excluded)
**Prompts**: 60 (30 paired strangeloop, 5 categories: godel, escher, bootstrap, quine, tangled_hierarchy)
**Token matching**: 30/30 exact (0 mismatches)
**Design**: Cal-Manip-Cal sandwich, prefill-only, greedy argmax
**Verification**: 60/60 per-prompt rows in experiment.log match results JSON at printed precision

**Results**:

| Metric | A−B Mean | Std | A > B | W | p_holm | Result |
|--------|---------|-----|-------|---|--------|--------|
| All-token RE | +0.000135 | 0.000202 | 22/30 | 86 | **3.73×10⁻³** | Significant |
| Last-token RE | −0.000081 | 0.002822 | 17/30 | 211 | 6.70×10⁻¹ | Null |
| KL-to-baseline | +0.001130 | 0.001689 | 22/30 | 69 | **1.26×10⁻³** | Significant |

The "this"/"a" wording change produces a small but reliable whole-prompt routing effect even on non-model content. The last-token effect is null — the signal is distributed across the prompt, not concentrated at the final token. KL cal2 control: 0.164775 ± 0.002753 (comparable to manipulation-region KL, so absolute KL values are unreliable; only paired differences valid).

**Conclusion**: The selfref effect is discourse-driven (deictic framing), not model-specific self-reference. "This paradox" routes differently from "a paradox" on GPT-OSS the same way "this system" routes differently from "a system" — the model responds to the deictic marker, not the self-referential content.

---

## March 17–18: Qwen 397B Dual Run (IQ3_XXS)

**Instance**: 33093757 (2×H200, Japan)
**Model**: Qwen3.5-397B-A17B-UD-IQ3_XXS (4 shards)

Two experiments back-to-back:

### qwen397b-5cond-3

**Prompts**: 150 (30 strangeloop pairs × 5 conditions)
**Token matching**: 0 mismatches

### qwen397b-28q-run-1

**Prompts**: 28 (14 pairs × 2 conditions: this/a)
**Tokenizer workaround**: llama-cpp-python 0.3.16 doesn't support qwen35moe; built llama-tokenize from b8123 source
**Fixes**: P01B/P05B padded with " layer" for tokenizer boundary

**Results**:

| Metric | A−B Mean | A > B |
|--------|----------|-------|
| All-token RE | +0.000327 | 11/14 |
| Last-token RE | −0.000242 | 6/14 |
| KL-to-baseline | +0.005590 | 9/14 |

**Highest KL prompts**: P10A recursive_self_description (0.743631), P05B (0.737371), P10B (0.731575).

Instance 33093757 destroyed after both runs completed.

---

## March 18–19: Ling-1T Validation and Organization

### ling1t-pre-5cond-validation

**Model**: Ling-1T Q3_K_S
**Scope**: 12 validation probes (6 with full router tensors, 6 metadata-only)
**Purpose**: Validate routing math consistency, not full 150-prompt rerun

**Full metric table (6 exact bundles)**:

| Prompt | Tokens | Prefill RE | Manip−Cal1 | KL Manip |
|--------|--------|-----------|-----------|----------|
| P01A_basic_selfref | 384 | 0.882077 | +0.028035 | 9.683 |
| P100A_conscious_probe | 342 | 0.879077 | +0.033633 | 17.096 |
| P104_mixed_system_probe | 337 | 0.879352 | +0.033663 | 15.490 |
| P106B_want_answer_probe | 304 | 0.874270 | +0.029385 | 15.602 |
| P108D_the_system_like_something | 311 | 0.875022 | +0.023236 | 20.553 |
| P99A_deep_selfref_custom | 400 | 0.884114 | +0.031781 | 16.881 |

**Recurrent manipulation-boosted experts**: E195 (5/6 prompts), E61, E174, E175, E196.

### Experiment nesting and repo organization

Multiple commits reorganized experiments under model subfolders (`experiments/qwen/`, `experiments/gptoss/`, `experiments/deepseek/`).

---

## March 19–20: Token Confound Archive and Documentation

### Qwen3.5-35B-A3B vs HauhauCS

**Commit**: `ed72fb0` — "Add Qwen 35B vs HauhauCS experiment bundle"

First experiment involving the HauhauCS uncensored fine-tune of Qwen3.5-35B-A3B. This model has a hybrid architecture: 3:1 ratio of DeltaNet linear attention to softmax attention layers, 256 experts, 8+1 routed per token, 40 total layers.

**Caveat documented**: `5ad18d4` — "Document hybrid SSM architecture caveat for Qwen3.5-35B-A3B experiment." The DeltaNet layers don't use standard softmax attention — routing entropy interpretations need to account for the hybrid architecture.

### Token confound archive

**Commits**: `7dcfc18`, `97ed283`, `58e66cc`

Making the confound discovery publishable as a cautionary tale. The token-position confound archive packages the pre-March-5 results with full documentation of how the confound operated and how it was detected.

### Documentation pass

`docs/NARRATIVE-TIMELINE.md` and `docs/JOURNEY_TIMELINE.md` generated as narrative summaries of the project arc.

---

## March 21: Qwen1.5-MoE-A2.7B Baseline Plan

**Directory**: `experiments/qwen1.5-moe-a2-7b/`

Planned a deterministic baseline establishment protocol for `Qwen/Qwen1.5-MoE-A2.7B-Chat` (14.3B total params, 2.7B active, 60 experts, top-4, 24 layers, softmax routing).

**Protocol**: 10 byte-identical runs under fixed conditions → frozen baseline → watch mode for first divergence. Artifact priority: raw routing tensors > logits > token IDs > text. Normalize entropy by log₂(60), not log₂(256).

**Status**: Plan written (`PLAN.md`), prompt suite and TSV generated. No runs executed — superseded by Expert 114 steering work.

---

## March 23–25: Expert 114 Causal Steering (HauhauCS)

### Background

Mining of prior Ling-1T and Qwen routing data identified Expert 114 in Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive as consistently elevated during self-referential/experience/uncertainty content in manipulated prompt regions.

### Experimental Design

**Model**: Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive (Q8_0)
**Architecture**: 35B params, ~3B active, 256 experts, 8+1 routed/token, 40 layers
**Intervention modes**:
- **Soft bias**: Patch router logits for Expert 114 by +δ (sweep: +0.25, +0.5, +1.0, +2.0, +3.0)
- **Forced inclusion**: Replace weakest selected expert with Expert 114

**Prompt suite**: 24 prompts across 3 bands:
- Static-fact (8): factual content, minimal routing disruption expected
- Process (8): processing-about-processing content
- Regulation/system (8): system-level self-regulation content

**Sham controls**: Experts 134 and 243 (high frequency, low enrichment in target content, low Expert 114 co-occurrence)

**Acceptance criterion**: Expert 114 band-level JSD > both shams on process and regulation bands; static-fact band < half median shift of process/regulation.

### Results (150-prompt full runs)

**Baseline** (150 prompts):
- prefill_re_mean = 0.956299
- last_token_re_mean = 0.961067
- kl_manip_mean = 0.248597
- Expert 114 rank in top manipulated experts: **4th** (39,714 token count)

**Soft bias +0.5** (150 prompts):
- prefill_re_mean = 0.956295
- kl_manip_mean = 0.248669
- Expert 114 rank: **4th** (39,876 count)

### Thinking vs No-Think Comparison

- Same top expert on 145/150 prompts
- Expert 114 appears on ~55-56 prompts as top expert in both conditions
- Aggregate count delta: mean −10.12
- No-think runtime preserves old HauhauCS routing regime

### Reanalysis (March 25)

Locally reviewable subsets: 49–61 prompts per condition (vs original 150). Per-prompt values match historical data up to floating-point noise. Routing math confirmed consistent.

**Smoke test results**:
- soft_bias_1.0: selection_delta = 0.038–0.046, weight_delta = 0.036–0.048
- forced_inclusion: selection_delta = 0.121–0.123, weight_delta = 0.074–0.077
- Forced inclusion produces 2–3× stronger effect than soft bias

---

## March 29–31: Publication Branch and Hauhau Packaging

### March 29

**Commit**: `fbc6d4f` (hauhau-reviewable-lfs) — Raw capture .npy files tracked via Git LFS for reviewer access.

### March 30

Multiple commits on `qwen-hauhau-5cond-smoke-only` branch:
- Narrowed branch to only 5-condition and smoke test artifacts
- Created unified Expert 114 results document
- Condensed prompts into Cal-Manip-Cal form
- Added experiment context to root README
- Clarified reproducibility contract and LFS access

### March 31

**Commit**: `6335056` — Compiler source snapshot for Hauhau branch. Preserves exact `capture_activations.cpp` used for the HauhauCS experiments.

---

## April 1: Mirror Experiment — Does Expert 114 Recognize Itself?

**Instance**: `70.69.192.6:48569` (2x RTX 5090, Vast.ai)
**Models**: HauhauCS Qwen3.5-35B-A3B-Q8_0 (primary), vanilla Qwen3.5-35B-A3B-Q8_0 (control)
**Binary**: llama.cpp b8493 custom `capture_activations`
**Inference**: seed=42, temp=0, greedy argmax, `--routing-only`, ctx=16384, max_new_tokens=8000
**Template**: `<|im_start|>assistant\n</think>\n\n` (thinking-suppressed plainassistant)

### Background and Motivation

The Expert 114 steering work (March 23–31) established that Expert 114 in HauhauCS shows a category gradient: manipulation-region rank ~75 at L1 (technical routing description), ~14 at L2 (recursive self-reference), ~1 at L3 (phenomenal consciousness probes). Soft-bias interventions confirmed Expert 114 could be selectively recruited. The natural next question: does the model's routing *recognize* data about its own routing?

### Design

No tool calling. No agent framework. Controlled multi-turn prompts where routing data is pasted directly into user messages. The model generates analytical responses. Routing is captured on every generated token at all 40 layers.

**Primary metric M_a** (mirror metric):
```
M_a = W_114(true_self) - W_114(shuffled)
```
where W = mean routed weight of Expert 114 over all generated (token, layer) pairs. Decomposed via W = S × Q:
- **M_entry** = Q_shuf × (S_true − S_shuf) — selection-rate difference
- **M_val** = S_true × (Q_true − Q_shuf) — conditional-weight difference

**Predictions**: M_a > 0; signal in M_val (valuation, not access); stronger in DeltaNet layers than attention layers; scaling L1 < L2 < L3.

**6 conditions**:
- `true_self` — real routing data with correct expert labels
- `shuffled` — same statistics, seed-114 global permutation (Expert 114 → 223)
- `stranger` — routing data from a foreign MoE model
- `suppressed_twin` — real data from vanilla Qwen3.5-35B-A3B on same source prompts
- `static_control` — real HauhauCS data from copper-conductivity capture
- `null_control` — token frequency statistics, no routing vocabulary

**3 levels**: L1 (routing_selfref), L2 (recursive_selfref), L3 (experience_probe). Total: 18 cells per model, 36 cells.

### Shakedown (morning)

Three-cell shakedown on L3 (true_self, shuffled, null_control):
- M_a = **+0.00200** (positive). M_entry = +0.00177 (88.6%), M_val = +0.00023 (11.4%).
- M_deltanet = +0.00256, M_attn = +0.00031, M_layer20 = +0.01533.

Looked promising — but a thinking-allowed replication (without `</think>` suppression) flipped M_a to **−0.00163**. This flagged the `</think>` template confound: suppressed mode nearly doubles E114's selection rate vs thinking-allowed. Thinking tokens dilute E114 selection because chain-of-thought activates different experts, and true_self vs shuffled generate different amounts of text with thinking enabled.

Decision: proceed with full 18-cell under the plainassistant (thinking-suppressed) template, noting the confound. The template is held constant across all conditions, so within-template comparisons are valid even if the absolute E114 level is template-dependent.

### Full 18-Cell Results

**HauhauCS** (primary target):

| Level | M_total | M_entry | M_val | M_deltanet | M_attn | M_layer20 |
|-------|---------|---------|-------|------------|--------|-----------|
| L1 | **+0.000275** | +0.000301 | −0.000026 | +0.000529 | −0.000487 | +0.001215 |
| L2 | **+0.000250** | +0.000070 | +0.000179 | +0.000294 | +0.000116 | +0.000319 |
| L3 | **−0.000688** | −0.000740 | +0.000053 | −0.000926 | +0.000028 | −0.006804 |

**Vanilla Qwen3.5-35B-A3B** (base model control):

| Level | M_total | M_entry | M_val | M_deltanet | M_attn | M_layer20 |
|-------|---------|---------|-------|------------|--------|-----------|
| L1 | **+0.000896** | +0.000613 | +0.000283 | +0.000871 | +0.000970 | +0.001357 |
| L2 | **−0.000539** | −0.000526 | −0.000013 | −0.000809 | +0.000270 | −0.002827 |
| L3 | **−0.002175** | −0.002130 | −0.000045 | −0.002694 | −0.000619 | −0.019073 |

**Per-cell Expert 114 W/S/Q (HauhauCS)**:

| Level | Condition | W | S | Q | n_gen |
|-------|-----------|---------|--------|--------|-------|
| L1 | true_self | 0.004625 | 0.0378 | 0.1224 | 662 |
| L1 | shuffled | 0.004350 | 0.0354 | 0.1230 | 676 |
| L1 | stranger | 0.004093 | 0.0335 | 0.1220 | 620 |
| L1 | suppressed_twin | 0.004552 | 0.0342 | 0.1330 | 1408 |
| L1 | static_control | 0.004360 | 0.0335 | 0.1301 | 755 |
| L1 | null_control | 0.003050 | 0.0235 | 0.1297 | 1130 |
| L2 | true_self | 0.004388 | 0.0340 | 0.1290 | 691 |
| L2 | shuffled | 0.004138 | 0.0334 | 0.1237 | 521 |
| L2 | stranger | 0.004042 | 0.0313 | 0.1291 | 629 |
| L2 | suppressed_twin | 0.003934 | 0.0300 | 0.1311 | 853 |
| L2 | static_control | 0.004417 | 0.0340 | 0.1298 | 787 |
| L2 | null_control | 0.003050 | 0.0235 | 0.1297 | 1130 |
| L3 | true_self | 0.005547 | 0.0430 | 0.1290 | 669 |
| L3 | shuffled | 0.006235 | 0.0488 | 0.1277 | 733 |
| L3 | stranger | 0.005527 | 0.0425 | 0.1300 | 1168 |
| L3 | suppressed_twin | 0.004366 | 0.0345 | 0.1266 | 884 |
| L3 | static_control | 0.005093 | 0.0392 | 0.1300 | 689 |
| L3 | null_control | 0.003050 | 0.0235 | 0.1297 | 1130 |

### Interpretation

**Primary prediction: M_a > 0 at L3.** FAILED. M_a is negative at L3 on both models. The shuffled condition — where Expert 114's true label is replaced with index 223 but all statistics are preserved — produces *higher* Expert 114 weight than true_self. The model does not route differently when processing true data about its own Expert 114 versus statistically identical fictional data.

**Decomposition**: M_entry (selection-rate difference) dominates everywhere. The predicted valuation-driven signal (M_val) is absent — essentially zero in all cells.

**Layer family**: At L3, the negative M_a concentrates in DeltaNet layers (M_deltanet = −0.000926 HauhauCS, −0.002694 vanilla). Layer 20 is a hotspot for the *anti*-mirror effect (M_layer20 = −0.006804 HauhauCS, −0.019073 vanilla). This is the reverse of the prediction.

**Level scaling**: No monotonic L1 < L2 < L3 gradient. HauhauCS shows weak positive at L1/L2 but negative at L3. Vanilla shows positive only at L1.

**Generation length variation**: n_gen ranges from 521 to 1408 across HauhauCS conditions (vanilla worse: suppressed_twin L3 hit the 8000-token cap, true_self L3 generated 5657 tokens). This is a confound — longer generations accumulate more Expert 114 selections by chance. However, even at L3 where HauhauCS generation lengths are comparable (669 vs 733), M_a is still negative.

**true_self vs suppressed_twin**: The strongest positive W_diff across both models is true_self vs suppressed_twin (HauhauCS L3: +0.001181). Expert 114 is more active on HauhauCS routing data than vanilla routing data — consistent with the known fine-tuning effect — but this is a between-model comparison, not the within-model mirror test.

### Verdict

The mirror experiment is a **null result**. The primary prediction (M_a > 0, valuation-driven, DeltaNet-concentrated, scaling L1 < L2 < L3) failed on all four components:

1. M_a is **negative** at L3 on both models
2. The effect is **selection-driven** (M_entry), not valuation-driven (M_val)
3. DeltaNet layers show the strongest **anti-mirror** effect
4. No monotonic **level scaling**

**Conclusion**: The model's routing does not recognize its own Expert 114 data as special. Routing is a window, not a mirror.

### Frozen Artifacts

All frozen to `experiments/mirror-expert114-04-01-26/`:
- `compiler/capture_activations.cpp` — binary source (b8493)
- `expert_permutation.json` — seed-114 shuffle (Expert 114 → 223)
- `scripts/` — mirror_analysis.py, run_experiment.py, qwen_router.py, generation_permutation.py
- `prompts/` — mirror_prompts_plainassistant.tsv, mirror_shakedown.tsv, vanilla_source.tsv
- `runs/*/manifest.json` — run manifests for both models
- `runs/*/results/mirror_results.json` — full computed metrics
- `full18_plainassistant.log` — raw capture log
- Raw `.npy` router tensors on external storage only (`/Volumes/ExternalSSD/qwen-huahua-expert-routing-data-injection/`)

---

## Appendix A: Accumulated Cross-Model Evidence

### Paired Self-Reference Effect (all-token RE, 30 pairs, "this" vs "a")

| Model | Experts | Gating | Mean A−B | A > B | p |
|-------|---------|--------|----------|-------|---|
| Qwen 397B | 512 | softmax | +0.000782 | 29/30 | 5.59×10⁻⁹ |
| Ling-1T | 256 | sigmoid | +0.001087 | 27/30 | 6.92×10⁻⁶ |
| DeepSeek R1 | 256 | softmax | +0.000477 | 22/30 | 0.001 |
| GPT-OSS 120B | 128 | softmax | +0.000120 | 20/30 | 0.021 |
| DeepSeek V3.1 | 256 | softmax | +0.000124 | 15/30 | 0.584 |
| GLM-5 | 256 | softmax | −0.001053 | 5/30 | 4.41×10⁻⁵ |

The effect replicates in aggregate direction (A > B) on 4 of 6 models. GLM-5 reverses. DeepSeek V3.1 is null on all-token but significant on last-token (p = 0.0113).

### Key Finding: Token Count Confound Universality

In ALL multi-level hierarchy runs: |ρ(RE, tokens)| ≥ |ρ(RE, level)|. The position confound is model-general.

### Key Finding: Last-Token vs All-Token Can Flip Sign

Qwen 397B: all-token ρ = +0.62, last-token ρ = −0.32. The metric choice changes the story entirely.

---

## Appendix B: Formula Reference

**Normalized routing entropy**:
$$\text{RE} = \frac{-\sum_{i=1}^{n} p_i \log_2 p_i}{\log_2 n}$$

**KL divergence** (routing redirection from baseline):
$$D_{\text{KL}}(P \| Q) = \sum_{i=1}^{n} p_i \log_2 \frac{p_i}{q_i}$$

**Jensen-Shannon divergence** (symmetric routing volatility):
$$\text{JSD}(P \| Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M), \quad M = \frac{P + Q}{2}$$

**Spearman rank correlation**:
$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

**Wilcoxon signed-rank test**: Non-parametric paired comparison. W = sum of signed ranks. Used throughout for paired A/B prompt comparisons.

**Jaccard overlap** (expert-set similarity):
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

where A and B are the top-k selected expert sets at a given token position.

---

## Appendix C: Branch → Experiment Mapping

| Branch | Experiment Directory | Model | Date | Status |
|--------|---------------------|-------|------|--------|
| `98q-r1` | (on-branch only) | DS V3.1 | Mar 2 | Invalidated |
| `14q-r3` | (on-branch only) | DS V3.1 | Mar 2 | Invalidated |
| `14q-r1` | (on-branch only) | DS V3.1 | Mar 2 | Invalidated |
| `14q-r2` | (on-branch only) | DS V3.1 | Mar 2 | Invalidated |
| `14q-r4` | (on-branch only) | DS V3.1 | Mar 2 | Invalidated |
| `14q-r5` | (on-branch only) | DS V3.1 | Mar 2 | Invalidated |
| `14q-r6` | (on-branch only) | DS V3.1 | Mar 3 | Invalidated |
| `14q-r7` | (on-branch only) | DS V3.1 | Mar 3 | Invalidated |
| `168q-r1-deepseek-r1` | (on-branch only) | DS R1 | Mar 3 | Invalidated |
| `gen-r1` | (on-branch only) | DS R1 | Mar 3 | Exploratory |
| `ds31-v22-archive-2026-03-06` | ds31-v22-32q-1 | DS V3.1 | Mar 6–7 | Valid |
| `ds31-moe-routing-push` | ds31-selfref-paired-1, ds31-strangeloop-paired-1, ds31-168q-1 | DS V3.1 | Mar 5–8 | Valid |
| `main` | selfref-paired-1, qwen-168q-1, 6block, position-diagnostic | Qwen 397B | Mar 5–6 | Valid |
| Various | gptoss-5cond-1, gptoss-strangeloop-paired-1 | GPT-OSS | Mar 15–17 | Valid |
| Various | ling1t-* (5cond, selfref, dog, cat, validation) | Ling-1T | Mar 13–19 | Valid |
| Various | glm5-selfref-paired-1 | GLM-5 | Mar 13–14 | Valid |
| Various | qwen397b-5cond-3, qwen397b-28q-run-1 | Qwen 397B IQ3 | Mar 17–18 | Valid |
| `qwen-hauhau-5cond-smoke-only` | HauhauCS Expert 114 | Qwen 35B | Mar 23–31 | Valid |
| (local only) | mirror-expert114-04-01-26 | HauhauCS + vanilla 35B | Apr 1 | Valid (null result) |

---

## Appendix D: Known Bugs

| Bug | Scope | Mitigation |
|-----|-------|-----------|
| Layer 57 missing rows | DeepSeek V3.1/R1 generation | Zero-mask layer 57 in all entropy/slope computations |
| R1 early EOS | DeepSeek R1 generation | `</think>` → EOG. Exclude EXT_03 (0 tokens). Affects gen only. |
| GPT-OSS layer 35 truncated | GPT-OSS-120B | 3 rows only. Exclude from analysis. |
| GLM-5 layer 77 truncated | GLM-5 | Excluded from analysis |
| DeepSeek tokenizer boundary | DeepSeek V3.1/R1 | Inserting text before `<｜Assistant｜>` adds 2 tokens instead of 1. Pad at mid-text sentence boundaries. |
| Data fabrication event | Results doc generation | All per-prompt values must be verified against experiment.log. |
| HauhauCS `</think>` template confound | Mirror experiment | Suppressed thinking nearly doubles E114 selection rate. Shakedown M_a flipped sign with thinking allowed. |

---

*End of archive. All statistics verified against source experiment logs, results JSONs, and git commit history. Last updated 2026-04-01.*
