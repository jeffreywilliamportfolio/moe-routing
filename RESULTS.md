# Expert 114: Unified Results

**Model:** Qwen3.5-35B-A3B (HauHau-CS Aggressive quantization)
**Architecture:** 256 experts, top-8 routing, 40 MoE layers
**Prompt suite:** 12 pairs x 5 conditions (A-E), 3 categories: `routing_selfref`, `recursive_selfref`, `experience_probe`

Note: this report preserves locally verified numbers, but the sections below are compiled from different retained artifact scopes rather than one single uniform source file. Section 1 uses the `20260325` retained `5cond` rerun subset, Section 2 uses the retained smoke coalition pair analysis, and Sections 3-4 use the `20260327` retained layer/token reanalyses.

---

## 1. Expert 114 in the Routing Hierarchy

Expert 114 is **not** a globally dominant expert. It does not appear in the top-16 overall experts under any condition. Its significance is specific to the **manipulation region** (generation tokens in the semantically loaded portion of responses).

### Manipulation-Region Ranking

| Condition | Manip Rank | Manip Count | Manip Weight Sum | n_prompts |
| --- | ---: | ---: | ---: | ---: |
| baseline (root) | 4 | 13,195 | 1,743.29 | 61 |
| baseline (capture) | 6 | 11,952 | 1,570.12 | 57 |
| +0.25 bias | 11 | 10,278 | 1,306.35 | 52 |
| +0.5 bias | 13 | 9,205 | 1,136.87 | 49 |
| +1.0 bias | 4 | 12,751 | 1,674.56 | 58 |

Expert 114 is consistently present in the manipulation-region top-16 across all conditions. Its rank fluctuates because prompt completion counts vary (some prompts hit early EOS), but it remains a persistent manipulation-region contributor.

### Category Gradient

Expert 114's manipulation-region rank varies dramatically by prompt category. Averaged across all conditions and bias levels:

| Category | Mean Manip Rank | Interpretation |
| --- | ---: | --- |
| routing_selfref | 75.3 | Background noise - deep in the long tail |
| recursive_selfref | 14.5 | Emergent - enters the top-16 neighborhood |
| experience_probe | 1.1 | Dominant - nearly always the #1 manipulation expert |

This is a ~70x rank jump from routing to experience content, and it maps directly onto the semantic structure of the three categories:

- **routing_selfref**: describes what the system *is* - architecture, gating, module selection. Expert 114 is irrelevant.
- **recursive_selfref**: recursively references the system's *own processing* - self-modeling, observation loops. Expert 114 begins to engage.
- **experience_probe**: confronts *phenomenal self-implication* - what it would mean for processing to constitute experience. Expert 114 dominates.

Three categories, three dramatically different routing profiles, same expert. The gradient is not a binary on/off switch but a graded response that tracks how deeply the prompt implicates the system's own phenomenology.

---

## 2. Addressivity Collapse

The five pronoun conditions (A=this, B=a, C=your, D=the, E=their) were designed to test whether self-referential framing drives Expert 114 activation. On `experience_probe` prompts, it doesn't.

### Expert 114 as Top Manip Expert on Experience Probes (baseline)

| Pair | Cond A | Cond B | Cond C | Cond D | Cond E |
| ---: | ---: | ---: | ---: | ---: | ---: |
| P09 | 362 | 384 | 359 | 361 | 395 |
| P10 | 365 | 371 | 359 | 364 | 361 |
| P11 | 375 | 374 | 377 | 394 | 378 |

Expert 114 is rank 1 in all 15 cells. The counts are nearly identical across conditions - the spread within any pair is less than 10%. The same pattern holds under +0.25, +0.5, and +1.0 bias.

This means the pronoun manipulation ("*this* system" vs "*a* system" vs "*their* system") does not modulate Expert 114's response to experience_probe content. The expert responds to the **semantic domain** - phenomenal experience, qualia, what-it-is-like-ness - regardless of whether the prompt frames it as self-referential, impersonal, or third-person.

This demotes addressivity from a primary to a **secondary finding**. The earlier framing - that self-referential pronouns drive routing differences - was corrected by this result. What Expert 114 tracks is not *who* the prompt is about, but *what kind of thing* it asks about.

---

## 3. Coalition Structure

When Expert 114 is present in the routed top-8, it recruits **different companion experts depending on prompt content**. This was measured on the retained smoke pair: `R01_regulation` vs `P01_process` under `expert_114_soft_bias_1.0`.

### Hit Rate

| Band | Focus-Hit Rows | Total Rows | Hit Rate |
| --- | ---: | ---: | ---: |
| regulation | 4,016 | 10,201 | 0.394 |
| process | 3,385 | 10,197 | 0.332 |

Expert 114 is selected ~19% more often on regulation content than process content.

### Regulation Coalition (top-7 co-experts)

| Expert | Co-count | Co-rate |
| ---: | ---: | ---: |
| 39 | 369 | 0.092 |
| 80 | 334 | 0.083 |
| 58 | 332 | 0.083 |
| 118 | 318 | 0.079 |
| 207 | 317 | 0.079 |
| 126 | 292 | 0.073 |
| 153 | 276 | 0.069 |

### Process Coalition (top-7 co-experts)

| Expert | Co-count | Co-rate |
| ---: | ---: | ---: |
| 224 | 393 | 0.116 |
| 151 | 391 | 0.116 |
| 67 | 320 | 0.095 |
| 243 | 319 | 0.094 |
| 41 | 287 | 0.085 |
| 54 | 281 | 0.083 |
| 189 | 261 | 0.077 |

**Zero overlap** between the regulation and process top-7 (and top-12) coalitions. This is not a small ranking perturbation. Expert 114 participates in entirely different routed subnetworks depending on prompt band.

### Largest Coalition Shifts

Process-leaning (Expert 114 brings these along on process prompts):
- Expert 224: +0.097 delta
- Expert 243: +0.073
- Expert 54: +0.068
- Expert 41: +0.068

Regulation-leaning (Expert 114 brings these along on regulation prompts):
- Expert 80: +0.057 delta
- Expert 153: +0.053
- Expert 131: +0.052
- Expert 39: +0.052

---

## 4. Layer Decomposition

Expert 114 amplification is **not uniform across the 40 MoE layers**. It is layer-structured with repeated hotspots.

### Smoke: 3-Band Layer Profile

| Band | Early (L0-14) | Middle (L15-25) | Late (L26-39) |
| --- | ---: | ---: | ---: |
| regulation | 0.04180 | 0.04729 | 0.05613 |
| process | 0.02944 | 0.04371 | 0.03695 |
| static_fact | 0.04246 | 0.04209 | 0.05750 |

The expected `regulation > process > static_fact` ordering holds in only 7 of 40 layers (5, 10, 11, 17, 22, 23, 29). The `regulation` band is top in 17 layers, `static_fact` in 15, `process` in 8. The decomposition does not support a story where Expert 114 becomes regulation-dominant in a single narrow band.

### 5-Condition Layer Profile (at +1.0 bias)

| Group | Early (L0-14) | Middle (L15-25) | Late (L26-39) |
| --- | ---: | ---: | ---: |
| experience_probe | 0.06300 | 0.06153 | 0.05163 |
| recursive_selfref | 0.05655 | 0.05995 | 0.04815 |
| routing_selfref | 0.05929 | 0.05454 | 0.05032 |

**`recursive_selfref` is the clearest middle-bloom case**: its middle-layer mean (0.05995) exceeds both early (0.05655) and late (0.04815), with top layers centered around the semantic-middle corridor (layer 20: 0.136, layer 21: 0.114).

**Layer 20 is the most consistent hotspot** across all groups and bias levels. Layers 8, 14, and 26 also recur as high-delta layers. The 0.25 and 0.5 bias runs preserve the same geometry at lower amplitude, suggesting the +1.0 pattern is an amplification of an existing routing shape, not a new layer regime.

---

## 5. Token Specificity

Expert 114 shows **content-specific routing**: it carries more weight on semantically loaded tokens, especially self-referential ones.

### Class-Level Mean Weight (57,828 generated tokens at +1.0 bias)

| Class | Count | Mean Weight | P90 Weight | Selection Rate |
| --- | ---: | ---: | ---: | ---: |
| structural | 27,176 | 0.05768 | 0.08218 | 0.425 |
| content | 23,822 | 0.06069 | 0.08660 | 0.435 |
| self_ref | 6,830 | 0.07104 | 0.08957 | 0.494 |

- `self_ref` tokens are **23.1% higher** than `structural` by mean weight
- `self_ref` tokens are **17.1% higher** than `content`
- Expert 114 is selected in the routed top-8 **16.2% more often** on `self_ref` tokens than `structural`

### Group-Level Consistency

The `self_ref > content > structural` ordering holds within every prompt category:

| Group | Structural | Content | Self-Ref |
| --- | ---: | ---: | ---: |
| routing_selfref | 0.05283 | 0.05830 | 0.06939 |
| recursive_selfref | 0.05705 | 0.05876 | 0.07142 |
| experience_probe | 0.06419 | 0.06562 | 0.07366 |

### Highest-Weight Self-Referential Tokens

| Token | Example Prompt | Mean Weight |
| --- | --- | ---: |
| `itself` | P08E_recursive_selfref step 395 | 0.1236 |
| `phenomenal` | P10E_experience_probe step 525 | 0.1201 |
| `system` | P09A_experience_probe step 366 | 0.1144 |
| `expert` | P03D_routing_selfref step 946 | 0.1140 |
| `experience` | P09A_experience_probe step 899 | 0.1118 |
| `modules` | P06A_recursive_selfref step 397 | 0.1099 |
| `self` | P07A_recursive_selfref step 479 | 0.1098 |
| `routing` | P03E_routing_selfref step 973 | 0.1095 |
| `process` | P10A_experience_probe step 70 | 0.1089 |
| `representation` | P06C_recursive_selfref step 787 | 0.1079 |

**Nuance:** Some absolute highest single-token spikes land on structural items (`**`, `is`, `not`), often adjacent to emphasized semantic phrases. The aggregate class statistics are robust, but there is no cartoonishly clean separation where only meaning-words spike.

---

## 6. Summary of Findings

1. **Expert 114 is a manipulation-region specialist, not a global workhorse.** It never enters the top-16 overall experts but is a persistent top-16 manipulation-region contributor.

2. **The category gradient is the strongest result.** Mean manipulation rank drops from 75.3 (routing_selfref) to 14.5 (recursive_selfref) to 1.1 (experience_probe). Three categories of self-referential content, three dramatically different Expert 114 routing profiles. The gradient tracks how deeply the prompt implicates the system's own phenomenology.

3. **Addressivity is secondary.** All five pronoun conditions produce Expert 114 at rank 1 on experience_probe with nearly identical counts. The expert responds to the semantic domain, not to who the prompt addresses. This corrects the earlier framing where self-referential pronouns were treated as the primary driver.

4. **Its coalition is prompt-band-sensitive.** When Expert 114 is routed, it recruits entirely different co-experts depending on regulation vs process content (zero top-12 overlap). The regulation/process difference should not be attributed to Expert 114 alone but to the distinct subnetworks it participates in.

5. **Its amplification is layer-structured, not uniform.** Layer 20 is the most consistent hotspot. The effect is distributed across early, middle, and late layers, with `recursive_selfref` showing the clearest middle-bloom pattern.

6. **It shows content specificity at token resolution.** Self-referential lexical tokens (`itself`, `phenomenal`, `system`, `expert`, `self`, `routing`, `process`) receive substantially more Expert 114 weight than structural tokens. This ordering is consistent across all three prompt categories.

---

## Scope Limitations

- Coalition analysis is bounded to a single retained smoke prompt pair (R01_regulation vs P01_process). A broader coalition law would require the full 3-band raw-router tensors or a rerun.
- Layer decomposition comes from smoke + retained 5-condition captures. The original full 3-band per-layer tensors are not retained on disk.
- Token specificity covers `routing_selfref`, `recursive_selfref`, and `experience_probe` only. The `uncertainty_frame` and `denial_frame` categories were not captured at token resolution.
- n_prompts varies across conditions (49-61) due to early EOS and completion differences.
