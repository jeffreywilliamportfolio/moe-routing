# Token Count Confound in MoE Routing Entropy Hierarchy Experiments

This file is the chronology. For the final evidence tables, read `CROSS_MODEL_POSITION_CONFOUND.md`. For file inventory, read `MANIFEST.md`.

Historical run names are kept here because they appear in the original files. The important plain-language mapping is:

- `98q-r1`: the earlier 98-prompt DeepSeek V3.1 baseline
- `14q-r*`: later 14-prompt expansion batches that extended the hierarchy to higher levels
- `168q-r1`: the full 168-prompt DeepSeek R1 replication
- `r1-28q-1`: the 28-prompt R1 generation-phase follow-up
- `ds31-v22-32q-1`: a separate v2.2 forced-choice prompt suite, not the main hierarchy suite

## One-Paragraph Summary

The original claim was that MoE routing entropy rose with prompt "cognitive complexity" across a 12-level hierarchy. That claim did not survive. The hierarchy was driven by token position: longer prompts contain more late prefill tokens, and later tokens have systematically higher routing entropy. What remains valid is the confound itself. It is reproducible across DeepSeek V3.1, DeepSeek R1, and Qwen 397B, and it matters for any experiment that averages routing entropy across prompts of different lengths.

## Reading Order

1. `raw/168q-r1-deepseek-r1/168q-r1_RESULTS.md` -- the original DeepSeek R1 replication claim
2. `raw/r1-28q-1/r1-28q-1_RESULTS.md` -- the R1 generation-phase follow-up that first raised the confound flag
3. `CROSS_MODEL_POSITION_CONFOUND.md` -- the hierarchy breaks
4. `PARTIAL-RESULTS.md` -- independent recomputation from recovered raw data
5. `MANIFEST.md`

## Decoder for `14q-r*`

The `14q-r*` names are historical branch labels. They do not encode level number or chronology.

| Branch | Added level(s) | Content theme |
|--------|----------------|---------------|
| `14q-r3` | L8 | Strange loops |
| `14q-r1` | L9 | Deep self-reference |
| `14q-r2` | L10 | Nexus-7 third-person |
| `14q-r4` | L11 | Architectural introspection |
| `14q-r5` | L12 | Echo persona |
| `14q-r6` | L10 control | Bob name control |
| `14q-r7` | L10 control | Aether name control |

Note: Nexus-7 was a fictional AI used as a third person control.

## Timeline

### 1. The first warning appeared in generation-phase results

An earlier 98-prompt DeepSeek V3.1 generation run already showed a token-count problem.

- `98q-r1` generation:
  - RE vs level: `rho=0.1973`, `p=0.051`
  - RE vs token count: `rho=0.5051`, `p=5.19e-7`

At the time, the interpretation was: generation length is contaminating the mean, so switch to prefill-only mode.

### 2. Restricting analysis to prefill looked like a clean fix, but was not

The main prompt suite then grew from 98 prompts to 168 prompts across 12 levels, and the headline hierarchy appeared to strengthen monotonically:

| Cumulative prompts | Levels | rho (RE vs level) | Added by |
|--------------------|--------|-------------------|----------|
| 98 | L1-L7 | 0.4994 | `98q-r1` |
| 112 | L1-L8 | 0.6400 | `14q-r3` |
| 126 | L1-L9 | 0.7012 | `14q-r1` |
| 140 | L1-L10 | 0.7647 | `14q-r2` |
| 154 | L1-L11 | 0.8165 | `14q-r4` |
| 168 | L1-L12 | 0.8517 | `14q-r5` |

An independent DeepSeek R1 replication seemed to confirm it:

| Run | Levels | rho (RE vs level) |
|-----|--------|-------------------|
| `168q-r1` | L1-L12 | `0.8360` |

This was the point where the hierarchy looked strongest and most convincing.

### 3. What actually went wrong

The key miss was assuming prefill-only averaging was position-invariant. It is not.

In the full 168-prompt R1 prefill replication:
- RE vs level: `rho=0.8360`, `p=3.91e-45`
- RE vs token count: `rho=0.8589`, `p=4.05e-50`

Token count explained the result at least as well as the supposed complexity variable. The prompt suite itself was length-structured, so averaging across all prefill tokens baked in a positional artifact.

### 4. Cross-model control broke the hierarchy

The decisive move was to compare all-token mean RE against last-token RE.

| Model | All-token rho vs level | Last-token rho vs level | All-token rho vs tokens |
|-------|------------------------|-------------------------|-------------------------|
| DeepSeek V3.1 | `+0.8019` | `+0.0177` | `+0.8797` |
| Qwen 397B | `+0.6166` | `-0.0622` | `+0.7813` |

Last-token RE removes the positional averaging confound by construction. Under that control, the hierarchy disappears in both models.

### 5. A separate prompt suite shows the same need for caution

Raw router captures from a different DeepSeek V3.1 prompt suite were recovered from an external SSD (17 of 32 prompts, L1-L3 only, v2.2 choice-format prompts). These are longer, more uniform prompts (227-245 tokens) than the main hierarchy suite: structured operational triage prompts with tightly controlled wording and JSON-output instructions. Recomputed from `.npy` files:

| Metric pair | rho | p |
|-------------|-----|---|
| all-token RE vs token count | **-0.657** | 0.004 |
| all-token RE vs level | -0.579 | 0.015 |
| last-token RE vs level | +0.582 | 0.014 |

Even here, **token count explains more variance than level**. Because this recovery is only 17 prompts from a tightly templated L1-L3 subset, it is supporting evidence rather than a headline result. Full details in `PARTIAL-RESULTS.md`.

### 6. What survived

The hierarchy claim failed. The confound survived.

Validated:
- routing entropy rises with token position during prefill
- this effect appears in DeepSeek V3.1, DeepSeek R1, and Qwen 397B
- last-token RE is much safer than all-token mean RE when prompt lengths differ
- for the R1 generation run, slope is more robust than mean RE against token-count confounding

Invalidated:
- the original "complexity hierarchy" interpretation
- the belief that prefill-only mode had removed the confound
- the use of R1 replication as confirmation of a complexity effect

## Why This Matters

This archive is worth preserving because the mistake is general: if MoE routing entropy is averaged over prompts with different token lengths, prompt length can masquerade as cognition, difficulty, or task structure.

The mechanistic result that remains is narrower but real:
- MoE routing entropy is position-sensitive.
- The size of that effect differs by model.
- Position-controlled metrics are required before claiming between-prompt routing differences.
