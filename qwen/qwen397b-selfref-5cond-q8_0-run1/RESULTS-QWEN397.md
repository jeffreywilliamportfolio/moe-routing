# RESULTS-QWEN397

## Summary

This experiment measures routing behavior in `Qwen3.5-397B-A17B` across `30` paired prompt sets with `5` controlled wording conditions:

- `A`: `this system`
- `B`: `a system`
- `C`: `your system`
- `D`: `the system`
- `E`: `their system`

The run produced `150` analyzed prompts, `30` complete pairs, and `0` token-mismatch pairs after correction. Routing was reconstructed as `softmax(512) -> topk(10) -> renormalize`, entropy was normalized by `log2(10)`, and all reported pairwise tests used Holm-Bonferroni correction.

## Main Finding

Self-referential phrasing produces a distinct routing signature in this model.

The strongest separation appears in the `your system` condition (`C`). Across the full prompt body, `C` shows the lowest mean routing entropy, indicating a more concentrated expert-allocation pattern than the other phrasings. In the manipulation region, `C` also shows the highest KL divergence from the calibration baseline, indicating the largest routing shift under the self-referential wording.

## Condition Means

| Condition | Phrase | Mean all-token RE | Mean last-token RE | Mean KL-manip | Mean KL-cal2 |
|---|---|---:|---:|---:|---:|
| A | this system | 0.938412310 | 0.955353712 | 0.657441645 | 0.683993353 |
| B | a system | 0.937943224 | 0.955394809 | 0.641502769 | 0.681993607 |
| C | your system | 0.937567857 | 0.954987439 | 0.688179900 | 0.697404039 |
| D | the system | 0.938482144 | 0.955595301 | 0.634574799 | 0.678388399 |
| E | their system | 0.938258616 | 0.955804306 | 0.650319780 | 0.684178208 |

These condition means define a clear structure:

- All-token RE ordering: `D > A > E > B > C`
- Last-token RE ordering: `E > D > B > A > C`
- KL-manip ordering: `C > A > E > B > D`

## Strongest Pairwise Separation

The clearest contrast is between `your system` (`C`) and `the system` (`D`):

| Metric | C-D mean diff | Direction | Holm p |
|---|---:|---|---:|
| All-token RE | -0.000914286 | `D > C` | 5.587935e-08 |
| Last-token RE | -0.000607863 | `D > C` | 0.022222003 |
| KL-manip | 0.053605101 | `C > D` | 5.587935e-08 |

This shows that `your system` combines lower routing entropy with a larger calibration-to-manipulation routing shift than `the system`.

## Significant Effects

### All-token Routing Entropy

Holm-corrected effects show broad separation across wording conditions:

- `A > B`
- `A > C`
- `A > E`
- `B > C`
- `D > B`
- `E > B`
- `D > C`
- `E > C`
- `D > E`

### Last-token Routing Entropy

The last token carries a weaker but still structured effect:

- `E > A`
- `E > B`
- `D > C`
- `E > C`

### KL to Calibration Baseline

KL separation is strong and highly organized:

- `A > B`
- `C > A`
- `A > D`
- `A > E`
- `C > B`
- `B > D`
- `E > B`
- `C > D`
- `C > E`
- `E > D`

## Interpretation

The routing response is not flat across wording variants. The model organizes these five phrasings into a consistent internal pattern, and the self-referential phrase `your system` occupies a distinctive position in that structure.

Two properties stand out:

- `your system` produces the largest shift away from calibration during the manipulation region.
- `your system` also produces the lowest mean routing entropy, indicating more concentrated expert selection.

Together, these results show a stable self-referential routing effect in `Qwen3.5-397B-A17B` under the corrected five-condition design.
