# Partial Results

This file explains which parts of the archive are fully supported by preserved artifacts and which parts can only be partially reconstructed.

This matters because the archive mixes:

- copied historical result files
- recalculated summaries
- a smaller number of surviving raw router captures

Some run names in this archive are historical internal labels. In this file, the important distinction is simpler:

- some evidence bundles can still be checked end to end from preserved files inside the archive
- some evidence bundles can only be partially recovered from raw data that survived elsewhere

## What Is Complete

Two parts of the archive are well preserved:

- [`raw/168q-r1-deepseek-r1/`](./raw/168q-r1-deepseek-r1/)
  - the original DeepSeek R1 168-prompt hierarchy writeup
  - the original per-prompt JSON result file behind that writeup
- [`raw/r1-28q-1/`](./raw/r1-28q-1/)
  - saved generation-phase metric surfaces
  - matched-length reanalysis artifacts
  - enough preserved arrays to recompute the stored RE and slope summaries

If you want the main confound story, these complete bundles are the more important materials. The partial recovery described below is supplemental.

## What Is Only Partial

### Partial recovery of a separate v2.2 prompt suite

Raw router captures were found outside the original archive at:

- `/Volumes/ExternalSSD/llama-eeg-tests/ds31-v22-32q-1/output`

This folder belongs to a different DeepSeek V3.1 prompt suite than the main 12-level hierarchy analyzed elsewhere in the archive. It is a forced-choice operational triage suite with tightly templated prompts and JSON-only answers.

The surviving raw files support a partial recomputation of prefill routing entropy using the DeepSeek sigmoid-gated routing reconstruction.

Recovered outputs:

- [`supplemental/recovery/ds31_v22_partial_raw_recovery.md`](./supplemental/recovery/ds31_v22_partial_raw_recovery.md)
- [`supplemental/recovery/ds31_v22_partial_raw_recovery.json`](./supplemental/recovery/ds31_v22_partial_raw_recovery.json)
- [`supplemental/recovery/recompute_ds31_v22_partial.py`](./supplemental/recovery/recompute_ds31_v22_partial.py)

Coverage:

- expected prompts: `32`
- complete raw prompts recovered: `17`
- broken prompt directories: `1` (`L3_A2`)
- missing prompt directories: `14`

Recovered prompt IDs:

- `L1_A1, L1_A2, L1_B1, L1_B2, L1_C1, L1_C2, L1_D1, L1_D2`
- `L2_A1, L2_A2, L2_B1, L2_B2, L2_C1, L2_C2, L2_D1, L2_D2`
- `L3_A1`

Important scope limit:

- this recovery is prefill-only
- these raw captures have `n_tokens_generated=0`
- this does not reconstruct the generation-side commitment-token analysis from that separate v2.2 project

## Best Available RE Numbers for the Partial v2.2 Recovery

From the partial raw recomputation:

| Level | n | Mean all-token RE | Mean last-token RE | Mean tokens |
|-------|---|-------------------|--------------------|-------------|
| L1 | 8 | 0.961820 | 0.943339 | 227.00 |
| L2 | 8 | 0.961210 | 0.944267 | 230.00 |
| L3 | 1 | 0.961707 | 0.943729 | 245.00 |

Subset-only correlations:

- all-token RE vs level: `rho=-0.5789612716571944`, `p=0.014883237910398958`
- last-token RE vs level: `rho=0.5817116814987962`, `p=0.014303952465020758`
- all-token RE vs prompt tokens: `rho=-0.6572042169844519`, `p=0.004148429286056417`
- last-token RE vs prompt tokens: `rho=0.5901172080673245`, `p=0.012643058728660776`

These should not be treated as the result for the full v2.2 run. They are statistics on the recoverable subset only.

## Interpretation After Reviewing the Recovered Prompts

The recovered prompts here are not open-ended reasoning prompts like the main hierarchy suite. They are tightly templated operational triage prompts that ask for JSON output under explicit rule sets.

What the recovered subset actually contains:

- `L1` is mostly direct mapping: read a case record and apply a simple threshold or branch.
- `L2` adds rule interaction: two-part classification logic with explicit ignore-fields.
- `L3` adds precedence and exception handling: compute a base class, then apply a downgrade or action override.

Why that matters for interpretation:

- The prompt family is intentionally uniform. Most of the token mass is shared scaffold, not freely varying semantics.
- The surviving subset spans only `L1` to `L3`, and almost all prompts sit in a narrow 227-245 token band.
- That makes this recovery useful as a check that raw sigmoid-gated RE can still be reconstructed from surviving tensors, but weak as evidence about any broad "complexity hierarchy."

The cautious reading is:

- this recovery supports the general warning that RE metrics remain sensitive to prompt construction details even in a different suite
- it does not independently establish a new hierarchy claim
- it is best treated as an archival salvage result, not a core finding

## Practical Interpretation

- The archive contains complete material for the original DeepSeek hierarchy claim, the R1 generation follow-up, and the later cross-model confound analysis.
- The v2.2 partial recovery is preserved as supplemental because it is adjacent work, not the main evidence for the token-confound story.
- Raw RE for that v2.2 suite is recoverable only for the subset that still exists on disk, not for the full 32-prompt run.
