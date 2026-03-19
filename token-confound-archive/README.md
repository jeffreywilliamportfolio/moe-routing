# Token Confound Archive

This archive documents one specific failure mode: an apparent MoE routing-entropy hierarchy that later collapsed under token-position controls.

The shortest honest summary is:

- The original DeepSeek hierarchy claims were invalid.
- The invalidation was not random noise. It exposed a real token-position confound in MoE routing entropy.
- The core evidence is preserved here in both forms: the original mistaken results and the later analysis explaining why they were mistaken.

## Historical Warning

Some files in `raw/` are preserved exactly because they show what was believed at the time.

- They are part of the evidence trail.
- They are not the archive's final interpretation.
- If a raw writeup conflicts with `NARRATIVE.md` or `CROSS_MODEL_POSITION_CONFOUND.md`, the archive-level documents are the corrected interpretation.

## Start Here

1. `NARRATIVE.md` -- short chronology of what was claimed, what broke, and what survived
2. `CROSS_MODEL_POSITION_CONFOUND.md` -- final evidence file showing the hierarchy disappears under last-token controls
3. `raw/168q-r1-deepseek-r1/168q-r1_RESULTS.md` -- original R1 headline result as it was written
4. `raw/r1-28q-1/r1-28q-1_RESULTS.md` -- generation follow-up that exposed a token-count confound
5. `MANIFEST.md` -- full file inventory

If you want the recovery-status view first, read `PARTIAL-RESULTS.md`. It lists which runs are fully supported in the archive and which are only partially recoverable from surviving raw artifacts.

## Naming Decoder

The `14q-r*` branch names are historical labels, not level numbers and not chronological order.

| Branch | Added level(s) | Content theme |
|--------|----------------|---------------|
| `14q-r3` | L8 | Strange loops |
| `14q-r1` | L9 | Deep self-reference |
| `14q-r2` | L10 | Nexus-7 third-person |
| `14q-r4` | L11 | Architectural introspection |
| `14q-r5` | L12 | Echo persona |
| `14q-r6` | L10 control | Bob name control |
| `14q-r7` | L10 control | Aether name control |

This is why filenames like `14q-r1_prefill.json` do not line up with level order.

## Raw vs Recalculated

- `raw/` contains original copied experiment artifacts and result writeups.
- `data/` contains recalculated summaries derived from those originals.

Example:

- `raw/168q-r1-deepseek-r1/results_168_r1_prefill.json` is the original per-prompt R1 results file.
- `data/168q-r1_R1_prefill.json` is a recalculated summary derived from that source so it can be compared consistently with the rest of the archive.

The recalculated files are easier to scan. The raw files are the source of record.

## Scope

Core archive:
- DeepSeek V3.1 hierarchy buildup
- DeepSeek R1 hierarchy replication
- R1 generation follow-up
- Qwen position-control comparison

Supplemental archive:
- DeepSeek v2.2 forced-choice and multiseed runs
- post-confound methodology notes

Those supplemental files are preserved because they were adjacent in time, but they are not load-bearing for the token-confound claim.
