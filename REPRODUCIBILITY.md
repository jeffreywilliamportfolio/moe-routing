# Reproducibility Status

Reviewed against the checked-in artifacts in this repo on 2026-03-18.

## Rubric

### `Reproducible`

A folder can be marked `Reproducible` only if the checked-in bundle is sufficient for a third party to rerun or re-derive the reported result with minimal guesswork.

Minimum standard:

- exact model and quant are documented
- model-specific prompt wrapper or chat template is preserved
- prompt suite and generated TSV are preserved
- run script and analysis script are preserved
- checked-in results file is present
- run log is present
- artifact completeness checks are documented or encoded well enough that an incomplete rerun is detectable
- the checked-in artifact set does not obviously depend on missing external raw-data paths for the reported exact metrics

### `Auditable but not reproducible`

Use this when the methodology and conclusions can be reviewed credibly from the checked-in files, but a third party still cannot rerun or fully re-derive the exact reported results from the folder alone.

Typical reasons:

- exact build provenance is incomplete
- raw capture artifacts are missing or only partially preserved
- the report depends on external paths or bundles not included locally
- logs or manifests are incomplete enough that rerun fidelity still requires inference

### `Not reproducible`

Use this when the checked-in folder is not a self-contained rerun package and also has material completeness gaps in the preserved result bundle itself.

Typical reasons:

- partial or missing `output/` tree
- missing per-prompt artifacts needed to support the checked-in result
- rerun path is known to permit incomplete outputs without hard failure
- the current repo snapshot is not sufficient to validate that the preserved result file came from a complete run

## Current Status

| Folder | Status | Notes |
|---|---|---|
| `qwen/qwen397b-selfref-5cond-q8_0-run1` | `Reproducible` | Strongest checked-in bundle: model/quant documented, wrapper preserved, prompt suite + TSV preserved, run and analysis scripts present, result JSON present, `experiment.log` present, and `reproducibility_manifest.json` checksums core files. |
| `qwen/qwen397b-strangeloop-5cond-ud_iq3_xxs-run1` | `Reproducible` | Same general standard as the Qwen self-ref run: prompt/TSV/scripts/results/log preserved and checksum manifest present. |
| `deepseek/ds31-5cond-1` | `Auditable but not reproducible` | Methodology is documented well and logs/results are present, but exact build provenance is not preserved as cleanly as it should be for a no-guess rerun package. |
| `ling1t-test-run` | `Auditable but not reproducible` | Documentation and provenance improved substantially, including build commit/checksum and tensor preflight evidence, but the folder is still not self-contained for the exact reported metrics: no local full `output/` tree, no checked-in `experiment.log`, no local prompt-to-artifact manifest, and the report still references external source bundles for exact numbers. |
| `gptoss/gptoss-5cond-1` | `Not reproducible` | The current repo snapshot includes only a partial checked-in `output/` tree rather than a clearly complete 150-prompt preserved bundle; at review time it contained 68 prompt directories, with at least one preserved prompt missing `metadata.txt`. |
| `gptoss/gptoss-strangeloop-paired-1` | `Not reproducible` | Result bundle and scripts are useful for review, but the folder still does not meet the same self-contained rerun standard as the Qwen bundles, and the rerun path still requires manual completeness checking. |

## Notes

- These labels are about the current checked-in repo snapshot, not about whether the original experiment was run correctly.
- A folder can move upward if additional raw artifacts, logs, manifests, or build provenance are checked in later.
- `ling1t-test-run/RESULTS.md` already contains more specific Ling-only reproduction notes and a repair checklist for making that folder fully self-contained.
