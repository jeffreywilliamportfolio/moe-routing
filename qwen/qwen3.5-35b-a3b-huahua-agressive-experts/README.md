# Qwen3.5-35B HauhauCS Aggressive Experts

This is the researcher-facing entrypoint for the HauhauCS aggressive Expert 114 intervention work.

## Scope

- `METHOD/`: execution, routing, and analysis scripts for the HauhauCS aggressive experiment path.
- `PROMPTS/`: prompt suites, generated TSVs, rubric markers, and sham-control metadata.
- `DOCS/`: reproduction notes, result summaries, and local reanalysis outputs.

See [MANIFEST.md](MANIFEST.md) for the minimal include/exclude summary.

## Git LFS

The raw reviewer-facing `.npy` files for the `5cond` and `smoke-test` runs are tracked in this branch with Git LFS under the original Hauhau experiment `runs/` paths. After cloning:

```bash
git lfs pull
```

## Reproducibility

- Yes: reproducible local reanalysis of the included `5cond` and `smoke-test` raw `.npy` files.
- No: a one-command end-to-end rerun from scratch.
- Missing for a true rerun: model artifact, exact capture binary/build, and portable runtime environment.

For the full reproducibility caveats, start with [DOCS/REPRODUCE.md](DOCS/REPRODUCE.md).

## Reading Order

1. [DOCS/PLAN.md](DOCS/PLAN.md)
2. [DOCS/RESULTS.md](DOCS/RESULTS.md)
3. [METHOD/run_experiment.py](METHOD/run_experiment.py)
4. [METHOD/analyze_5cond_condition.py](METHOD/analyze_5cond_condition.py)
5. [METHOD/analyze_generation.py](METHOD/analyze_generation.py)
6. [METHOD/qwen_router.py](METHOD/qwen_router.py)
7. [DOCS/20260325-raw-npy-rerun](DOCS/20260325-raw-npy-rerun)
