# qwen-hauhau-5cond-smoke-only Scripts

This folder collects the local analysis scripts corresponding to the retained
publication branch `qwen-hauhau-5cond-smoke-only`.

The GitHub branch retains run outputs and manifests, but not the Python source
files that produced those calculations. These files were recovered from the
matching local Hauhau tool bundle.

## File Map

- `analyze_5cond_condition.py`
  Calculates the per-condition 5-condition outputs such as
  `analysis-baseline.json`, `analysis-expert_114_soft_bias_0.25.json`,
  `analysis-expert_114_soft_bias_0.5.json`, and
  `analysis-expert_114_soft_bias_1.0.json`.

- `analyze_generation.py`
  Calculates the smoke-run `analysis.json` and compares intervention conditions
  against baseline using rubric, routing, and co-occurrence metrics.

- `qwen_router.py`
  Shared routing reconstruction math used by both analyzers:
  `softmax -> top-8 select -> renormalize`, normalized entropy, and JSD helpers.

- `mine_sham_controls.py`
  Selects sham experts used by the steering experiment setup.

- `run_experiment.py`
  Orchestrates capture plus post-run analysis. It calls
  `mine_sham_controls.py` when needed and imports `analyze_generation.py` for
  smoke analysis output generation.

## Notes

- `analyze_5cond_condition.py` is the relevant calculator for the retained
  5-condition branch artifacts.
- `analyze_generation.py` is the relevant calculator for the retained smoke
  branch artifacts.
- `qwen_router.py` is required by both and should be treated as part of the
  calculation path, not just a utility.
