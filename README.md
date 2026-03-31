# HauhauCS 5cond + Smoke Only

This branch is a narrow publication branch for the HauhauCS aggressive run family. It contains only the reviewer-facing `5cond` and `smoke-test` artifacts.

## Included

- `experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/nothink-5cond-boost-1024-20260323/`
- `experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/smoke-20260323b/`

## Git LFS

The raw router `.npy` files are tracked with Git LFS. After cloning:

```bash
git lfs pull
```

## Main Results

5cond result summaries:

- [`RESULTS-baseline.md`](./experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/nothink-5cond-boost-1024-20260323/RESULTS-baseline.md)
- [`RESULTS-expert_114_soft_bias_0.25.md`](./experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/nothink-5cond-boost-1024-20260323/RESULTS-expert_114_soft_bias_0.25.md)
- [`RESULTS-expert_114_soft_bias_0.5.md`](./experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/nothink-5cond-boost-1024-20260323/RESULTS-expert_114_soft_bias_0.5.md)
- [`RESULTS-expert_114_soft_bias_1.0.md`](./experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/nothink-5cond-boost-1024-20260323/RESULTS-expert_114_soft_bias_1.0.md)

Smoke artifacts:

- [`analysis.json`](./experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/smoke-20260323b/analysis.json)
- [`run_manifest.json`](./experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/smoke-20260323b/run_manifest.json)

## Reproducibility

- Yes: reproducible local reanalysis of the included `5cond` and `smoke-test` raw `.npy` files.
- No: a self-contained end-to-end rerun from scratch.
