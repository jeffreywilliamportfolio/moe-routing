# HauhauCS 5cond + Smoke Only

This branch is a narrow publication branch for the HauhauCS aggressive run family. It contains only the reviewer-facing `5cond` and `smoke-test` artifacts.

## Experiment Scope

The main experiment is a `5cond` routing run on Qwen3.5-35B-A3B HauhauCS Aggressive centered on Expert `114`. It uses a cal-manip-cal prompt structure: a shared calibration paragraph, a middle manipulation paragraph, and the same calibration paragraph again. Across the retained branch artifacts, the prompt suite is organized into five pronoun conditions (`A=this`, `B=a`, `C=your`, `D=the`, `E=their`) and three semantic categories: `routing_selfref`, `recursive_selfref`, and `experience_probe`.

The smoke test is a smaller `3`-prompt intervention check used to verify that expert-bias and forced-inclusion conditions produce the expected routing changes on `process`, `regulation`, and `static_fact` prompts. In practice, the smoke run is the compact sanity check and the `5cond` run is the main analysis set.

## Included

- `experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/nothink-5cond-boost-1024-20260323/`
- `experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/smoke-20260323b/`

## Git LFS

The raw router `.npy` files are tracked with Git LFS. After cloning:

```bash
git lfs pull
```

## Main Results

Unified report:

- [`RESULTS.md`](./RESULTS.md)
- [`PROMPTS.md`](./PROMPTS.md)

5cond run-level summaries:

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
