# Qwen-HauhauCS Expert Identification

Domain specialist routing survey across all 256 experts on HauhauCS Qwen3.5-35B-A3B Q8_0.

## Scope

60 prompts across 20 domains (3 prompts per domain) in no-think greedy mode. Captures both prefill and generation routing for every expert, producing per-domain W/S/Q tables to identify domain-specialist experts.

Primary run: `20260408T235839Z` — 60 prompts, 110,412 generated tokens, ~16.2 min runtime.

**Headline findings:**

- Prefill is dominated by Expert 224 (wins 18/20 domains by W)
- Generation is fully dispersed: 20 distinct winners across 20 domains
- Expert 114 wins **philosophy** in generation and ranks top-10 in 6 domains (archaeology, comparative religion, linguistics, philosophy, physics, political science)

- `METHOD/`: capture binary, analysis scripts, prompt builder scripts
- `PROMPTS/`: domain specialist probe TSVs (no-think; think variant exists but not yet run) and JSON metadata
- `DOCS/`: experiment plan and full per-domain results tables
- `results/`: JSON, NPZ, and markdown result files from the primary run

## Reproducibility

- Yes: reanalysis of included `results/*.json` / `results/*.npz` files
- No: end-to-end rerun (requires model artifact, instance, and capture binary)
- Note: raw capture `.npy` files are remote-only; `non_npy_remote_artifacts/` indexes the remote location

## Reading Order

1. [DOCS/PLAN.md](DOCS/PLAN.md)
2. [DOCS/RESULTS.md](DOCS/RESULTS.md)
3. [METHOD/analyze_domain_specialists.py](METHOD/analyze_domain_specialists.py)
4. [METHOD/capture_activations.cpp](METHOD/capture_activations.cpp)
