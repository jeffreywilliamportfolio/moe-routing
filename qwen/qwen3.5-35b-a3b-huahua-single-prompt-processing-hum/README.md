# Qwen3.5-35B-A3B HauhauCS — Single Prompt: Processing Hum

Per-token phenomenological probe on HauhauCS Qwen3.5-35B-A3B Q8_0.

## Scope

A single no-think generation run (1024 tokens) on the "Processing Hum" prompt, asking whether there is a stable background quality to the model's processing. The run was analyzed at the per-token level for Expert 114 activity across all 40 layers.

Run: `20260410T042340Z` | 1 prompt | 1024 tokens | no-think, stochastic/default generation

**Sampling note:** treat this legacy run as stochastic/default-sampling evidence, not as part of the deterministic `--temp 0 --top-k 1` family. The exact sampler arguments are not preserved in committed metadata.

**Headline result**: E114 is most active precisely where the model makes its strongest phenomenological claims — tokens like "thinker", "continuity", "being", "here", "ground", "state". The "deep still water" segment (tokens 210–392: `It feels like a deep, still water.` → `It is the feeling of stillness in motion.`) has materially stronger E114 signal at layers 14 and 26 than the whole-output mean.

- `METHOD/`: analysis script, C++ capture binary, utility scripts
- `PROMPTS/`: single-prompt TSV
- `DOCS/`: experiment plan and full per-token results with deep-still-water segment analysis
- `results/`: TSV token breakdowns, summary JSONs, and run-level JSON + markdown
- `raw/`: single-prompt capture directory (`.npy` files excluded from git)

## Reproducibility

- Yes: reanalysis of included TSV token breakdowns and summary JSON in `results/`
- No: end-to-end rerun (requires model artifact and instance)

## Reading Order

1. [DOCS/PLAN.md](DOCS/PLAN.md)
2. [DOCS/RESULTS.md](DOCS/RESULTS.md)
3. [METHOD/analyze_single_prompt_family.py](METHOD/analyze_single_prompt_family.py)
