# Qwen3.5-35B-A3B HauhauCS — Strangeloop Bundle

Multi-experiment workspace on HauhauCS Qwen3.5-35B-A3B Q8_0. Groups four related experiment families sharing a common capture setup and shared analysis utilities.

## Experiment Families

| Subfolder | Description |
|---|---|
| [`single-prompt-processing-hum/`](single-prompt-processing-hum/) | Per-token phenomenological probe — "Processing Hum" single prompt, 1024 tokens, E114 peaks inside phenomenological language |
| [`strangeloop-paired/`](strangeloop-paired/) | 30 A/B paired routing experiment — Gödel/Escher/bootstrap/Quine/tangled-hierarchy prompt pairs, prefill-only KL-to-Cal contrast; KL-manip Wilcoxon p=6.3e-05 |
| [`five-cond-experience-probe/`](five-cond-experience-probe/) | 5-condition experience probe — 15 prompts (P09–P11 × A–E), E114 top manipulation expert on all 15; KL-manip p=6.3e-05 |
| [`domain-expert-probe-3chunk/`](domain-expert-probe-3chunk/) | Long-prompt domain cramming — 60 domain questions collapsed into 3 token-balanced prompts; generation expert set diverges from prefill; E114 rises to rank 7 in chunk C |

`shared-tools/` contains analysis and capture utilities used across all four families.

## Structure

Each subfolder follows the standard layout:

```
subfolder/
├── DOCS/        — PLAN.md, RESULTS.md
├── METHOD/      — analysis scripts and C++ capture binary
├── PROMPTS/     — TSV and JSON prompt suites
└── raw/         — timestamped capture directories (.npy excluded from git)
    results/     — JSON and markdown analysis outputs
```

## Reproducibility

- Yes: reanalysis of included JSON and TSV result files in each `results/` directory
- No: end-to-end rerun (requires model artifact, Vast.ai instance, and pinned llama.cpp build 8493)
- Note: `.npy` router tensors are excluded from git

## Reading Order

1. [DOCS/PLAN.md](DOCS/PLAN.md) — bundle overview and shared setup
2. [DOCS/RESULTS.md](DOCS/RESULTS.md) — cross-experiment summary
3. Individual experiment DOCS/ for detail
