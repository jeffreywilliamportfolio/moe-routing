# Qwen-HauhauCS Philosophy Expert Cluster Suppression

Expert suppression study targeting the philosophy core cluster on HauhauCS Qwen3.5-35B-A3B Q8_0.

## Scope

Suppresses experts E114, E87, E170, and E68 (the philosophy core cluster identified in the domain specialist survey) during generation of philosophical and domain specialist content. Runs the 60-prompt domain specialist probe in both no-think and think modes across multiple suppression bias levels (−5, −8 and combinations).

Multiple capture rounds were executed:
- `m8-p5` series: E114 at −8, others at −5
- `p8-only` series: focused single-expert suppression variants (62 granular subdirectories)

- `METHOD/`: analysis and builder scripts, C++ capture binary, shell run scripts
- `PROMPTS/`: 60-prompt domain specialist TSVs (no-think and think) and JSON metadata
- `DOCS/`: experiment plan and results summary
- `captures/`: timestamped raw capture directories (`.npy` files excluded from git)
- `pulled-generated-text-m8-p5/`, `pulled-generated-text-p8-only/`: per-prompt generated text pulled from remote
- `results-m8-p8/`, `results-partial-m8-p5/`: analysis output directories
- `prompt-results.md`: 2.1 MB compiled transcript of all generated text and results

## Reproducibility

- Yes: reanalysis of included generated text in `pulled-generated-text-*/` directories
- No: end-to-end rerun (requires model artifact, instance, and multi-expert suppression capture binary)
- Note: `.npy` files excluded from git; `non_npy_remote_artifacts/` indexes the remote location

## Reading Order

1. [DOCS/PLAN.md](DOCS/PLAN.md)
2. [DOCS/RESULTS.md](DOCS/RESULTS.md)
3. [METHOD/analyze_domain_specialists.py](METHOD/analyze_domain_specialists.py)
