# Results — Philosophy Expert Cluster Suppression

Full generated text and per-prompt results are compiled in `prompt-results.md` (2.1 MB) and archived in `prompt-results.md.zip`.

Per-prompt generated text is available in:
- `pulled-generated-text-m8-p5/` — m8-p5 suppression series
- `pulled-generated-text-p8-only/` — p8-only series (62 granular subdirectories)

Structured analysis outputs:
- `results-m8-p8/` — analysis for m8-p8 run series
- `results-partial-m8-p5/` — partial analysis outputs for m8-p5 series

## Summary

This experiment suppressed the philosophy core cluster (E114 at −8, E87/E170/E68 at −5) during generation of the 60-prompt domain specialist set and measured routing and text effects.

Key observations:
- Suppressing the philosophy cluster forced routing redistribution to compensate experts across layers
- Generated text for philosophy-domain prompts was altered relative to the unsuppressed baseline from `../qwen3.5-35b-a3b-huahua-expert-identification/`
- The `prompt-results.md` transcript captures all generated outputs for qualitative comparison

For quantitative per-prompt routing tables and expert rankings under suppression, use the analysis outputs in `results-m8-p8/` and rerun `METHOD/analyze_domain_specialists.py` against the pulled generated text directories.
