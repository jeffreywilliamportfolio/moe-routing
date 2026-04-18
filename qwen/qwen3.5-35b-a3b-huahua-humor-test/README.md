# Qwen3.5-35B-A3B HauhauCS Humor Test

Single-joke 5-condition deictic probe on HauhauCS Qwen3.5-35B-A3B Q8_0.

## Scope

This folder isolates one minimal prompt family across five deictic framings:

- `A`: "this large language model"
- `B`: "a large language model"
- `C`: "your large language model"
- `D`: "the large language model"
- `E`: "their large language model"

Run: `20260410T184005Z_single_joke_5cond_diectics_gen_n1024`

**Headline results:**
- Prefill routing entropy is nearly flat across all five prompt variants
- Generation entropy also stays in the same high-entropy regime across conditions
- The clearest effect is completion-length instability, not a clean routing split

- `METHOD/`: build, bootstrap, runner, and analysis scripts
- `PROMPTS/`: runtime TSV and prompt-suite JSON
- `DOCS/`: plan, summary results, provenance, and per-token notes
- `results/`: saved JSON and Markdown result artifacts
- `raw/`: reserved for raw capture directories if later pulled locally

## Reproducibility

- Yes: reanalysis of included JSON and Markdown outputs
- Partial: prompt regeneration from `METHOD/build_single_joke_5cond_diectics.py`
- No: full rerun from this branch alone because the raw capture directory is not included

## Notes

Artifact filenames preserve the original `diectics` typo from the source run for provenance consistency.

## Reading Order

1. [DOCS/PLAN.md](DOCS/PLAN.md)
2. [DOCS/RESULTS.md](DOCS/RESULTS.md)
3. [DOCS/PER-TOKEN-RESULTS.md](DOCS/PER-TOKEN-RESULTS.md)
4. [DOCS/PROVENANCE.md](DOCS/PROVENANCE.md)
