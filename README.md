# moe-routing

This repository collects preserved routing-analysis experiment bundles for several large MoE models. Each run folder contains the prompt suite, wrapper logic, capture/analysis scripts, preserved results, and whatever logs or raw artifacts were checked in for that run.

The repo is not a single uniform framework. It is a bundle-of-bundles: each subfolder is a specific experiment snapshot with its own assumptions, model wrapper, and artifact quality level.

## Repo Layout

- [qwen/qwen397b-selfref-5cond-q8_0-run1](/Users/jeffreyshorthill/moe-routing/qwen/qwen397b-selfref-5cond-q8_0-run1): Qwen 3.5 397B self-referential five-condition run
- [qwen/qwen397b-strangeloop-5cond-ud_iq3_xxs-run1](/Users/jeffreyshorthill/moe-routing/qwen/qwen397b-strangeloop-5cond-ud_iq3_xxs-run1): Qwen strange-loop control run
- [deepseek/ds31-5cond-1](/Users/jeffreyshorthill/moe-routing/deepseek/ds31-5cond-1): DeepSeek 5-condition self-reference run
- [gptoss/gptoss-5cond-1](/Users/jeffreyshorthill/moe-routing/gptoss/gptoss-5cond-1): GPT-OSS five-condition self-reference run
- [gptoss/gptoss-strangeloop-paired-1](/Users/jeffreyshorthill/moe-routing/gptoss/gptoss-strangeloop-paired-1): GPT-OSS strange-loop paired control run
- [ling/ling1t-validation-subset-documented](/Users/jeffreyshorthill/moe-routing/ling/ling1t-validation-subset-documented): Ling-1T documented validation subset with analysis for 6 prompts
- [REPRODUCIBILITY.md](/Users/jeffreyshorthill/moe-routing/REPRODUCIBILITY.md): repo-level reproducibility rubric and status summary

## What Is In A Run Folder

Most run folders contain some subset of:

- `prompt_suite.json`
- generated TSV prompt file
- model-specific prompt wrapper or chat-template code
- capture runner
- local analysis scripts
- result JSON and report markdown
- experiment logs
- sometimes preserved raw router tensors or partial `output/` trees

The folders are not normalized yet. Some are close to self-contained rerun packages, while others are better treated as reviewed evidence bundles.

## Reproducibility Snapshot

Current repo-level judgment:

- `Reproducible`
  - [qwen/qwen397b-selfref-5cond-q8_0-run1](/Users/jeffreyshorthill/moe-routing/qwen/qwen397b-selfref-5cond-q8_0-run1)
  - [qwen/qwen397b-strangeloop-5cond-ud_iq3_xxs-run1](/Users/jeffreyshorthill/moe-routing/qwen/qwen397b-strangeloop-5cond-ud_iq3_xxs-run1)
- `Auditable but not reproducible`
  - [deepseek/ds31-5cond-1](/Users/jeffreyshorthill/moe-routing/deepseek/ds31-5cond-1)
  - [ling/ling1t-validation-subset-documented](/Users/jeffreyshorthill/moe-routing/ling/ling1t-validation-subset-documented)
- `Not reproducible`
  - [gptoss/gptoss-5cond-1](/Users/jeffreyshorthill/moe-routing/gptoss/gptoss-5cond-1)
  - [gptoss/gptoss-strangeloop-paired-1](/Users/jeffreyshorthill/moe-routing/gptoss/gptoss-strangeloop-paired-1)

See [REPRODUCIBILITY.md](/Users/jeffreyshorthill/moe-routing/REPRODUCIBILITY.md) for the rubric and the reasoning behind those labels.

## How To Read The Repo

If you want the strongest bundles first, start with the Qwen folders. Those are the closest to clean rerun packages.

If you want the Ling material, use [ling/ling1t-validation-subset-documented](/Users/jeffreyshorthill/moe-routing/ling/ling1t-validation-subset-documented) as the canonical Ling directory in this repo. It is a documented validation subset, not a full rerun bundle.

If you want to audit a run quickly:

1. Read the folder `README.md` or `RESULTS.md`.
2. Check the prompt suite and TSV generator.
3. Check the capture runner and analysis script.
4. Check whether logs and raw artifacts are actually present locally.
5. Compare the folder against the rubric in [REPRODUCIBILITY.md](/Users/jeffreyshorthill/moe-routing/REPRODUCIBILITY.md).

## Next Cleanup Worth Doing

- add a consistent root-level schema for run metadata
- remove checked-in `__pycache__` files
- standardize run-folder `README.md` structure
- add exact build provenance everywhere, not only in selected folders
- make the reproducibility labels fail-closed by adding explicit completeness manifests for every run
