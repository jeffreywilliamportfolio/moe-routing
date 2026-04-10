# Repository Guidelines

## Project Structure & Module Organization

This repository is organized by experiment family, mostly under `qwen3.5-35b-a3b-huahua-*`. Standard experiment folders contain:

- `README.md` and `MANIFEST.md`: scope and inventory
- `DOCS/`: `PLAN.md`, `RESULTS.md`, optional reproduce notes
- `METHOD/`: Python analysis scripts, shell runners, and `capture_activations.cpp`
- `PROMPTS/`: TSV prompt sets and JSON suites
- `results/`: JSON/Markdown outputs and logs

Some larger studies also include `raw/`, `captures/`, or `non_npy_remote_artifacts/`. `Documentation/` is local reference material and is gitignored.

## Build, Test, and Development Commands

- `python3 METHOD/<script>.py --help`: inspect analysis or prompt-builder options
- `bash METHOD/bootstrap_remote_instance.sh`: prepare a remote `llama.cpp` environment and capture binary
- `bash METHOD/run_*.sh`: execute an experiment capture workflow
- `git lfs pull`: required for `qwen3.5-35b-a3b-huahua-agressive-experts/`

There is no single project-wide build command; contributors usually work inside one experiment folder at a time.

## Coding Style & Naming Conventions

Use 4-space indentation for Python and keep shell scripts POSIX/Bash-friendly with `set -euo pipefail`. Prefer descriptive, experiment-specific filenames such as `build_domain_expert_probe_3chunk_no_think.py` and timestamped output names like `results_<study>_20260410T173400Z.json`.

Keep folder names in the established `qwen3.5-35b-a3b-huahua-<study>` pattern. Preserve artifact filenames when they already appear in saved results or manifests.

## Testing Guidelines

There is no centralized `pytest` suite. Validation is artifact-based:

- rerun the relevant analysis script on included JSON/TSV outputs
- confirm `README.md`, `MANIFEST.md`, and `DOCS/` match the folder contents
- check that generated artifacts remain ignored where required (`*.npy`, `*.npz`, `*.tar`, `captures/`)

When adding a new experiment, include at least one reproducible analysis path from `METHOD/` to `results/`.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, often with a scope prefix, for example:

- `Initial commit: standardize all experiment folders to qwen3.5-35b-a3b-huahua-* naming`
- `Consolidate philosophy-experts-bias: 5,404 → 65 tracked files`

PRs should state which experiment folders changed, whether artifacts were regenerated or only reorganized, and any excluded large files or remote-only data. Include before/after structure notes when renaming folders or normalizing layouts.

## Data & Configuration Notes

Do not commit large raw tensors or archives. `.npy`, `.npz`, `.tar`, `captures/`, and `Documentation/` are intentionally excluded. Treat remote paths, model locations, and Vast.ai-specific settings in shell scripts as environment-specific defaults, not portable assumptions.
