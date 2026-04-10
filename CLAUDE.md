# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

MoE routing experiment suite measuring Expert 114 behavior in Qwen3.5-35B-A3B and HauhauCS/Uncensored-Aggressive (Q8_0). All experiments use a custom llama.cpp capture binary to record per-token, per-layer router logits during prefill and generation, then analyze routing with Python scripts.

**Core metrics used everywhere:**
- **W** = S × Q (unconditional contribution of an expert)
- **S** = selection rate (fraction of tokens where expert is in top-8)
- **Q** = conditional weight (mean weight given selection)
- **KL-manip** = KL divergence between manipulation and calibration regions
- **L1/L2/L3** = prompt categories of increasing self-reference depth

## Canonical Experiment Folder Structure

Every folder follows `qwen3.5-35b-a3b-huahua-<experiment>/` naming and this layout:

```
README.md       — researcher entrypoint, headline result, reading order
MANIFEST.md     — file inventory, what's tracked vs. excluded
DOCS/PLAN.md    — hypothesis, model config, prompt design, measurements
DOCS/RESULTS.md — findings with tables, interpretation, limitations
METHOD/         — Python analysis scripts + capture_activations.cpp source
PROMPTS/        — .tsv prompt suites (think/no-think) + .json metadata
results/        — JSON outputs, markdown summaries, generated text
```

The `qwen3.5-35b-a3b-huahua-agressive-experts/` folder is the reference template.

The strangeloop folder (`qwen3.5-35b-a3b-huahua-strangeloop/`) is the standalone paired definiteness-control experiment. The former bundle sub-experiments (processing-hum, five-cond-experience-probe, domain-expert-probe-3chunk) are now separate root-level folders.

## What's Excluded from Git

`.gitignore` excludes: `*.npy`, `*.npz`, `*.tar`, `captures/`, `Documentation/`, `.DS_Store`, `__pycache__/`, `*.pyc`. These are large binary tensors, raw capture directories, and local reference material.

## No Build System

There is no Makefile, requirements.txt, or package manager. Python scripts use numpy + stdlib. The C++ capture binary (`capture_activations.cpp`) is compiled against llama.cpp externally on Vast.ai instances (pinned build 8493 / 1772701f). Shell scripts in `METHOD/` (e.g., `bootstrap_remote_instance.sh`) handle remote deployment.

## Key Shared Code

`qwen_router.py` (found in most METHOD/ directories) provides:
- `softmax()`, `reconstruct_probs()` — top-8 select + renorm routing reconstruction
- `normalized_entropy()` — sparse routing entropy (normalized by log2(8))
- `js_divergence()` — Jensen-Shannon divergence for routing distribution comparison
- Constants: `N_EXPERTS=256`, `TOP_K=8`

## Writing New Experiment Documentation

When creating a new experiment folder:
1. Copy the canonical structure (README, MANIFEST, DOCS/, METHOD/, PROMPTS/)
2. PLAN.md should state: goal/hypothesis, model+hardware, prompt design, measurements, analysis method
3. RESULTS.md should include: TL;DR, headline table with W/S/Q values, per-condition breakdown, interpretation with explicit limitations
4. Use relative paths in cross-references between experiment folders (e.g., `../qwen3.5-35b-a3b-huahua-6cond-moe-manips/`)
5. MANIFEST.md must note what's excluded from git (npy, npz, captures, tar archives)

## Cross-References Between Experiments

Experiments form a dependency chain. Key relationships:
- `agressive-experts` is the original 5-condition causal study (template)
- `vs-base-run1` compares vanilla vs. HauhauCS on 150 prompts
- `6cond-moe-manips` → `6cond-hvac` (same design, different topic domain, tests ML-specialist hypothesis)
- `expert-identification` → `philosophy-experts-bias` (identification feeds suppression targets)
- `114-pm` is the single-prompt bias sweep sibling of `6cond-moe-manips`
- `strangeloop` is the standalone paired definiteness-control run (A/B deictic contrast on self-referential content)
- `single-prompt-processing-hum`, `five-cond-experience-probe`, `domain-expert-probe-3chunk` are independent experiments (formerly bundled under strangeloop, now separate)
- `humor-test` is a standalone canary study testing E114 response to joke structure

## Branch

Active branch: `qwen3.5-35b-a3b-huahua-experiments-2026-04-10`.
