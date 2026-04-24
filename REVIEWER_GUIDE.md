# Reviewer Guide

This guide is for reviewers who need to inspect the repository without reconstructing the project history from branch names or chat context.

## Safe Review Path

Start here:

1. [`README.md`](README.md) for the repository-level framing, current claims, and metric glossary.
2. [`qwen/README.md`](qwen/README.md) for the current Qwen evidence map.
3. [`qwen/qwen3.5-35b-a3b-huahua-114-greedy-reference/`](qwen/qwen3.5-35b-a3b-huahua-114-greedy-reference/) for the current strongest 35B E114 corroboration.
4. [`qwen/qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/`](qwen/qwen3.5-122B-A10B-huahua-single-prompt-processing-hum/) and [`qwen/qwen3.5-122B-A10B-huahua-five-cond-experience-probe/`](qwen/qwen3.5-122B-A10B-huahua-five-cond-experience-probe/) for the 122B E48 candidate-analogue thread.
5. [`token-confound-archive/`](token-confound-archive/) only after reading the Qwen path; it explains an invalidated early result and the methodology correction.

## Current Claim Boundary

The current repository claim is routing-mechanistic, not metaphysical:

- 35B: E114 at L14 is best read as a generated-output signal for inhabited first-person / phenomenological / mental-state register under the tested HauhauCS Qwen3.5-35B-A3B regimes.
- 122B: E48 is the clearest softmax-side generation carrier in the committed processing-hum / experience-probe followups, but it is a candidate analogue, not a proved clone of 35B E114.
- The repository does not claim that routing establishes consciousness, subjective experience, moral status, or phenomenology.

## What Not To Treat As Current Evidence

These artifacts are intentionally preserved, but they are not the current primary evidence:

- Pre-March-5 hierarchy-style routing-entropy claims.
- Raw all-token routing-entropy averages without token-position controls.
- DeepSeek, GPT-OSS, Ling-1T, and Qwen397B folders as direct evidence for E114/E48. They are useful for method comparison and provenance.
- Historical branch layouts as current publication structure.

## Existing Review Links

The following remote branches are intentionally retained for continuity:

- `main`: current public tree.
- `qwen-hauhau-5cond-smoke-only`: frozen historical snapshot.
- `qwen3.5-35b-a3b-huahua-experiments-2026-04-10`: frozen source snapshot for 35B bundles copied into `main`.
- `qwen35-greedy-reference-rerun`: reference branch for the deterministic 35B greedy rerun.

Reviewer-facing cleanup should avoid force-pushing or deleting these branches.

## Data Policy

Tracked artifacts should be small enough to review in GitHub: prompts, scripts, metadata, generated text, JSON summaries, Markdown reports, and lightweight plots.

Do not add:

- raw `.npy` router tensors
- model weights or checkpoints
- local environments
- Python caches
- credentials, hostnames, ports, private keys, API tokens, or `.env` files

Some historical folders still contain older raw/cached artifacts for provenance. Treat that as legacy history, not the current import policy.
