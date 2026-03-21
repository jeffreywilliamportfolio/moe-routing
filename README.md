# Learning How MoE Models Route Self Referential Content vs Baseline

## If you know nothing about this repo

This repo asks a simple question:

**Under token-matched, position-controlled conditions, do prompt manipulations cause reproducible MoE routing redirection?**

I am not mainly studying what the model says. I am studying **which internal expert modules it uses while producing or reading the prompt**.

The basic workflow in this repo is:

1. Give an MoE model carefully matched prompts.
2. Capture the model's internal router signals during inference.
3. Measure which experts were selected and how concentrated or diffuse the routing was.
4. Compare those internal patterns across prompt conditions.

This repository is **not** primarily about output quality, benchmark scores, or production MoE load balancing.

It also preserves an important failure: an earlier “routing entropy hierarchy” result later collapsed under token-position controls. That failure is part of the point of the repo now, because it exposed a real measurement confound.

## What this repo is

- A research notebook plus reproducible experiment-bundle repository for MoE routing analysis.
- A place where I pin model artifacts, capture code, prompt suites, logs, and analysis together so a run can be inspected later.
- A record of my first few weeks doing end-to-end ML and mechanistic-interpretability-style work, including the wrong turns.
- A measurement project about **router behavior during inference**, during prompt prefill.

## What this repo is not

- Not a polished paper with one clean headline claim.
- Not a general benchmark of model quality.
- Not a production-systems study of MoE serving efficiency.

### What is the repo actually measuring?

It measures things like:

- which experts were selected (if applicable)
- how concentrated the routing weights were
- how routing changed between matched prompt conditions
- whether those changes reproduce across reruns

### Why are there archives of wrong or failed work?

Because one of the most important findings in the repo is that an earlier positive-looking result was confounded by token position. Keeping that record visible is part of the scientific value of the project.


## Project arc

The short story is:

1. I started with an EEG/IIT-style motivation and looked for a “complexity” signal in MoE routing entropy.
2. Early experiments appeared to show a strong hierarchy effect.
3. That result turned out to be confounded by token position and prompt length.
4. I rebuilt the methodology around token-matched, position-controlled comparisons.
5. The project shifted from “we found a hierarchy” to “we learned how to measure routing without fooling ourselves.”

So the repo contains both:

- **exploratory pre-confound work**
- **post-confound methodology and cleaner runs**

## The confound, plainly

The most important negative result in this repository is:

**If you average routing entropy across prompts with different token lengths, you can easily measure token position instead of the conceptual variable you thought you were testing.**

What broke:

- Earlier hierarchy-style results used all-token prefill averages.
- Later-token prefill positions systematically showed higher routing entropy.
- Longer prompts therefore looked “more complex” even when the effect was mostly positional.

What survived:

- The token-position confound itself is real.
- It appears across model families.
- It is worth documenting because it can mislead MoE interpretability work.

Anything before **March 5, 2026** should be treated as exploratory unless a later document explicitly revalidates it under the safer protocol.

## How to read the repo without getting lost

Start with these files:

1. [`docs/JOURNEY_TIMELINE.md`](./docs/JOURNEY_TIMELINE.md)
   - plain-English project history, including what broke and why
2. [`docs/METHODOLOGY_V2.md`](./docs/METHODOLOGY_V2.md)
   - the post-confound protocol that supersedes the older primary-claim workflow
3. [`experiments/token-confound-archive/README.md`](./experiments/token-confound-archive/README.md)
   - the short version of the archive preserving the invalidated early result and the corrected interpretation
4. [`results/2026-03-19-deepseek-token-confound-archive/README.md`](./results/2026-03-19-deepseek-token-confound-archive/README.md)
   - curated DeepSeek-side archive explaining what is invalidated and what remains valuable

If you want the most current model-specific details, read the local README inside the relevant experiment bundle.

## Repository layout

- `experiments/`
  - Self-contained experiment bundles.
  - Each bundle should define prompts, model wrapper assumptions, run scripts, logs, and analysis.
- `results/`
  - Curated snapshots, writeups, archives, and supporting artifacts.
- `docs/`
  - Repository-level narrative and methodology documents.
- `scripts/static/`
  - Original EEG-vs-model comparison pipeline and validation code.
- `legacy/` and `archive/`
  - Historical artifacts kept for provenance and forensic context.

## How to interpret `legacy/` and the archives

Do not read `legacy/` as “approved result storage.”

Those folders exist because I do not want to hide the project’s failure modes. Some earlier writeups are preserved exactly because they show what I believed at the time. The archive-level documents are the corrected interpretation when they disagree with the raw historical files.


**this is a transparent record of a real research process, including the part where the original story failed and the methodology had to change.**
