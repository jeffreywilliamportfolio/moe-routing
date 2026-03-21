# Learning How MoE Models Route Self Referential Content vs Baseline

Local folder name is historical. The public GitHub repository is `moe-routing`.

## If you know nothing about this repo

This repo asks a simple question:

**When you change the wording of a prompt, does a Mixture-of-Experts language model send the token through different experts inside the model?**

In plain language, I am not mainly studying what the model says. I am studying **which internal expert modules it uses while producing or reading the prompt**.

The basic workflow in this repo is:

1. Give an MoE model carefully matched prompts.
2. Capture the model's internal router signals during inference.
3. Measure which experts were selected and how concentrated or diffuse the routing was.
4. Compare those internal patterns across prompt conditions.

If you want the one-sentence version:

**This is a repository of experiments about how MoE LLMs internally route tokens across experts when prompt wording changes, plus an archive of a confound that invalidated the first main result.**

## Short version

This repository studies **Mixture-of-Experts routing behavior inside open-weight LLMs** under tightly controlled prompt changes.

The simplest accurate description is:

- I capture router tensors during inference.
- I reconstruct or summarize the per-token expert distribution.
- I compare how routing changes when prompt content changes in controlled ways.

This repository is **not** primarily about output quality, benchmark scores, or production MoE load balancing.

It also preserves an important failure: an earlier “routing entropy hierarchy” result later collapsed under token-position controls. That failure is part of the point of the repo now, because it exposed a real measurement confound.

## What this repo is

- A research notebook plus reproducible experiment-bundle repository for MoE routing analysis.
- A place where I pin model artifacts, capture code, prompt suites, logs, and analysis together so a run can be inspected later.
- A record of my first few weeks doing end-to-end ML and mechanistic-interpretability-style work, including the wrong turns.
- A measurement project about **router behavior during inference**, especially prompt prefill.

## What this repo is not

- Not a polished paper with one clean headline claim.
- Not a general benchmark of model quality.
- Not a neuroscience result and not evidence of consciousness.
- Not a production-systems study of MoE serving efficiency.
- Not a claim that “routing entropy” by itself is a valid master metric.

## Zero-context FAQ

### What is an MoE model in this repo?

An MoE model is a model that has many candidate expert sub-networks, but only some of them are used for a given token. The router decides which experts get used.

### What does “routing” mean here?

It means the model's internal choice of which experts a token is sent to. This is an internal computation signal, not a serving-system queue or hardware scheduler.

### What is the repo actually measuring?

It measures things like:

- which experts were selected
- how concentrated the routing weights were
- how routing changed between matched prompt conditions
- whether those changes reproduce across reruns

### Why is this not a load-balancing repo?

Because the question here is not “are experts evenly used across the fleet?” The question is “does the model internally redirect computation when the prompt meaning changes?”

### Why are there archives of wrong or failed work?

Because one of the most important findings in the repo is that an earlier positive-looking result was confounded by token position. Keeping that record visible is part of the scientific value of the project.

## Important clarification: this is not “load balancing”

One common reaction is: “isn’t routing entropy just load balancing?”

The answer is: **not in the way this repo uses it**.

In production MoE systems, people care about load balancing because they do not want a few experts overloaded while others sit idle. That is a training or serving concern.

Here, I am using router outputs as an **internal model signal**:

- which experts got selected
- how concentrated or diffuse the selected routing weights were
- whether a prompt manipulation redirected routing relative to a matched baseline

Routing entropy in this repo means: “how spread out is the routed distribution for this token or region?” It does **not** mean “is the model globally load balanced?” After the confound discovery, routing entropy is treated as a **descriptive companion metric**, not the sole target.

## Why the repo is named `llama-eeg-tests`

The name is historical.

The project started from an EEG / IIT-inspired question: can a model-internal signal play the role of a consciousness-relevant or complexity-relevant marker?

That original framing produced the first wave of experiments. The repo kept the original name even after the work became more concrete and mechanistic.

The current repo is better understood as:

**MoE routing measurement under controlled prompt manipulations, plus an archive of the token-position confound that killed the first headline result.**

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

That split matters. They should not be read as equally strong evidence.

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

## Current purpose of the repo

The current purpose is narrower and more defensible than the original ambition.

The main question now is:

**Under token-matched, position-controlled conditions, do prompt manipulations cause reproducible MoE routing redirection?**

Examples of the kinds of things this repo now tries to support:

- paired self-reference versus matched third-person controls
- wrong-answer versus right-answer commitment-token comparisons
- base model versus retrained model routing comparisons
- cross-model comparisons of the same failure mode or effect

## What counts as a stronger claim here

Post-confound, the default standard is:

- prefill-only by default
- token-matched prompt pairs
- exact tokenizer verification before running
- last-token or boundary-controlled measurements
- pinned model and capture-binary provenance
- machine-readable results plus execution logs

Routing entropy is still reported, but not as the only thing that matters. The repo now also emphasizes:

- divergence from baseline routing
- expert overlap
- cross-layer agreement or disagreement
- reproducibility under fixed settings

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

That means:

- some old artifacts are wrong in their conclusions
- they are still useful as evidence
- the archive preserves both the mistaken claim and the later explanation

## What I would want a new reader to know

If you only remember five things, they should be:

- This repo studies **router behavior**, not just outputs.
- It is **not** a systems load-balancing project.
- The first big positive-looking result was later invalidated by a token-position confound.
- That invalidation is preserved on purpose because it is scientifically useful.
- The stronger work in the repo is the post-March-5, token-controlled, reproducibility-focused work.

## Current experiment examples

Representative bundles on the current branch include:

- [`experiments/deepseek/ds31-5cond-1`](./experiments/deepseek/ds31-5cond-1)
  - DeepSeek V3.1 five-condition routing bundle
- [`experiments/gptoss/gptoss-selfref-paired-2`](./experiments/gptoss/gptoss-selfref-paired-2)
  - GPT-OSS paired self-reference routing bundle
- [`experiments/Qwen3.5-35B-A3B-vs-HauhauCS-Qwen3.5-uncensored`](./experiments/Qwen3.5-35B-A3B-vs-HauhauCS-Qwen3.5-uncensored)
  - Base-versus-retrain Qwen routing comparison
- [`experiments/token-confound-archive`](./experiments/token-confound-archive)
  - outsider-readable archive of the confound story
- [`experiments/token-confound-archive-mechinterp`](./experiments/token-confound-archive-mechinterp)
  - same confound story packaged more for a mech-interp audience

## Practical status

This repo contains:

- current experiment bundles
- historical bundles
- corrected methodology
- narrative context
- preserved mistakes

That mix is intentional. The right way to read it is not “everything here is equally validated,” but:

**this is a transparent record of a real research process, including the part where the original story failed and the methodology had to change.**
