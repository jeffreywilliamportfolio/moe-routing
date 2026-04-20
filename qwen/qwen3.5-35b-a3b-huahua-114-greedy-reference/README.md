# Qwen3.5-35B-A3B HauhauCS — E114 greedy reference

Canonical deterministic greedy rerun for the HauhauCS Qwen3.5-35B-A3B E114 residual-analysis workflow.

This folder supersedes the earlier stochastic/default-sampling [`../qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](../qwen3.5-35b-a3b-huahua-114-selfref-heldout/) folder as the reference anchor. The earlier run is retained as historical corroboration; this folder is the clean rerun with explicit greedy decoding.

## Scope

Two captures were produced on a fresh 2x RTX 5090 Vast.ai instance:

- **Single prompt:** `single_prompt_processing_hum_no_think.tsv`, cap `-n 1024`.
- **Heldout:** 20 matched-token prompts, 10 predicted fire and 10 predicted nofire, cap `-n 256`.

Both captures used the same deterministic decoding flags:

```text
--temp 0 --top-k 1 --seed 0
```

Captured tensors at L13/L14/L15:

- router: `ffn_moe_logits-{13,14,15}`
- residual: `attn_post_norm-{13,14,15}`

Raw `.npy` router/residual tensors are excluded from git. Non-`.npy` manifests, generated text, token JSON, analysis outputs, provenance, and method scripts are included.

## Headline Result

E114 at L14 is a strong discriminator for **inhabited first-person phenomenological register in the generated output** under matched lexical conditions.

Heldout L14 trimmed-generation W114:

| class | n | mean-of-means | stddev-of-means | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| fire | 10 | 0.068089 | 0.034584 | 0.017242 | 0.099368 |
| nofire | 10 | 0.003249 | 0.006195 | 0.000000 | 0.019354 |

- fire/nofire ratio: 20.955x
- Cohen's d: 2.61
- range overlap: yes, driven by N08 crossing into the target inhabited register during generation

The central accomplishment is not simply "self-reference prompt vs. non-self-reference prompt." It is that E114 discriminates generated register under matched lexical conditions. Fire prompts that generated technical explanation were low; a nofire prompt about a cat's purr rose because the generated answer entered an inward experiential voice.

## Reading Order

1. [`DOCS/RESULTS.md`](DOCS/RESULTS.md) — full results, interpretation, and comparison to the prior stochastic heldout.
2. [`DOCS/COMMANDS.md`](DOCS/COMMANDS.md) — sanitized execution log for provisioning, capture, copy-back, analysis, and cleanup.
3. [`provenance/`](provenance/) — capture config, prompt checksums, and remote environment facts.
4. [`heldout/analysis/heldout_20260418T160353Z_greedy/heldout_stats.tsv`](heldout/analysis/heldout_20260418T160353Z_greedy/heldout_stats.tsv) — per-prompt heldout W/S/Q table.
5. [`heldout/analysis/heldout_20260418T160353Z_greedy/heldout_timeseries_top4.png`](heldout/analysis/heldout_20260418T160353Z_greedy/heldout_timeseries_top4.png) — top-2 fire and top-2 nofire temporal traces.

## Folder Contents

- [`DOCS/`](DOCS/): `PLAN.md`, `RESULTS.md`, `COMMANDS.md`.
- [`METHOD/`](METHOD/): capture binary source, router reconstruction, bootstrap, heldout analysis, and single-prompt Step 1-3 scripts.
- [`PROMPTS/`](PROMPTS/): single-prompt TSV, heldout prompt TSV, and heldout class sidecar.
- [`provenance/`](provenance/): deterministic capture config, sanitized remote environment, prompt checksums.
- [`single_prompt/`](single_prompt/): non-`.npy` raw capture artifacts and Step 1-3 analysis outputs.
- [`heldout/`](heldout/): non-`.npy` raw heldout artifacts and heldout analysis outputs.

## Boundary

This establishes E114 as a strong discriminator for the target register in this model/regime. It does not establish uniqueness relative to all neighboring experts, and it does not claim generalization across models, seeds, templates, or quantization regimes. Those are the next questions.
