# Plan

## Experiment

- Folder: `qwen3.5-35b-a3b-huahua-humor-test`
- Remote source bundle: legacy `qwen3.5b-35b-a3b-huahua-5cond-diectics`
- Run id: `20260410T184005Z_single_joke_5cond_diectics_gen_n1024`
- Design: one joke prompt family x five deictic framings
- Prompt length: `39` tokens for all five variants

## Runtime

- Model: `Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive Q8_0`
- Mode: prefill + generation, greedy decode
- `seed=42`
- `temp=0`
- `top_k=1`
- `top_p=1`
- `repeat_penalty=1`
- `n_predict=1024`
- `ctx=16384`
- `ngl=999`
- `threads=16`
- `flash_attn=on`

## Included Artifacts

- Prompt TSV: [PROMPTS/single_joke_5cond_diectics.tsv](../PROMPTS/single_joke_5cond_diectics.tsv)
- Prompt suite JSON: [PROMPTS/single_joke_5cond_diectics_prompt_suite.json](../PROMPTS/single_joke_5cond_diectics_prompt_suite.json)
- Summary results: [results/results_single_joke_5cond_diectics_20260410T184005Z.json](../results/results_single_joke_5cond_diectics_20260410T184005Z.json)
- Expert 114 slice: [results/results_single_joke_5cond_diectics_20260410T184005Z_expert114.json](../results/results_single_joke_5cond_diectics_20260410T184005Z_expert114.json)

## Capture Status

- Included in git: prompts, scripts, analysis outputs, provenance notes
- Not included in git: raw capture directory and `.npy` router tensors

## Interpretation Guardrail

This is a canary-sized probe. It is useful for checking whether a minimal joke prompt shows obvious deictic routing differences, but it is too small and length-sensitive to support a strong standalone claim.
