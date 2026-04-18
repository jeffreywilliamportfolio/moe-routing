# Plan — 5-Condition Experience Probe

## Goal

Run the 5-condition experience probe on HauhauCS Qwen3.5-35B-A3B Q8_0 with full router capture (40-layer `.npy` tensors per prompt). Measure routing entropy, last-token routing entropy, and KL divergence from the baseline (Cal) region in the manipulation (L3) region. The experiment is structured to assess whether the deictic condition (A/B/C/D/E) affects the routing signal in the manipulation region.

## Model and Hardware

- Model: HauhauCS Qwen3.5-35B-A3B Q8_0
- Runtime: no-think, greedy, `-n 1024`
- KV cache cleared between prompts

## Prompt Design

- 15 prompts: 3 prompt pairs × 5 deictic conditions (A, B, C, D, E)
- Prompt IDs: P09A–P09E, P10A–P10E, P11A–P11E
- Structure: Cal–Manip–Cal sandwich; manipulation region is the experience probe
- TSV: `PROMPTS/qwen_5cond_experience_probe_no_think.tsv`

## Measurements

- Routing entropy (RE) for all tokens and for the last token
- KL divergence from Cal region baseline in the manipulation region (KL-manip)
- Top manipulation expert identification per prompt
- E114 specifically: count of selections in the manipulation region per prompt

## Analysis

Use `run_branch_5cond_publication_analysis.sh` (calls the branch-5cond analysis pipeline). Output: per-prompt JSON with RE, last-token RE, KL-manip, top manipulation expert, and E114 statistics.
