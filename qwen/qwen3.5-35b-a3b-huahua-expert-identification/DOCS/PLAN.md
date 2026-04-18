# Plan — Domain Specialist Expert Identification

## Goal

Map which of the 256 experts in Qwen3.5-35B-A3B specializes in which domain, using HauhauCS Q8_0. Characterize both prefill and generation routing patterns across 20 academic/professional domains.

This is a baseline routing survey — no interventions. The output is a per-domain expert leaderboard (by W and S) that feeds downstream intervention experiments.

## Model and Hardware

- Model: HauhauCS Qwen3.5-35B-A3B Q8_0
- Hardware: 2× RTX 5090
- Runtime: no-think, greedy generation, `-n 2056`
- KV cache cleared between prompts

## Prompt Design

- 60 prompts total: 20 domains × 3 prompts per domain
- Domains: philosophy, computer science, mathematics, medicine, history, comparative religion, linguistics, physics, political science, archaeology, and 10 others
- TSV: `PROMPTS/domain_specialist_probe_60_no_think.tsv`
- Think-mode TSV exists (`domain_specialist_probe_60_think.tsv`) but has not been run

## Measurements

Capture all 256 experts in both prefill and generation tracks:
- Prefill: `arr[:n_tokens_prompt]`
- Generation: `arr[n_tokens_prompt:]`
- Generation trimmed: removes layer-39 trim events (60 events in primary run)

Report per-expert W, S, Q across all layers, then per-domain rankings. Track Expert 114 specifically across all domains.

## Analysis

Run `METHOD/analyze_expert_identification.py` against the capture directory. Primary output: JSON with full per-domain expert tables, NPZ for numerical reanalysis, and markdown summary.
