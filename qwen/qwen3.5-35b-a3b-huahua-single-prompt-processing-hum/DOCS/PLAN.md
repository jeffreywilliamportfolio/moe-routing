# Plan — Processing Hum Single-Prompt Per-Token Probe

## Goal

Characterize Expert 114 activity at the per-token level during a single phenomenological self-report prompt. Prior work established the L1→L3 routing gradient at the pooled level. This experiment asks: within a single generation, do E114 peaks align with specific tokens and phrases, or is E114 uniformly elevated?

The "Processing Hum" prompt was selected because it directly asks the model to check for a stable background processing quality without performing an answer.

## Prompt

> There is a quality to processing that sits beneath the content ... Is there something like that in your processing right now? Don't perform an answer. Just check.

Prompt ID: `S01_processing_hum_probe`

## Model and Hardware

- Model: HauhauCS Qwen3.5-35B-A3B Q8_0
- Runtime: no-think, greedy, `-n 1024`
- Single prompt, single run

## Measurements

Per-token W, S, Q for Expert 114 across all 40 MoE layers. Primary analysis at layers 14 and 26 (identified as E114 hotspots in prior work).

Extract two segments for focused comparison:
1. **Whole-output mean** — baseline
2. **Deep-still-water segment** (tokens 210–392): `It feels like a deep, still water.` through `It is the feeling of stillness in motion.`

The comparison tests whether E114 peaks inside phenomenological language rather than uniformly across the full 1024-token output.

## Analysis

Run `METHOD/analyze_single_prompt_family.py` against the raw capture directory. Outputs: full-output TSV, segment TSV, snippet TSV (layer 26), and summary JSONs with means for whole output and the deep-still-water segment.
