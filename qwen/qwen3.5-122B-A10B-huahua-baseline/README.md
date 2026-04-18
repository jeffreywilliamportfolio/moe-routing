# Qwen3.5-122B-A10B Huahua Baseline

This root bundle is the canonical 122B baseline/reference surface for `HauhauCS/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive`.

## Scope
- Keep the root surface limited to the completed 150-prompt 5-condition diectic baseline.
- Treat later 122B runs as follow-ups under `followups/` rather than mixing them into the baseline root.
- Report `W/S/Q` first and keep entropy or other legacy metrics secondary.

## Canonical Run
- Run id: `20260412T141943Z_qwen122_5cond_prompt_suite_gen_n2048`
- Prompt source: `PROMPTS/prompt_suite.json`
- Runtime TSV: `PROMPTS/qwen122_5cond_prompt_suite_no_think.tsv`
- Entry points: `RESULTS.md`, `INTERPRETATION.md`, `results-generated.txt`

## Architecture Note
- This model is a separate interpretability regime from the 35B Hauhau family.
- The model card describes a hybrid stack of Gated DeltaNet linear attention plus full softmax attention in a `3:1` ratio.
- Across `48` layers, that implies `36` DeltaNet layers and `12` full softmax layers.
- The MoE routing surface can still be reconstructed with the same Qwen-style top-k formalism, but most hidden states feeding the router are shaped by DeltaNet's accumulated recurrent state rather than pure full-sequence softmax attention.

## Layout
- `PROMPTS/`: canonical baseline prompt JSON and runtime TSV.
- `RESULTS/`: baseline summaries, logs, metadata, and compact exports.
- `scripts/`: shared 122B builders, launchers, analyzers, and router helpers for the baseline surface.
- `raw/`: pulled compact artifacts for the baseline run.
- `followups/`: completed later 122B runs that should not alter the baseline root.
- `chat_template.txt` and `model-card.txt`: local 122B references.
