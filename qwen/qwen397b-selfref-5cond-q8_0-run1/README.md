# Qwen 3.5 397B Self-Referential 5-Condition Run

## What This Folder Is

This folder contains the reviewed Qwen3.5-397B-A17B self-referential five-condition routing experiment.

- Model: `Qwen3.5-397B-A17B-Q8_0`
- Quantization: `Q8_0`
- Design: 30 prompt groups x 5 determiner conditions = 150 prompts
- Conditions: `A=this system`, `B=a system`, `C=your system`, `D=the system`, `E=their system`
- Inference mode: prefill-only (`n_predict=0`), cold KV cache, greedy argmax
- Routing reconstruction: `softmax(512) -> topk(10) -> renormalize`
- Entropy normalization: `log2(10)`
- Architecture assumptions: 60 captured MoE layers with layer `59` excluded from analysis

## Artifact Status

The checked-in result bundle looks usable and internally consistent at the high-impact level.

- `results_selfref_5cond_prefill_qwen.json` contains 150 per-prompt rows
- All 30 five-condition prompt groups are complete
- `token_mismatch_pairs` is empty in the final artifact
- Every analyzed prompt uses 59 valid MoE layers with excluded layer set `(59,)`

## Important Caveat

The checked-in artifact looks complete. A fresh rerun is still not fully fail-closed.

`run_experiment.py` writes the canonical JSON during the run via `save_partial_results()`, before the final 150-prompt completeness check. If a rerun is interrupted, it can leave behind a normal-looking partial result file at the final output path.

For any rerun, manually verify at minimum:

- the final JSON has 150 per-prompt rows and 30 complete `ABCDE` groups
- `token_mismatch_pairs` is empty unless prompt construction was intentionally changed
- `experiment.log` ends with successful completion for all 150 prompts

## Main Files

- `results_selfref_5cond_prefill_qwen.json`: reviewed result bundle with per-prompt metrics and pairwise tests
- `RESULTS-QWEN397.md`: concise narrative summary of the main result
- `RESULTS-results_selfref_5cond_prefill_qwen.md`: validation-focused summary of the checked-in artifact
- `run_experiment.py`: capture plus analysis runner
- `analyze_5cond.py`: local paired analysis across all five conditions
- `qwen_router.py`: Qwen routing reconstruction logic
- `prompt_suite.json`: prompt definitions
- `prompts_selfref_5cond.tsv`: capture input TSV

## Running It

The runner expects a compiled `capture_activations` binary and a local Qwen GGUF model. The exact paths can be adjusted in `run_experiment.py` or the helper scripts in this folder.

Typical flow:

```bash
python3 generate_tsv.py
python3 run_experiment.py
```

## Notes On Interpretation

`RESULTS-QWEN397.md` is the shortest reader-facing summary of the observed effect. The main pattern in this run is that `your system` produces the lowest mean routing entropy and the largest KL shift from calibration in the manipulation region.
