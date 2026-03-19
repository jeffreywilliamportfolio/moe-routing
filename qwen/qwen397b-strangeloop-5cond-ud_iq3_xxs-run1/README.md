# Qwen 3.5 397B Strange-Loop 5-Condition Run

## What This Folder Is

This folder contains the reviewed Qwen3.5-397B-A17B strange-loop five-condition routing experiment.

- Model: `Qwen3.5-397B-A17B-UD-IQ3_XXS`
- Quantization: `UD-IQ3_XXS`
- Design: 30 prompt groups x 5 determiner conditions = 150 prompts
- Conditions: `A=this`, `B=a`, `C=your`, `D=the`, `E=their`
- Content: abstract strange-loop topics such as Godel, Escher, bootstrap, quine, and tangled hierarchy
- Inference mode: prefill-only (`n_predict=0`), cold KV cache, greedy argmax
- Routing reconstruction: `softmax(512) -> topk(10) -> renormalize`
- Entropy normalization: `log2(10)`
- Architecture assumptions: 60 captured MoE layers with layer `59` excluded from analysis

## Artifact Status

The checked-in result bundle looks usable and internally consistent at the high-impact level.

- `results_strangeloop_5cond_prefill_qwen.json` contains 150 per-prompt rows
- All 30 five-condition prompt groups are complete
- `token_mismatch_pairs` is empty in the checked-in artifact
- Every analyzed prompt uses 59 valid MoE layers with excluded layer set `(59,)`

## Important Caveat

The checked-in artifact looks complete. Fresh reruns are not fully fail-closed.

`run_experiment.py` writes the canonical JSON incrementally and does not hard-assert final prompt coverage before printing `DONE`. An interrupted rerun can therefore leave behind a partial result file that looks finished.

For any rerun, manually verify at minimum:

- the final JSON has 150 per-prompt rows and 30 complete `ABCDE` groups
- `token_mismatch_pairs` is empty unless prompt construction was intentionally changed
- `experiment.log` shows full prompt coverage rather than a partial analysis

## Main Files

- `results_strangeloop_5cond_prefill_qwen.json`: reviewed result bundle with per-prompt metrics and pairwise tests
- `RESULTS.md`: concise narrative summary of the main result
- `NOTES.md`: supplemental setup and run notes
- `run_experiment.py`: capture plus analysis runner
- `analyze_5cond.py`: local paired analysis across all five conditions
- `generate_suite.py`: expands the source paired strangeloop prompts to five conditions
- `qwen_router.py`: Qwen routing reconstruction logic
- `prompt_suite.json`: prompt definitions
- `prompts_strangeloop_5cond.tsv`: capture input TSV

## Running It

The runner expects a compiled `capture_activations` binary and a local Qwen GGUF model. The exact paths can be adjusted in `run_experiment.py` or the helper scripts in this folder.

Typical flow:

```bash
python3 generate_suite.py
python3 generate_tsv.py
python3 token_verify.py
python3 run_experiment.py
```

## Notes On Interpretation

`RESULTS.md` is the shortest reader-facing summary of the observed effect. In this run, `this` reliably raises all-token routing entropy above `a`, while `your` shows the lowest routing entropy and the largest KL shift from calibration.
