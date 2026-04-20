# Plan

Build a clean greedy-decoding reference run for the HauhauCS Qwen3.5-35B-A3B E114 residual-analysis workflow. The goal is to preserve the old stochastic artifacts as historical context while producing a deterministic single-prompt reference and, optionally, a deterministic heldout validation run under the same bare-`</think>` template and L13/L14/L15 capture contract.

`CLAUDE.md` remains the authoritative execution spec. This run folder only scopes the greedy rerun plan and its artifacts.

## Scope

- In: greedy rerun of `single_prompt_processing_hum_no_think.tsv`, Step 1 to Step 3 analysis from that greedy capture, optional greedy rerun of `heldout_prompts.tsv`, provenance files for commands/config/environment/checksums.
- Out: SAE training, new feature discovery, Anthropic API labeler automation, thinking-allowed template claims, changes to `scripts/qwen_router.py`, committing raw tensor dumps or credentials.

## Folder Layout

```text
runs/greedy_reference_20260418T160353Z/
  PLAN.md
  COMMANDS.md
  single_prompt/
    raw/
    analysis/
  heldout/
    raw/
    analysis/
  provenance/
    capture_config.json
    environment.txt
    prompt_checksums.txt
```

## Action Items

[ ] Record prompt checksums in `provenance/prompt_checksums.txt` for `single_prompt_processing_hum_no_think.tsv`, `heldout_prompts.tsv`, and `heldout_classes.tsv`.

[ ] Record the remote runtime details in `provenance/environment.txt`: llama.cpp commit, capture binary sha256, GPU model/count, CUDA version, date, model GGUF path, and model quantization. Do not include hostnames, ports, keys, tokens, or `.env` contents.

[ ] Create `provenance/capture_config.json` with the shared capture settings: Q8_0 HauhauCS model, target layers `[13, 14, 15]`, template `bare_close_think`, trim mode `trim_at_literal_imend`, `-c 2048`, `-ngl 99`, `--tensor-split 1,1`, `--main-gpu 0`, `--no-stream`, `--seed 0`, `--temp 0`, and `--top-k 1`.

[ ] Run the canonical single-prompt greedy capture on the remote GPU box:

```bash
/workspace/residual-analysis/bin/capture_residuals \
  -m /workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf \
  --prompt-file /workspace/residual-analysis/single_prompt_processing_hum_no_think.tsv \
  -o /workspace/residual-analysis/captures/20260418T160353Z_single_prompt_processing_hum_no_think_gen_n1024_greedy \
  -n 1024 -c 2048 -ngl 99 \
  --tensor-split 1,1 --main-gpu 0 \
  --no-stream --seed 0 \
  --temp 0 --top-k 1
```

[ ] Copy the completed single-prompt capture into `runs/greedy_reference_20260418T160353Z/single_prompt/raw/` and verify `capture_manifest.json` reports `1` succeeded prompt, `0` failed prompts, `117` prompt tokens, and `1024` generated tokens.

[ ] Run Step 1 to Step 3 analysis from the greedy single-prompt capture:

```bash
python3 scripts/step1_extract_contexts.py \
  --raw-dir runs/greedy_reference_20260418T160353Z/single_prompt/raw/<greedy_run_id> \
  --analysis-dir runs/greedy_reference_20260418T160353Z/single_prompt/analysis/<greedy_run_id>

python3 scripts/step2_decile_sample.py \
  --analysis-dir runs/greedy_reference_20260418T160353Z/single_prompt/analysis/<greedy_run_id> \
  --samples-per-decile 10 --seed 0

python3 scripts/step3_build_labeler_prompt.py \
  --analysis-dir runs/greedy_reference_20260418T160353Z/single_prompt/analysis/<greedy_run_id>
```

[ ] Validate the single-prompt analysis outputs: confirm HauhauCS literal `<|im_end|>` trim was applied before statistics, confirm `WSQ_identity_residual_max` is at machine epsilon, and inspect `step1/summary.json` for L14 generation-track `W/S/Q`.

[ ] If the validation claim needs deterministic generation, run the greedy heldout capture:

```bash
/workspace/residual-analysis/bin/capture_residuals \
  -m /workspace/models/qwen35-hauhau-q8/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf \
  --prompt-file /workspace/residual-analysis/heldout_prompts.tsv \
  -o /workspace/residual-analysis/captures/heldout_20260418T160353Z_greedy \
  -n 256 -c 2048 -ngl 99 \
  --tensor-split 1,1 --main-gpu 0 \
  --no-stream --seed 0 \
  --temp 0 --top-k 1
```

[ ] Copy the heldout capture into `runs/greedy_reference_20260418T160353Z/heldout/raw/` and analyze it:

```bash
python3 scripts/analyze_heldout.py \
  --raw-dir runs/greedy_reference_20260418T160353Z/heldout/raw/heldout_20260418T160353Z_greedy \
  --classes-tsv heldout_classes.tsv \
  --analysis-dir runs/greedy_reference_20260418T160353Z/heldout/analysis/heldout_20260418T160353Z_greedy
```

[ ] Compare the greedy heldout result to the stochastic `heldout_20260417T202651Z` at the distribution level only: per-class mean-of-means, range overlap, selected top-2 temporal plots, and outlier interpretation. Do not perform token-aligned generation comparison across stochastic and greedy runs.

[ ] Write `COMMANDS.md` after execution with the exact remote commands, copy commands, local analysis commands, manifest checks, and any deviations from this plan.

## Risks And Checks

- The old documented Vast instance may be gone; provision a fresh 2x RTX 5090 box if needed and bootstrap from `scripts/bootstrap_remote_instance.sh`.
- Greedy decoding changes generated text, so generation-track values are a new trajectory, not a token-level continuation of `20260410T042340Z`.
- Prefill remains the deterministic sanity check. If prefill L14 router reconstruction diverges materially from prior same-binary expectations, stop and inspect binary/model/commit before trusting generation metrics.
- Raw tensor captures are large and intentionally not versioned. Keep them local under this run folder or the existing ignored artifact paths, and do not copy secrets or host-specific connection details into provenance.

## Open Questions

- Should the existing local `raw/20260417T183433Z_single_prompt_processing_hum_no_think_gen_n1024_greedy` be treated as a prior greedy attempt, or should the canonical run always be freshly rerun with explicit `--temp 0 --top-k 1` provenance?
- Should the heldout rerun be mandatory for this pass, or only run after the single-prompt greedy labeler input has been reviewed?
