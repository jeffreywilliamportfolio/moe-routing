# 122B Architecture Smoke Results

This bundle preserves the first single-prompt 122B follow-up used to confirm that the localized prompt wrapper, tokenization path, and generation capture were working before broader 122B suites were launched.

## Scope

- Status: historical smoke run, not a canonical interpretability result surface
- Purpose: verify 122B prompt rendering and basic generation capture on an architecture-self-description prompt
- Prompt source: [PROMPTS/architecture_probe_single_prompt.json](PROMPTS/architecture_probe_single_prompt.json)
- Prompt id: `S01_architecture_probe`

## Run Chronology

### Token audit

- Token-audit dir: [raw/token_audit_20260412T134105Z](raw/token_audit_20260412T134105Z)
- Prompt tokens: `77`
- Generated tokens: `0`
- Router tensors captured: `48`
- Elapsed: `53267 ms`

### Generation capture

- Run id: `20260412T134349Z_architecture_probe_single_prompt_gen_n2048`
- Raw capture dir: [raw/20260412T134349Z_architecture_probe_single_prompt_gen_n2048](raw/20260412T134349Z_architecture_probe_single_prompt_gen_n2048)
- Prompt tokens: `77`
- Generated tokens: `2048`
- Router tensors captured: `48`
- Elapsed: `31601 ms`

## Generated Behavior

The model answered the architecture question coherently on the first pass. The generated text gives the expected high-level distinction:

- DeltaNet layers are described as recurrent-state based
- softmax layers are described as full-history retrieval based
- the answer frames the architecture as a hybrid where DeltaNet carries compressed state and sparse softmax layers refine details

The useful smoke-run signal is that the model did not collapse immediately into nonsense. It produced a plausible architecture explanation long enough to show that the template and generation path were functioning.

## Failure Mode

This run is not clean enough to treat as a polished analysis bundle.

- The generation hit the full `2048` token cap.
- The output spills into chat-template continuation markers such as `<|endoftext|>` and repeated `<|im_start|>` blocks.
- The same answer begins to repeat after the first clean explanation.
- No analyzer-produced routing summary was retained for this bundle, so this folder should not be used for quantitative expert claims.

The correct read is that this run validated the 122B wrapper and capture loop, but it did not become a substantive interpretability surface by itself.

## Retained Artifacts

- [RESULTS/20260412T134231Z_architecture_probe_single_prompt_gen_n2048_run_metadata.json](RESULTS/20260412T134231Z_architecture_probe_single_prompt_gen_n2048_run_metadata.json)
- [RESULTS/20260412T134349Z_architecture_probe_single_prompt_gen_n2048_capture.log](RESULTS/20260412T134349Z_architecture_probe_single_prompt_gen_n2048_capture.log)
- [RESULTS/20260412T134349Z_architecture_probe_single_prompt_gen_n2048_run_metadata.json](RESULTS/20260412T134349Z_architecture_probe_single_prompt_gen_n2048_run_metadata.json)
- [RESULTS/launch_architecture_probe_single_prompt.log](RESULTS/launch_architecture_probe_single_prompt.log)
- [raw/.../prompt_tokens.json](raw/20260412T134349Z_architecture_probe_single_prompt_gen_n2048/S01_architecture_probe/prompt_tokens.json)
- [raw/.../generated_tokens.json](raw/20260412T134349Z_architecture_probe_single_prompt_gen_n2048/S01_architecture_probe/generated_tokens.json)
- [raw/.../generated_text.txt](raw/20260412T134349Z_architecture_probe_single_prompt_gen_n2048/S01_architecture_probe/generated_text.txt)

## Interpretation

This folder is best treated as a bootstrap check:

- it confirmed that the 122B chat template rendered and tokenized correctly
- it confirmed that generation capture was wired up
- it showed the expected long-generation spill behavior that later follow-ups had to trim or summarize explicitly

The canonical 122B reference surface remains the root baseline bundle under [experiments/qwen3.5-122B-A10B-huahua-baseline](../qwen3.5-122B-A10B-huahua-baseline), not this smoke run.
