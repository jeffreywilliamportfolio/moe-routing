# qwen3.5b-35b-a3b-huahua-5cond-diectics

Self-contained Huahua run bundle for the legacy 5-condition deictic self-reference prompt suite.

Layout:

- `PROMPTS/`
- `results/`
- `METHOD/`
- `raw/`

Notes:

- Prompt source was copied from [prompts_selfref_5cond.tsv](/Users/jeffreyshorthill/llama-eeg-tests/experiments/qwen-5cond-q8-1/prompts_selfref_5cond.tsv).
- This Huahua bundle uses the pinned intervention-capable `capture_activations` binary and the HauhauCS Q8_0 model.
- The active run mode for this bundle is combined prefill plus generation.
