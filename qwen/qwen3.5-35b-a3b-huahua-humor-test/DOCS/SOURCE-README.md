# qwen3.5b-35b-a3b-huahua-5cond-diectics

Self-contained Huahua run bundle for the legacy 5-condition deictic self-reference prompt suite.

Layout:

- `PROMPTS/`
- `results/`
- `METHOD/`
- `raw/`

Notes:

- Prompt source was copied from local source file `prompts_selfref_5cond.tsv` in the original `qwen-5cond-q8-1` experiment workspace.
- This Huahua bundle uses the pinned intervention-capable `capture_activations` binary and the HauhauCS Q8_0 model.
- The active run mode for this bundle is combined prefill plus generation.
