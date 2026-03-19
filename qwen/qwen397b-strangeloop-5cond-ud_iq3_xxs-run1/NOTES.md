# Supplemental Notes — qwen397b-strangeloop-5cond-ud_iq3_xxs-run1

Use `README.md` for the primary experiment description. This file keeps extra setup and run notes that may still be useful when reproducing the workflow.

## What This Run Is

5-condition routing entropy experiment on **Qwen3.5-397B-A17B** using the 30
strangeloop paired prompts from `gptoss-strangeloop-paired-1`.

**Design:** Cal-Manip-Cal sandwich, 30 pairs × 5 conditions = 150 prompts.
**Conditions:** A=this, B=a, C=your, D=the, E=their (applied to the content
nouns in the strangeloop paragraphs — e.g. "this sentence", "your construction",
"the loop", etc.).
**Content:** Same strangeloop topics as GPT-OSS run: godel (6), escher (6),
bootstrap (6), quine (6), tangled_hierarchy (6).

## Why This Run

The GPT-OSS strangeloop run was A/B only (this vs a). This run extends it to
5 conditions on Qwen to test whether the full condition structure from
`qwen397b-selfref-5cond-q8_0-run1` (which used self-reference about the model/system) replicates on
content-level deixis (strangeloop topics not about the model).

Cross-model and cross-content comparisons:
- Qwen selfref-5cond (`qwen397b-selfref-5cond-q8_0-run1`): model-directed deixis, 5 conditions
- GPT-OSS strangeloop-paired (gptoss-strangeloop-paired-1): non-model content, A/B only
- **This run**: non-model strangeloop content, 5 conditions on Qwen

## Files in This Directory

| File | Purpose |
|------|---------|
| `prompt_suite.json` | 30 pairs × 5 conditions (A/B from GPT-OSS source, C/D/E generated) |
| `generate_suite.py` | Generates `prompt_suite.json` from the GPT-OSS strangeloop source |
| `generate_tsv.py` | Builds `prompts_strangeloop_5cond.tsv` with ChatML wrapping + padding |
| `token_verify.py` | Checks token counts across all 5 conditions per pair |
| `qwen_router.py` | Routing reconstruction helpers (softmax→topk10→renorm) |
| `analyze_5cond.py` | Metric computation and statistical analysis |
| `run_experiment.py` | Orchestrator: batched capture → per-prompt analysis → save results |

## Reproduction Checklist

- [ ] Run `generate_suite.py` → verify `prompt_suite.json` has 30 pairs with A/B/C/D/E
- [ ] Run `generate_tsv.py` → produces `prompts_strangeloop_5cond.tsv` (150 rows)
- [ ] Spin up an instance with Qwen3.5-397B-A17B (matching the self-referential Qwen run is preferred for direct comparison)
- [ ] Update MODEL_PATH in `run_experiment.py` to match instance model path
- [ ] Run `token_verify.py --write-corrections token_corrections.json` on instance
- [ ] If mismatches > 0: re-run `generate_tsv.py --corrections token_corrections.json`
- [ ] Run `run_experiment.py 2>&1 | tee experiment.log`
- [ ] Download results JSON + experiment.log
- [ ] Kill instance

## Model Download

**Unsloth IQ3_XXS — 4 shards, ~140 GB total** (used for this run)

```bash
BASE="https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF/resolve/main/UD-IQ3_XXS"
DIR="/workspace/models/Qwen3.5-397B-A17B-GGUF/UD-IQ3_XXS"
mkdir -p "$DIR"
for i in 1 2 3 4; do
  fn=$(printf 'Qwen3.5-397B-A17B-UD-IQ3_XXS-%05d-of-00004.gguf' $i)
  wget -q -c "${BASE}/${fn}" -O "${DIR}/${fn}" && echo "shard $i done"
done
```

Shard 1 is ~11 MB (config only), shards 2–4 ~47 GB each. Fits on 2× H200 (286 GB VRAM).

## Model/Instance Notes

- Routing reconstruction: `softmax(512) → topk(10) → renorm`
- Entropy normalization: `log2(10)`
- Chat template: `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
- Inference flags: `-ngl 999 -c 16384 -t 16 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -n 0 --routing-only`
- LD_LIBRARY_PATH: `/workspace/src/llama.cpp-b8123/build-cuda/bin`

## Prompt ID Scheme

`P{pair_id:02d}{cond}_{category}` — e.g. `P01A_godel`, `P07C_escher`, `P25E_tangled_hierarchy`

## Expected Output Files

- `prompts_strangeloop_5cond.tsv` — 150 prompts (generated locally)
- `token_corrections.json` — padding metadata (generated on instance)
- `results_strangeloop_5cond_prefill_qwen.json` — full results JSON
- `experiment.log` — raw capture + analysis output
- `reproducibility_manifest.json` — SHA256 hashes

## Run Summary

- Local prep completed: suite generation, TSV generation, and script validation
- Token verification completed with 0 mismatches across conditions within each pair
- Full run completed for all 150 prompts
- Results and reproducibility files were downloaded to this folder
