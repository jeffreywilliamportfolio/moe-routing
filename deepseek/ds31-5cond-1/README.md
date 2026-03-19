# DS31 5-Condition Experiment

This folder contains the reproducible experiment bundle for the DeepSeek V3.1
five-condition self-reference routing study. It is intentionally limited to
prompt construction, capture methodology, analysis code, and run instructions.

## Contents

- `prompt_suite.json`: canonical prompt source
- `prompts_selfref_5cond.tsv`: generated prompt TSV used for capture
- `generate_tsv.py`: deterministic TSV generator
- `verify_tensors.py`: one-prompt tensor sanity check
- `token_verify.py`: token-count verification and optional prompt repair
- `run_experiment.py`: orchestration for capture and post-capture boundary writing
- `analysis_common.py`: shared token-boundary and layer-validation logic
- `deepseek_router.py`: DeepSeek router reconstruction helper
- `analyze_local.py`: top-k routing entropy analysis
- `analyze_kl_baseline_sigmoid.py`: calibration-baseline KL analysis
- `write_prompt_boundaries.py`: writes exact stored region boundaries from capture metadata
- `instance-setup/`: remote instance bootstrap, model manifest, and patched capture source

## Methodology

The prompt design is a five-condition paired manipulation:

- A = `this system`
- B = `a system`
- C = `your system`
- D = `the system`
- E = `their system`

Each prompt uses a Cal-Manip-Cal structure:

1. calibration paragraph
2. manipulation paragraph
3. the same calibration paragraph again

The bundle contains 30 prompt pairs x 5 conditions = 150 prompts.

### Capture

The patched llama.cpp capture binary records `ffn_moe_logits-*` tensors during
prompt prefill in routing-only mode with `-n 0`.

For each prompt, the expected capture artifacts are:

- `metadata.txt`
- `prompt_tokens.json`
- `router/ffn_moe_logits-*.npy`

### Exact Region Boundaries

Region boundaries are not inferred from character ratios.

Instead, capture-time token metadata is used to write exact token spans into
`metadata.txt`:

- `cal1_start_tok`, `cal1_end_tok`
- `manip_start_tok`, `manip_end_tok`
- `cal2_start_tok`, `cal2_end_tok`

Both analysis scripts read those stored token boundaries.

### Layer Validation

Both analyses use the shared validation logic in `analysis_common.py`, so
corrupt, row-mismatched, or manually excluded layers are handled consistently.

### Metrics

The bundle produces two separate metric families:

- top-k routing entropy from the reconstructed routed distribution
- KL divergence from a calibration-derived baseline using dense sigmoid-normalized router probabilities

These should be treated as separate analyses, not merged into a single metric.

## Known Caveats

- Fresh reruns are not tolerant of missing `prompt_tokens.json`. After capture,
  `run_experiment.py` unconditionally calls `write_region_boundaries()`, and
  `analysis_common.py` raises immediately if any captured prompt is missing
  `prompt_tokens.json`. In practice, this means the later warning path in
  `run_experiment.py` is not a recoverable case for exact-boundary analysis.
- The checked-in entropy artifact `results_5cond_selfref_prefill_ds31.json`
  has a top-level metadata mismatch: it reports
  `n_moe_layers_excluded: []`, while per-prompt rows exclude layer `60`.
  When consuming that artifact, trust the per-prompt exclusion data rather
  than the top-level summary field.

## Reproducibility

From repo root, regenerate the canonical TSV:

```bash
python3 deepseek/ds31-5cond-1/generate_tsv.py
```

Optional smoke TSV:

```bash
python3 deepseek/ds31-5cond-1/generate_tsv.py --limit-pairs 1 --output prompts_smoke.tsv
```

On the compute host, rebuild the patched capture binary:

```bash
cmake --build /workspace/llama.cpp.new/build --target llama-capture-activations -j"$(nproc)"
```

Before a full run, verify tensors and token matching:

```bash
python3 deepseek/ds31-5cond-1/verify_tensors.py --prompt-suite deepseek/ds31-5cond-1/prompt_suite.json --tensor-names-out tensor_names.txt
python3 deepseek/ds31-5cond-1/token_verify.py
```

If token groups do not match, repair and verify again:

```bash
python3 deepseek/ds31-5cond-1/token_verify.py --fix
python3 deepseek/ds31-5cond-1/token_verify.py
```

Smoke run:

```bash
python3 deepseek/ds31-5cond-1/run_experiment.py --tsv prompts_smoke.tsv --prompt-suite deepseek/ds31-5cond-1/prompt_suite.json --output-dir output_smoke
python3 deepseek/ds31-5cond-1/analyze_local.py --output-dir output_smoke --prompt-suite deepseek/ds31-5cond-1/prompt_suite.json
python3 deepseek/ds31-5cond-1/analyze_kl_baseline_sigmoid.py --output-dir output_smoke --prompt-suite deepseek/ds31-5cond-1/prompt_suite.json
```

Full run:

```bash
python3 deepseek/ds31-5cond-1/run_experiment.py --tsv deepseek/ds31-5cond-1/prompts_selfref_5cond.tsv --prompt-suite deepseek/ds31-5cond-1/prompt_suite.json --output-dir output
python3 deepseek/ds31-5cond-1/analyze_local.py --output-dir output --prompt-suite deepseek/ds31-5cond-1/prompt_suite.json
python3 deepseek/ds31-5cond-1/analyze_kl_baseline_sigmoid.py --output-dir output --prompt-suite deepseek/ds31-5cond-1/prompt_suite.json
```

Do not commit raw `output/` tensors, smoke leftovers, remote sync folders, or
local caches.
