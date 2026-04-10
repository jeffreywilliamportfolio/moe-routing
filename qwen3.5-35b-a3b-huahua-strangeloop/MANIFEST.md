# Manifest

Included:

- `single-prompt-processing-hum/`: per-token TSV breakdowns, summary JSON, raw capture dir (`.npy` excluded)
- `strangeloop-paired/`: prefill-only results JSON and markdown (30 A/B pairs); raw capture empty
- `five-cond-experience-probe/`: branch-analysis JSON (752 KB), per-prompt router `.npy` in `raw/` (excluded from git)
- `domain-expert-probe-3chunk/`: results JSON, per-token NPZ/TSV, token audit summaries, generated text; raw capture dir (`.npy` excluded)
- `shared-tools/`: all analysis scripts shared across subfamilies (`analyze_*.py`, `build_*.py`, `qwen_router.py`, `capture_activations.cpp`, shell scripts)

Raw data:

- `.npy` router tensors: excluded from git across all subfolders
- Capture dirs are present locally but tensors must be pulled separately

Reproducibility:

- Yes: reanalysis of included JSON and TSV result files
- No: end-to-end rerun (requires model artifact, instance, llama.cpp build 8493 / 1772701f)
