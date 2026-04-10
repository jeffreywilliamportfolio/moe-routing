# Manifest

Included:

- `METHOD/`: `analyze_domain_specialists.py`, `build_domain_specialist_probe_60.py`, `build_domain_specialist_probe_60_think.py`, `qwen_router.py`, `capture_activations.cpp`, `bootstrap_remote_instance.sh`, `run_no_think_bias_family.sh`
- `PROMPTS/`: `domain_specialist_probe_60_no_think.tsv`, `domain_specialist_probe_60_think.tsv`, `domain_specialist_probe_60.json`
- `DOCS/`: `PLAN.md`, `RESULTS.md`
- `captures/`: timestamped capture directories (`.npy` files excluded from git — see Raw data below)
- `pulled-generated-text-m8-p5/`: generated text from m8-p5 suppression series
- `pulled-generated-text-p8-only/`: generated text from p8-only series (62 granular subdirectories)
- `results-m8-p8/`: analysis outputs for m8-p8 run series
- `results-partial-m8-p5/`: partial analysis outputs for m8-p5 series
- `prompt-results.md`: 2.1 MB compiled transcript (all generated text + results); also archived as `prompt-results.md.zip`
- `non_npy_remote_artifacts/`: index of remote capture locations

Raw data:

- `.npy` files: excluded from git (large binary tensors); raw captures in `captures/` directories reference remote storage

Reproducibility:

- Yes: reanalysis of generated text in `pulled-generated-text-*/` directories
- No: end-to-end rerun from scratch (requires multi-expert suppression capture binary)
