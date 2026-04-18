# Manifest

Included:

- `METHOD/`: `analyze_domain_specialists.py`, `build_domain_specialist_probe_60.py`, `build_domain_specialist_probe_60_think.py`, `qwen_router.py`, `capture_activations.cpp`, `bootstrap_remote_instance.sh`, `run_no_think_bias_family.sh`
- `PROMPTS/`: `domain_specialist_probe_60_no_think.tsv`, `domain_specialist_probe_60_think.tsv`, `domain_specialist_probe_60.json`
- `DOCS/`: `PLAN.md`, `RESULTS.md`
- `generated-text-merged/`: per-condition merged generated text (9 conditions + p8-only, one file each)
- `results-m8-p8/`: analysis outputs for m8-p8 run series
- `results-partial-m8-p5/`: partial analysis outputs for m8-p5 series (JSON + MD; NPZ excluded from git)
- `prompt-results.md`: 3.1 MB compiled transcript (all generated text + results); also archived as `prompt-results.md.zip`
- `non_npy_remote_artifacts/`: index of remote capture locations

Excluded from git (local only):

- `captures/`: timestamped raw capture directories (gitignored)
- `pulled-generated-text-m8-p5.tar`: 2.4 GB tar archive of original per-prompt text tree (9 conditions × 60 prompts)
- `pulled-generated-text-p8-only.tar`: tar archive of per-prompt p8-only text tree (60 prompts)
- `.npy` / `.npz` files: large binary tensors

Reproducibility:

- Yes: reanalysis of generated text in `generated-text-merged/` or `prompt-results.md`
- No: end-to-end rerun from scratch (requires multi-expert suppression capture binary)
