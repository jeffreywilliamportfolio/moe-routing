# Manifest

Included:

- `METHOD/`: `capture_residuals.cpp` (residual + router tap at L13/L14/L15; pinned to `llama.cpp@1772701f9`), `bootstrap_remote_instance.sh` (Vast.ai provisioning for CUDA 12.9 / Blackwell with `--break-system-packages` for Ubuntu 24.04+ PEP 668), `qwen_router.py` (canonical `softmax → top-k=8 → renorm`; `W = S × Q` identity at machine epsilon), `analyze_heldout.py` (per-prompt W/S/Q + top-4 timeseries plot).
- `PROMPTS/`: `heldout_prompts.tsv` (2-col `prompt_id` + prompt with `\n` as literal newline escapes, direct input for `capture_residuals --prompt-file`), `heldout_classes.tsv` (sidecar `prompt_id\tpredicted_class`, separate from the prompts file because `capture_residuals` splits on the first tab only).
- `DOCS/`: `PLAN.md`, `RESULTS.md`.
- `results/`: `heldout_stats.tsv` (per-prompt mean/stddev of W₁₁₄ at L14 on trimmed generation track), `heldout_timeseries_top4.png` (per-token W₁₁₄ series for top-2 fire + top-2 nofire by within-class mean).
- `raw/20260417T202651Z_heldout/`: one subdir per prompt (`F01`…`F10`, `N01`…`N10`) containing `metadata.txt`, `prompt_tokens.json`, `generated_tokens.json`, `generated_text.txt`. Plus run-level `capture_manifest.json` and `capture.log`.

Excluded from git (regenerable from the capture binary + pinned commit + prompt TSV):

- `*.npy` router + residual tensors under `raw/20260417T202651Z_heldout/<prompt>/{router,residual}/`. 6 tensors per prompt × 20 prompts ≈ 160 MB on disk.
- `captures/`, `analysis/` pipeline intermediates (not used in the pilot-mode workflow for this experiment).

Reproducibility:

- Yes: re-analysis of the included `heldout_stats.tsv` and per-prompt `generated_text.txt` / `generated_tokens.json` / `metadata.txt`.
- Yes: re-run `METHOD/analyze_heldout.py` against a freshly captured `raw/<run_id>/` (needs the `.npy` tensors, which requires a re-capture).
- No: end-to-end rerun without a Vast.ai instance + HuggingFace access to `HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive`.
