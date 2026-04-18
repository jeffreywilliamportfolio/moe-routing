# Plan

The baseline build-out is complete. This file now tracks the organization rules for the clean 122B branch surface.

## Status
- [x] Bootstrap the pinned llama.cpp build for the 122B `Q8_K_P` path.
- [x] Build the canonical 150-prompt 5-condition diectic runtime TSV.
- [x] Run the large 122B baseline capture and verify the result artifacts.
- [x] Re-run analysis from raw for certainty on the baseline surface.
- [x] Separate later 122B follow-up runs from the baseline root.

## Root Rules
- Keep the root surface limited to the canonical diectic baseline and the shared 122B tooling needed to explain or rerun it.
- Put later completed runs under `followups/`.
- Do not add new ad hoc run artifacts directly to the root `PROMPTS/`, `RESULTS/`, or `raw/` directories unless they belong to the canonical baseline.

## Follow-Up Rules
- Each follow-up run under `followups/` should stay self-contained with its own prompts, results, scripts, and raw compact artifacts.
- The follow-up runs may reuse the shared 122B renderer and router assumptions, but they should not overwrite or redefine the baseline root surfaces.

## Reporting Rules
- Use `W/S/Q` as the primary analysis surface.
- Treat entropy and other legacy metrics as supporting evidence only.
- Keep the root `RESULTS.md` and `INTERPRETATION.md` baseline-only.
- Put run-specific reports for later 122B work inside the relevant follow-up bundle.
