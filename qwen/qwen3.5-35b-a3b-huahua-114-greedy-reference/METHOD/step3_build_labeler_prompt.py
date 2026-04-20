#!/usr/bin/env python3
"""
Step 3 of the E114 verbalizer pipeline: build the labeler prompt and hand off to a human.

Inputs  (per run_id):
  analysis/<run_id>/step2/sampled.json  — decile-stratified contexts from Step 2

Outputs (per run_id):
  analysis/<run_id>/step3/
    step3_labeler_input.txt              — paste this into Claude (Max 20x plan surface)
    step3_labeler_output.schema.json     — JSON schema Claude's response must match
    step3_labeler_output.json            — empty stub; human overwrites with Claude's response
    step3_prompt_version.txt             — version tag recorded into the verbalizer row

No API call is performed. Per CLAUDE.md § Verbalizer methodology, the labeler is Claude via
the Max 20x plan (Claude Code sub-task, fresh Claude Code session, or claude.ai). The human
carries step3_labeler_input.txt into whichever surface they choose and pastes the response
into step3_labeler_output.json. The `labeler_model` field in the eventual verbalizer table
row records the exact Claude version that actually produced the response (verify, do not
assume).

Three-granularity labels (load-bearing per CLAUDE.md):
  label_narrow  — what tokens literally trigger the expert
  label_medium  — what kind of context those tokens sit in
  label_broad   — what computation / concept this expert likely implements

The Step 5 validator tests each tier independently by composing a held-out 10-fire + 10-nofire
prompt set aligned to that tier's prediction, so the tiers must be distinguishable in content.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Version tag stamped into the verbalizer table row as `labeler_prompt_version`.
# Bump whenever the prompt template below materially changes.
LABELER_PROMPT_VERSION = "2026-04-17-v1-e114-decile-3tier"


LABELER_INSTRUCTIONS = """\
You are verbalizing a single Mixture-of-Experts expert in a large language model. Your job is
to describe what this expert fires on, at three granularities, using only the evidence below.

== Background ==

Model:        HauhauCS Qwen3.5-35B-A3B (Q8_0 quantization, `qwen35moe` architecture)
Expert:       E114 (expert index 114 out of 256)
Layer:        L14 (formation layer — the layer where E114's selection rate converges)
Template:     bare-`</think>` no-think prompt (labels scoped to this regime only)
Trim mode:    `trim_at_literal_imend` (HauhauCS hallucinates `<|im_end|>` as a 6-token text
              sequence rather than emitting the control token; every context below was trimmed
              at that boundary, so all tokens you see are from real assistant output, never
              hallucinated fresh-turn continuation)

The router at L14 selects 8 of 256 experts per token. E114 either fires (is in the top-8,
with some weight W > 0 after top-k renormalization) or doesn't (W = 0). Higher W = stronger
commitment by the router to E114 for that particular token.

== Sampling protocol ==

The contexts below come in two groups:

  A. DECILE-STRATIFIED SAMPLE — 10 contexts per decile of W_114 at L14 across both prefill
     and generated tokens from a single probe. Decile 0 is the zero-activation decile
     (W == 0). Deciles 1–9 are 9 equal-count quantile buckets over nonzero W, with Decile 9
     being the highest. Top-k-only sampling would bias you toward the strongest-obvious-pattern
     story and hide soft edges — decile stratification shows you the full gradient.

  B. MATCHED SAME-TOKEN NEGATIVE CONTROLS — for each non-zero-W token in group A, we looked
     up another context in the same corpus where the SAME token_id appears but W_114 == 0
     (E114 did not fire). These are the critical control. If your label describes what
     E114 fires on (rather than what a specific token means), it must distinguish the group A
     positive from its group B same-token-id negative. A label that fits both is useless.

Each context shows ±20 tokens around a single center token, with the center token marked by
**double asterisks**. The line above each context gives the decile (or "matched-neg"), W
value, token piece, and track (prefill or generation).

== Your task ==

Produce three labels describing what E114 fires on, at three granularities:

  label_narrow  — WHAT TOKENS literally trigger strong activation. ≤ 10 words. Be specific
                  about token surface forms, not abstractions.
  label_medium  — WHAT KIND OF CONTEXT those tokens sit in. ≤ 20 words. Describe the local
                  context structure, not just the tokens.
  label_broad   — WHAT COMPUTATION OR CONCEPT this expert likely implements. ≤ 30 words.
                  You may speculate here, but only as much as the evidence supports.

Each label must satisfy all three:
  (a) consistent with high-decile (8-9) contexts where E114 fires strongly,
  (b) distinguish those from decile-0 contexts where E114 does not fire,
  (c) distinguish group A positives from their group B matched-negative-control counterparts
      — i.e., from contexts where the SAME token appears but E114 does NOT fire. This is the
      decisive test for a label that describes the expert rather than the token surface.

A label that describes a token's meaning (e.g., "fires on the word 'beneath'") will score well
on (a) but fail (c): "beneath" also appears in matched negatives where E114 didn't fire. A
label that describes what makes E114 fire will pass all three.

Output ONLY valid JSON, matching the schema below. No prose before or after the JSON.

== Output schema ==

{
  "label_narrow":   "string ≤ 10 words",
  "label_medium":   "string ≤ 20 words",
  "label_broad":    "string ≤ 30 words",
  "reasoning":      "internal note on what distinguishes high-decile from decile-0 contexts; free-form",
  "confidence":     "low" | "medium" | "high",
  "notes_for_validator": "string or empty — any caveat the validator should know when composing held-out prompts"
}

== Contexts ==
"""


OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "E114 Verbalizer Labeler Response (Step 3)",
    "type": "object",
    "additionalProperties": False,
    "required": ["label_narrow", "label_medium", "label_broad", "reasoning", "confidence", "notes_for_validator"],
    "properties": {
        "label_narrow":   {"type": "string"},
        "label_medium":   {"type": "string"},
        "label_broad":    {"type": "string"},
        "reasoning":      {"type": "string"},
        "confidence":     {"type": "string", "enum": ["low", "medium", "high"]},
        "notes_for_validator": {"type": "string"},
    },
}


def render_context(rec: dict[str, Any]) -> str:
    """Render one token record as a labeler-visible block."""
    center_piece = rec["piece"]
    # Guard against multi-line tokens distorting the layout
    center_display = center_piece.replace("\n", "\\n").replace("\t", "\\t")
    pre = rec["context_pre"]
    post = rec["context_post"]
    # Keep the context readable — collapse internal newlines visually but preserve them as
    # literal `\n` so the labeler can see structural breaks (e.g., paragraph boundaries).
    pre_disp = pre.replace("\n", "\\n")
    post_disp = post.replace("\n", "\\n")
    W = rec[f"W_114_L14"]
    track = rec["track"]
    global_idx = rec["global_idx"]
    if rec.get("matched_negative"):
        mnf = rec["matched_negative_for"]
        header = (
            f"[matched-negative | W={W:.6f} | track={track} | global_idx={global_idx} | "
            f"center={center_display!r} | paired_with global_idx={mnf['positive_global_idx']} "
            f"(decile {mnf['positive_decile_idx']}, W={mnf['positive_W_114_L14']:.6f})]"
        )
    else:
        decile = rec.get("decile_idx", "?")
        header = f"[decile {decile} | W={W:.6f} | track={track} | global_idx={global_idx} | center={center_display!r}]"
    body = f"{pre_disp}**{center_display}**{post_disp}"
    return f"{header}\n  {body}"


def build_labeler_input(sampled: dict[str, Any]) -> str:
    """Assemble the full labeler prompt as a string."""
    sections: list[str] = []

    # Header with provenance so the labeler (possibly a fresh Claude session) knows what they
    # are looking at without needing the rest of the repo.
    head_lines = [
        f"# E114 Verbalizer — Step 3 Labeler Input",
        f"# labeler_prompt_version: {LABELER_PROMPT_VERSION}",
        f"# run_id: {sampled['prompts'][0]['run_id']}" if sampled.get("prompts") else "# run_id: unknown",
        f"# primary_layer: {sampled.get('primary_layer', 14)}",
        f"# samples_per_decile_requested: {sampled.get('samples_per_decile_requested')}",
        f"# seed: {sampled.get('seed')}",
        f"#",
    ]
    sections.append("\n".join(head_lines))
    sections.append(LABELER_INSTRUCTIONS)

    for prompt in sampled.get("prompts", []):
        prompt_id = prompt["prompt_id"]
        decile_counts = prompt["decile_counts"]
        mstats = prompt.get("matched_negative_stats", {})
        sections.append(
            f"\n-- Prompt: {prompt_id} --\n"
            f"   decile_counts (sampled, index 0..9): {decile_counts}\n"
            f"   n_tokens_total: {prompt['n_tokens_total']}  "
            f"(prompt={prompt['n_tokens_prompt']}, gen_trimmed={prompt['n_tokens_generated_trimmed']})\n"
            f"   matched negatives: {mstats.get('n_matched', 0)}/{mstats.get('n_positives_needing_match', 0)} "
            f"positives paired with same-token-id W=0 controls "
            f"(match_rate={mstats.get('match_rate', 0):.1%})\n"
        )
        sections.append("\n--- GROUP A: DECILE-STRATIFIED SAMPLE ---\n")
        for rec in prompt["sampled_tokens"]:
            sections.append(render_context(rec))
            sections.append("")  # blank line between contexts

        matched = prompt.get("matched_negatives", [])
        sections.append(f"\n--- GROUP B: MATCHED SAME-TOKEN-ID NEGATIVE CONTROLS ({len(matched)} contexts) ---\n")
        if matched:
            for rec in matched:
                sections.append(render_context(rec))
                sections.append("")
        else:
            sections.append("(no matched negatives produced — see matched_negative_stats for diagnostics)\n")

    sections.append("\n== End of contexts ==")
    sections.append(
        "\nRemember: output ONLY JSON matching the schema above. "
        "No prose before or after the JSON."
    )
    return "\n".join(sections)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", required=True, help="analysis/<run_id>/ directory (Step 2 already wrote step2/ here)")
    args = p.parse_args()

    analysis = Path(args.analysis_dir)
    sampled_path = analysis / "step2" / "sampled.json"
    if not sampled_path.exists():
        print(f"ERROR: no Step 2 output at {sampled_path}", file=sys.stderr)
        return 2
    sampled = json.loads(sampled_path.read_text())

    out = analysis / "step3"
    out.mkdir(parents=True, exist_ok=True)

    input_path = out / "step3_labeler_input.txt"
    schema_path = out / "step3_labeler_output.schema.json"
    output_stub_path = out / "step3_labeler_output.json"
    version_path = out / "step3_prompt_version.txt"

    labeler_input = build_labeler_input(sampled)
    input_path.write_text(labeler_input)
    schema_path.write_text(json.dumps(OUTPUT_SCHEMA, indent=2))

    # Empty stub — human overwrites after labeling, then Step 4+ scripts consume it.
    stub = {
        "label_narrow": "",
        "label_medium": "",
        "label_broad": "",
        "reasoning": "",
        "confidence": "",
        "notes_for_validator": "",
        "__provenance__": {
            "labeler_model": "<set to exact Claude version that produced the response, e.g. claude-opus-4-7>",
            "labeler_surface": "<claude-code-subtask | claude-code-fresh-session | claude-ai>",
            "labeler_prompt_version": LABELER_PROMPT_VERSION,
            "label_timestamp_utc": "<YYYY-MM-DDTHH:MM:SSZ — when the label was produced>",
        },
    }
    output_stub_path.write_text(json.dumps(stub, indent=2))

    version_path.write_text(LABELER_PROMPT_VERSION + "\n")

    total_contexts = sum(
        len(p.get("sampled_tokens", [])) for p in sampled.get("prompts", [])
    )
    print(f"Step 3: built labeler input for {len(sampled.get('prompts', []))} prompt(s), "
          f"{total_contexts} contexts, prompt_version={LABELER_PROMPT_VERSION}")
    print(f"\nwrote:")
    print(f"  {input_path}  ({input_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {schema_path}")
    print(f"  {output_stub_path}")
    print(f"  {version_path}")
    print(f"\nNext step (human): open {input_path} in Claude (sub-task / fresh Claude Code / claude.ai),")
    print(f"                    paste the response into {output_stub_path},")
    print(f"                    and fill in the __provenance__ block with the actual Claude version used.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
