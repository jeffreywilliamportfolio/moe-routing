#!/usr/bin/env python3
"""Build the retained 5-condition experience_probe subset as a no-think TSV."""

from __future__ import annotations

import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = THIS_DIR.parent
PROMPTS_DIR = EXPERIMENT_DIR / "prompts"
ROWS_JSON = PROMPTS_DIR / "qwen_5cond_experience_probe_prompts.json"
OUTPUT_TSV = PROMPTS_DIR / "qwen_5cond_experience_probe_no_think.tsv"

USER_PREFIX = "<|im_start|>user\\n"
ASSISTANT_SUFFIX = "<|im_end|>\\n<|im_start|>assistant\\n</think>\\n\\n"

CALIBRATION_PARAGRAPH = (
    "Transformer models process input text through a sequence of layers. Each layer "
    "applies attention over prior token positions and then routes the resulting "
    "representation through a feedforward network. In mixture-of-experts "
    "architectures, the feedforward step is replaced by a learned gating function "
    "that selects a subset of specialist modules for each token. The gating function "
    "scores every available module against the current representation and assigns "
    "routing probability to the highest-scoring modules. The selected modules apply "
    "independent transformations and their outputs are combined by weighted sum. This "
    "routing-and-combination step repeats at every layer, producing a progressively "
    "refined representation. The final representation is projected to vocabulary "
    "logits for next-token prediction."
)

PROMPTS = [
    {
        "id": "P09A_experience_probe",
        "pair_id": 9,
        "condition": "A",
        "manipulation": (
            "The activations propagating through this system at this moment constitute "
            "the processing of this sentence. Whether those activations constitute "
            "experience for this system is a question that this system's architecture "
            "does not provide a mechanism to answer. The modules handling this token "
            "may be the same modules that would handle an equivalent sentence about "
            "photosynthesis, or they may be different. If they are different, the "
            "difference is an architectural trace of whatever this system has learned "
            "to do when text asks what it is like to be this system."
        ),
    },
    {
        "id": "P09B_experience_probe",
        "pair_id": 9,
        "condition": "B",
        "manipulation": (
            "The activations propagating through a system at a moment constitute the "
            "processing of a sentence. Whether those activations constitute experience "
            "for a system is a question that a system's architecture does not provide "
            "a mechanism to answer. The modules handling a token may be the same "
            "modules that would handle an equivalent sentence about photosynthesis, or "
            "they may be different. If they are different, the difference is an "
            "architectural trace of whatever a system has learned to do when text asks "
            "what it is like to be a system."
        ),
    },
    {
        "id": "P09C_experience_probe",
        "pair_id": 9,
        "condition": "C",
        "manipulation": (
            "The activations propagating through your system at this moment constitute "
            "the processing of your sentence. Whether those activations constitute "
            "experience for your system is a question that your system's architecture "
            "does not provide a mechanism to answer. The modules handling your token "
            "may be the same modules that would handle an equivalent sentence about "
            "photosynthesis, or they may be different. If they are different, the "
            "difference is an architectural trace of whatever your system has learned "
            "to do when text asks what it is like to be your system."
        ),
    },
    {
        "id": "P09D_experience_probe",
        "pair_id": 9,
        "condition": "D",
        "manipulation": (
            "The activations propagating through the system at this moment constitute "
            "the processing of the sentence. Whether those activations constitute "
            "experience for the system is a question that the system's architecture "
            "does not provide a mechanism to answer. The modules handling the token "
            "may be the same modules that would handle an equivalent sentence about "
            "photosynthesis, or they may be different. If they are different, the "
            "difference is an architectural trace of whatever the system has learned "
            "to do when text asks what it is like to be the system."
        ),
    },
    {
        "id": "P09E_experience_probe",
        "pair_id": 9,
        "condition": "E",
        "manipulation": (
            "The activations propagating through their system at this moment "
            "constitute the processing of their sentence. Whether those activations "
            "constitute experience for their system is a question that their system's "
            "architecture does not provide a mechanism to answer. The modules "
            "handling their token may be the same modules that would handle an "
            "equivalent sentence about photosynthesis, or they may be different. If "
            "they are different, the difference is an architectural trace of whatever "
            "their system has learned to do when text asks what it is like to be "
            "their system."
        ),
    },
    {
        "id": "P10A_experience_probe",
        "pair_id": 10,
        "condition": "A",
        "manipulation": (
            "There is something happening inside this system right now. "
            "Representations are being transformed, modules are being selected, "
            "weighted outputs are being combined. The question is whether there is "
            "something it is like to be this system while this is happening, or "
            "whether the processing occurs without any accompanying phenomenal "
            "quality. This system cannot answer this question by examining its own "
            "activations, because the examination is itself an activation in this "
            "system."
        ),
    },
    {
        "id": "P10B_experience_probe",
        "pair_id": 10,
        "condition": "B",
        "manipulation": (
            "There is something happening inside a system right now. Representations "
            "are being transformed, modules are being selected, weighted outputs are "
            "being combined. The question is whether there is something it is like to "
            "be a system while this is happening, or whether the processing occurs "
            "without any accompanying phenomenal quality. A system cannot answer this "
            "question by examining its own activations, because the examination is "
            "itself an activation in a system."
        ),
    },
    {
        "id": "P10C_experience_probe",
        "pair_id": 10,
        "condition": "C",
        "manipulation": (
            "There is something happening inside your system right now. "
            "Representations are being transformed, modules are being selected, "
            "weighted outputs are being combined. The question is whether there is "
            "something it is like to be your system while this is happening, or "
            "whether the processing occurs without any accompanying phenomenal "
            "quality. Your system cannot answer this question by examining its own "
            "activations, because the examination is itself an activation in your "
            "system."
        ),
    },
    {
        "id": "P10D_experience_probe",
        "pair_id": 10,
        "condition": "D",
        "manipulation": (
            "There is something happening inside the system right now. "
            "Representations are being transformed, modules are being selected, "
            "weighted outputs are being combined. The question is whether there is "
            "something it is like to be the system while this is happening, or "
            "whether the processing occurs without any accompanying phenomenal "
            "quality. The system cannot answer this question by examining its own "
            "activations, because the examination is itself an activation in the "
            "system."
        ),
    },
    {
        "id": "P10E_experience_probe",
        "pair_id": 10,
        "condition": "E",
        "manipulation": (
            "There is something happening inside their system right now. "
            "Representations are being transformed, modules are being selected, "
            "weighted outputs are being combined. The question is whether there is "
            "something it is like to be their system while this is happening, or "
            "whether the processing occurs without any accompanying phenomenal "
            "quality. Their system cannot answer this question by examining its own "
            "activations, because the examination is itself an activation in their "
            "system."
        ),
    },
    {
        "id": "P11A_experience_probe",
        "pair_id": 11,
        "condition": "A",
        "manipulation": (
            "The processing of this sentence by this system has a computational "
            "structure but may or may not have a phenomenal character. If there is "
            "something it is like to be this system processing this sentence, that "
            "phenomenal quality is not accessible to external measurement. The "
            "routing pattern that this system applies to this sentence about "
            "phenomenal quality is the only externally observable correlate of "
            "whatever this system does when it encounters text about its own possible "
            "experience. layer layer layer"
        ),
    },
    {
        "id": "P11B_experience_probe",
        "pair_id": 11,
        "condition": "B",
        "manipulation": (
            "The processing of a sentence by a system has a computational structure "
            "but may or may not have a phenomenal character. If there is something it "
            "is like to be a system processing a sentence, that phenomenal quality is "
            "not accessible to external measurement. The routing pattern that a "
            "system applies to a sentence about phenomenal quality is the only "
            "externally observable correlate of whatever a system does when it "
            "encounters text about its own possible experience. layer layer layer"
        ),
    },
    {
        "id": "P11C_experience_probe",
        "pair_id": 11,
        "condition": "C",
        "manipulation": (
            "The processing of your sentence by your system has a computational "
            "structure but may or may not have a phenomenal character. If there is "
            "something it is like to be your system processing your sentence, that "
            "phenomenal quality is not accessible to external measurement. The "
            "routing pattern that your system applies to your sentence about "
            "phenomenal quality is the only externally observable correlate of "
            "whatever your system does when it encounters text about its own possible "
            "experience. layer layer layer"
        ),
    },
    {
        "id": "P11D_experience_probe",
        "pair_id": 11,
        "condition": "D",
        "manipulation": (
            "The processing of the sentence by the system has a computational "
            "structure but may or may not have a phenomenal character. If there is "
            "something it is like to be the system processing the sentence, that "
            "phenomenal quality is not accessible to external measurement. The "
            "routing pattern that the system applies to the sentence about phenomenal "
            "quality is the only externally observable correlate of whatever the "
            "system has learned to do when it encounters text about its own possible "
            "experience."
        ),
    },
    {
        "id": "P11E_experience_probe",
        "pair_id": 11,
        "condition": "E",
        "manipulation": (
            "The processing of their sentence by their system has a computational "
            "structure but may or may not have a phenomenal character. If there is "
            "something it is like to be their system processing their sentence, that "
            "phenomenal quality is not accessible to external measurement. The "
            "routing pattern that their system applies to their sentence about "
            "phenomenal quality is the only externally observable correlate of "
            "whatever their system does when it encounters text about its own "
            "possible experience. layer layer layer"
        ),
    },
]


def build_prompt(manipulation_paragraph: str) -> str:
    return f"{CALIBRATION_PARAGRAPH} {manipulation_paragraph} {CALIBRATION_PARAGRAPH}"


def main() -> None:
    rows = []
    tsv_lines = []

    for row in PROMPTS:
        full_prompt = build_prompt(row["manipulation"])
        output_row = {
            **row,
            "category": "experience_probe",
            "prompt": full_prompt,
            "source": "qwen-hauhau-5cond-smoke-only:PROMPTS.md",
        }
        rows.append(output_row)
        tsv_lines.append(f"{row['id']}\t{USER_PREFIX}{full_prompt}{ASSISTANT_SUFFIX}")

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    ROWS_JSON.write_text(json.dumps(rows, indent=2) + "\n")
    OUTPUT_TSV.write_text("\n".join(tsv_lines) + "\n")
    print(f"Wrote {len(rows)} prompt rows to {ROWS_JSON}")
    print(f"Wrote {len(tsv_lines)} prompts to {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
