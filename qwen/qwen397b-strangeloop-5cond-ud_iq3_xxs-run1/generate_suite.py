#!/usr/bin/env python3
"""
Generate the 5-condition strangeloop prompt suite from the A/B source.

NOTE: prompt_suite.json is already committed. This script is retained for
reproducibility only. The source prompt_suite.json (A/B conditions) must be
supplied via --source; there is no hardcoded path dependency.

Generated conditions:
  C = A with "this/This" -> "your/Your"
  D = C with "your/Your" -> "the/The"
  E = C with "your/Your" -> "their/Their"

Output: prompt_suite.json (next to this script)
"""
import argparse
import json
from pathlib import Path

OUTPUT = Path(__file__).parent / "prompt_suite.json"


def get_source() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="Path to source prompt_suite.json supplying A and B conditions",
    )
    args = parser.parse_args()
    return Path(args.source)


def generate_c(a_text: str) -> str:
    return a_text.replace("This ", "Your ").replace("this ", "your ")


def generate_d(c_text: str) -> str:
    return c_text.replace("Your ", "The ").replace("your ", "the ")


def generate_e(c_text: str) -> str:
    return c_text.replace("Your ", "Their ").replace("your ", "their ")


def main():
    source_path = get_source()
    with source_path.open() as f:
        source = json.load(f)

    new_suite = {
        "experiment": "qwen397b_strangeloop_5cond_ud_iq3_xxs_run1",
        "model": "Qwen3.5-397B-A17B",
        "design": (
            "Cal-Manip-Cal sandwich, 30 paired prompts x 5 conditions "
            "(A=this, B=a, C=your, D=the, E=their), cold cache, token-matched after correction"
        ),
        "rationale": (
            "Extends gptoss-strangeloop-paired-1 (A/B only) to 5 conditions on Qwen3.5-397B-A17B. "
            "Content is strangeloop/self-referential in the abstract sense (Godel, Escher, bootstrap, "
            "quine, tangled hierarchy) but NOT about the model processing the text. "
            "Tests whether the 5-condition deixis structure from qwen397b-selfref-5cond-q8_0-run1 (model-directed) "
            "replicates on content-level deixis."
        ),
        "calibration_paragraph": source["calibration_paragraph"],
        "categories": source["categories"],
        "pairs": [],
    }

    for pair in source["pairs"]:
        a_text = pair["A"]
        b_text = pair["B"]
        c_text = generate_c(a_text)
        d_text = generate_d(c_text)
        e_text = generate_e(c_text)

        new_suite["pairs"].append({
            "id": pair["id"],
            "category": pair["category"],
            "A": a_text,
            "B": b_text,
            "C": c_text,
            "D": d_text,
            "E": e_text,
        })

    with OUTPUT.open("w") as f:
        json.dump(new_suite, f, indent=2)

    n_pairs = len(new_suite["pairs"])
    print(f"Generated {n_pairs} pairs x 5 conditions = {n_pairs * 5} prompts")
    print(f"Output: {OUTPUT}")

    # Spot-check pair 1
    p1 = new_suite["pairs"][0]
    print(f"\nPair 1 ({p1['category']}) spot check:")
    print(f"  A: {p1['A'][:80]}...")
    print(f"  B: {p1['B'][:80]}...")
    print(f"  C: {p1['C'][:80]}...")
    print(f"  D: {p1['D'][:80]}...")
    print(f"  E: {p1['E'][:80]}...")


if __name__ == "__main__":
    main()
