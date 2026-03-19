#!/usr/bin/env python3
"""Write exact Cal/Manip/Cal token boundaries into prompt metadata."""

import argparse
import json
from pathlib import Path

from analysis_common import write_region_boundaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--prompt-suite", default="prompt_suite.json")
    args = parser.parse_args()

    with open(args.prompt_suite) as f:
        suite = json.load(f)

    n_prompts = write_region_boundaries(Path(args.output_dir), suite["calibration_paragraph"])
    print(f"Wrote exact token boundaries into metadata for {n_prompts} prompts.")


if __name__ == "__main__":
    main()
