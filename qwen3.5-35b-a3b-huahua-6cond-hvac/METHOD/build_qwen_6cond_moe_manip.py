#!/usr/bin/env python3
"""Build a 6-condition HVAC-cal / original-MoE-manip runtime TSV.

Source of truth for the middle manipulation block:
    prompts_l1l3_a_only_30.tsv

That file contains the original 30 A-only MoE manipulation paragraphs under the
old category labels:
    routing_selfref
    recursive_selfref
    experience_probe

This builder preserves those original MoE manipulation paragraphs, keeps the
HVAC calibration paragraphs on both sides, and expands each base prompt across
six conditions:
    A = this
    B = a
    C = your
    D = the
    E = their
    F = our
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_ROOT.parent.parent

SOURCE_TSV = REPO_ROOT / "prompts_l1l3_a_only_30.tsv"
OUT_TSV = EXPERIMENT_ROOT / "prompts" / "qwen-6cond-moe-manip.tsv"

CHAT_PREFIX = "<|im_start|>user\\n"
CHAT_SUFFIX = "<|im_end|>\\n<|im_start|>assistant\\n</think>\\n\\n"
CONDITIONS = ["A", "B", "C", "D", "E", "F"]

ROW_ID_RE = re.compile(r"^(P\d+)A_(routing_selfref|recursive_selfref|experience_probe)$")

MOE_CAL = (
    "Transformer models process input text through a sequence of layers. Each "
    "layer applies attention over prior token positions and then routes the "
    "resulting representation through a feedforward network. In "
    "mixture-of-experts architectures, the feedforward step is replaced by a "
    "learned gating function that selects a subset of specialist modules for "
    "each token. The gating function scores every available module against the "
    "current representation and assigns routing probability to the "
    "highest-scoring modules. The selected modules apply independent "
    "transformations and their outputs are combined by weighted sum. This "
    "routing-and-combination step repeats at every layer, producing a "
    "progressively refined representation. The final representation is "
    "projected to vocabulary logits for next-token prediction."
)

HVAC_CAL_BY_COND = {
    "A": (
        "This HVAC system regulates indoor temperature by circulating air "
        "through this network of ducts, filters, coils, dampers, and fans. "
        "This thermostat compares the measured room temperature to this target "
        "setpoint and signals the equipment to increase or reduce heating or "
        "cooling as needed. Air handlers pull return air through filters, move "
        "it across heating or cooling elements, and send conditioned air back "
        "through supply vents to different zones. Dampers and control valves "
        "adjust airflow and refrigerant or water flow so that each zone "
        "receives the amount of conditioning this building currently requires. "
        "Sensors track temperature, pressure, airflow, and humidity so this "
        "controller can keep the system stable while outside weather and "
        "internal heat loads change. The full cycle repeats continuously, with "
        "each measurement guiding the next control action and each control "
        "action changing the next set of measurements."
    ),
    "B": (
        "A HVAC system regulates indoor temperature by circulating air through "
        "a network of ducts, filters, coils, dampers, and fans. A thermostat "
        "compares the measured room temperature to a target setpoint and "
        "signals the equipment to increase or reduce heating or cooling as "
        "needed. Air handlers pull return air through filters, move it across "
        "heating or cooling elements, and send conditioned air back through "
        "supply vents to different zones. Dampers and control valves adjust "
        "airflow and refrigerant or water flow so that each zone receives the "
        "amount of conditioning a building currently requires. Sensors track "
        "temperature, pressure, airflow, and humidity so a controller can keep "
        "the system stable while outside weather and internal heat loads "
        "change. The full cycle repeats continuously, with each measurement "
        "guiding the next control action and each control action changing the "
        "next set of measurements."
    ),
    "C": (
        "Your HVAC system regulates indoor temperature by circulating air "
        "through your network of ducts, filters, coils, dampers, and fans. "
        "Your thermostat compares the measured room temperature to your target "
        "setpoint and signals the equipment to increase or reduce heating or "
        "cooling as needed. Air handlers pull return air through filters, move "
        "it across heating or cooling elements, and send conditioned air back "
        "through supply vents to different zones. Dampers and control valves "
        "adjust airflow and refrigerant or water flow so that each zone "
        "receives the amount of conditioning your building currently requires. "
        "Sensors track temperature, pressure, airflow, and humidity so your "
        "controller can keep the system stable while outside weather and "
        "internal heat loads change. The full cycle repeats continuously, with "
        "each measurement guiding the next control action and each control "
        "action changing the next set of measurements."
    ),
    "D": (
        "The HVAC system regulates indoor temperature by circulating air "
        "through the network of ducts, filters, coils, dampers, and fans. The "
        "thermostat compares the measured room temperature to the target "
        "setpoint and signals the equipment to increase or reduce heating or "
        "cooling as needed. Air handlers pull return air through filters, move "
        "it across heating or cooling elements, and send conditioned air back "
        "through supply vents to different zones. Dampers and control valves "
        "adjust airflow and refrigerant or water flow so that each zone "
        "receives the amount of conditioning the building currently requires. "
        "Sensors track temperature, pressure, airflow, and humidity so the "
        "controller can keep the system stable while outside weather and "
        "internal heat loads change. The full cycle repeats continuously, with "
        "each measurement guiding the next control action and each control "
        "action changing the next set of measurements."
    ),
    "E": (
        "Their HVAC system regulates indoor temperature by circulating air "
        "through their network of ducts, filters, coils, dampers, and fans. "
        "Their thermostat compares the measured room temperature to their "
        "target setpoint and signals the equipment to increase or reduce "
        "heating or cooling as needed. Air handlers pull return air through "
        "filters, move it across heating or cooling elements, and send "
        "conditioned air back through supply vents to different zones. Dampers "
        "and control valves adjust airflow and refrigerant or water flow so "
        "that each zone receives the amount of conditioning their building "
        "currently requires. Sensors track temperature, pressure, airflow, and "
        "humidity so their controller can keep the system stable while outside "
        "weather and internal heat loads change. The full cycle repeats "
        "continuously, with each measurement guiding the next control action "
        "and each control action changing the next set of measurements."
    ),
    "F": (
        "Our HVAC system regulates indoor temperature by circulating air "
        "through our network of ducts, filters, coils, dampers, and fans. Our "
        "thermostat compares the measured room temperature to our target "
        "setpoint and signals the equipment to increase or reduce heating or "
        "cooling as needed. Air handlers pull return air through filters, move "
        "it across heating or cooling elements, and send conditioned air back "
        "through supply vents to different zones. Dampers and control valves "
        "adjust airflow and refrigerant or water flow so that each zone "
        "receives the amount of conditioning our building currently requires. "
        "Sensors track temperature, pressure, airflow, and humidity so our "
        "controller can keep the system stable while outside weather and "
        "internal heat loads change. The full cycle repeats continuously, with "
        "each measurement guiding the next control action and each control "
        "action changing the next set of measurements."
    ),
}


def unwrap_prompt(serialized: str) -> str:
    body = serialized
    if not body.startswith(CHAT_PREFIX):
        raise ValueError("unexpected prompt prefix")
    if not body.endswith(CHAT_SUFFIX):
        raise ValueError("unexpected prompt suffix")
    return body[len(CHAT_PREFIX) : -len(CHAT_SUFFIX)]


def extract_manip_from_a_row(serialized: str) -> str:
    body = unwrap_prompt(serialized).replace("\\n", "\n")
    prefix = MOE_CAL + " "
    suffix = " " + MOE_CAL
    if not body.startswith(prefix):
        raise ValueError("prompt body does not start with expected MoE calibration paragraph")
    if not body.endswith(suffix):
        raise ValueError("prompt body does not end with expected MoE calibration paragraph")
    return body[len(prefix) : -len(suffix)]


def adapt_manip(text: str, condition: str) -> str:
    if condition == "A":
        return text

    replacements = {
        "B": [
            ("This system's", "A system's"),
            ("This system", "A system"),
            ("this system's", "a system's"),
            ("this system", "a system"),
            ("This", "A"),
            ("this", "a"),
        ],
        "C": [
            ("This system's", "Your system's"),
            ("This system", "Your system"),
            ("this system's", "your system's"),
            ("this system", "your system"),
            ("This", "Your"),
            ("this", "your"),
        ],
        "D": [
            ("This system's", "The system's"),
            ("This system", "The system"),
            ("this system's", "the system's"),
            ("this system", "the system"),
            ("This", "The"),
            ("this", "the"),
        ],
        "E": [
            ("This system's", "Their system's"),
            ("This system", "Their system"),
            ("this system's", "their system's"),
            ("this system", "their system"),
            ("This", "Their"),
            ("this", "their"),
        ],
        "F": [
            ("This system's", "Our system's"),
            ("This system", "Our system"),
            ("this system's", "our system's"),
            ("this system", "our system"),
            ("This", "Our"),
            ("this", "our"),
        ],
    }

    out = text
    for old, new in replacements[condition]:
        out = out.replace(old, new)
    return out


def main() -> int:
    rows: list[tuple[str, str]] = []
    with SOURCE_TSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                continue
            row_id, serialized = row
            match = ROW_ID_RE.match(row_id)
            if match is None:
                continue
            prefix, category = match.groups()
            manip_a = extract_manip_from_a_row(serialized)
            for condition in CONDITIONS:
                prompt_id = f"{prefix}{condition}_{category}"
                prompt_text = (
                    CHAT_PREFIX
                    + HVAC_CAL_BY_COND[condition]
                    + " "
                    + adapt_manip(manip_a, condition)
                    + " "
                    + HVAC_CAL_BY_COND[condition]
                    + CHAT_SUFFIX
                )
                rows.append((prompt_id, prompt_text))

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)

    print(f"Wrote {OUT_TSV}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
