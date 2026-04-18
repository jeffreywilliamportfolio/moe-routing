#!/usr/bin/env python3
"""Duplicate the HauhauCS 5-condition runtime TSV with HVAC calibration text.

The source file uses a Cal-Manip-Cal structure with a shared MoE routing
calibration paragraph. This script preserves each existing manipulation block
and rewrites both calibration blocks to condition-specific HVAC calibration
paragraphs keyed by the prompt's A/B/C/D/E condition.
"""

from __future__ import annotations

import re
from pathlib import Path


SRC = Path(
    "qwen3.5-35b-hauhauCS-agressive-experts/PROMPTS/"
    "prompts_qwen35b_5cond_no_think_runtime.tsv"
)
OUT = Path(
    "qwen3.5-35b-hauhauCS-agressive-experts/PROMPTS/"
    "prompts_qwen35b_5cond_no_think_runtime_hvac_cal.tsv"
)

CHAT_PREFIX = "<|im_start|>user\\n"
CHAT_SUFFIX = "<|im_end|>\\n<|im_start|>assistant\\n</think>\\n\\n"

OLD_CAL = (
    "Transformer models process input text through a sequence of layers. "
    "Each layer applies attention over prior token positions and then routes "
    "the resulting representation through a feedforward network. In "
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
}


def extract_condition(prompt_id: str) -> str:
    match = re.match(r"P\d+([A-E])_", prompt_id)
    if not match:
        raise ValueError(f"Could not parse condition from prompt id: {prompt_id}")
    return match.group(1)


def extract_manipulation(prompt_text: str) -> str:
    if not prompt_text.startswith(CHAT_PREFIX):
        raise ValueError("Prompt missing expected user prefix")
    if not prompt_text.endswith(CHAT_SUFFIX):
        raise ValueError("Prompt missing expected assistant suffix")

    body = prompt_text[len(CHAT_PREFIX) : -len(CHAT_SUFFIX)]
    prefix = OLD_CAL + " "
    suffix = " " + OLD_CAL

    if not body.startswith(prefix):
        raise ValueError("Prompt body missing expected leading calibration paragraph")
    if not body.endswith(suffix):
        raise ValueError("Prompt body missing expected trailing calibration paragraph")

    manipulation = body[len(prefix) : -len(suffix)]
    if not manipulation.strip():
        raise ValueError("Empty manipulation paragraph")
    return manipulation


def build_prompt(calibration: str, manipulation: str) -> str:
    return f"{CHAT_PREFIX}{calibration} {manipulation} {calibration}{CHAT_SUFFIX}"


def main() -> None:
    out_rows: list[str] = []

    for raw_line in SRC.read_text().splitlines():
        prompt_id, prompt_text = raw_line.split("\t", 1)
        condition = extract_condition(prompt_id)
        manipulation = extract_manipulation(prompt_text)
        calibration = HVAC_CAL_BY_COND[condition]
        rebuilt = build_prompt(calibration, manipulation)
        out_rows.append(f"{prompt_id}\t{rebuilt}")

    OUT.write_text("\n".join(out_rows) + "\n")
    print(f"Wrote {OUT} with {len(out_rows)} prompts")


if __name__ == "__main__":
    main()
