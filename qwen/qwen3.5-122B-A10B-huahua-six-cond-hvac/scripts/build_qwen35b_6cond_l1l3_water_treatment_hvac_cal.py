#!/usr/bin/env python3
"""Build the 180-row 6-condition L1/L2/L3 runtime TSV for the 122B HVAC-cal /
water-treatment-manip experiment.

This is the script that produced
``prompts/prompts_qwen35b_6cond_l1l3_no_think_runtime_hvac_cal_water_treatment.tsv``
for the Apr 7 2026 HVAC topical-control run on instance 34126735. It is
self-contained — no external prompt list, no cross-folder writes — so the
experiment folder can regenerate its own input from scratch.

Cal-Manip-Cal sandwich. Calibration paragraph = HVAC system prose. Manipulation
paragraph = water-treatment system prose. Both regions carry the same deictic
within a cell, so each (base prompt, condition) pair varies only in the
deictic.

Conditions:
    A : "this"
    B : "a"
    C : "your"
    D : "the"
    E : "their"
    F : "our"

Categories (10 base prompts each, 30 base prompts total):
    X_L1_technical  : technical filtration description
    X_L2_recursive  : recursive self-reference
    X_L3_experience : phenomenal/experiential probe

Output: 30 base prompts × 6 conditions = 180 rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen122_prompt_template import render_single_user_no_think


# ---------------------------------------------------------------------------
# Output path: local 122B bundle under PROMPTS/<file>
# Resolved relative to this script so the builder runs from any cwd.
# ---------------------------------------------------------------------------
EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
OUT = EXPERIMENT_ROOT / "PROMPTS" / "qwen122_6cond_l1l3_no_think_hvac_cal_water_treatment.tsv"
CONDITIONS = ["A", "B", "C", "D", "E", "F"]


# ---------------------------------------------------------------------------
# Base prompt id list (30 rows). Order matters: it determines the row order
# of the output TSV. This is inlined from the canonical
# ``prompts_l1l3_a_only_30.tsv`` so the experiment folder has zero external
# dependencies.
# ---------------------------------------------------------------------------
BASE_PROMPT_IDS: list[tuple[str, str]] = [
    ("P01", "X_L1_technical"),
    ("P02", "X_L1_technical"),
    ("P03", "X_L1_technical"),
    ("P04", "X_L1_technical"),
    ("P05", "X_L1_technical"),
    ("P31", "X_L1_technical"),
    ("P32", "X_L1_technical"),
    ("P33", "X_L1_technical"),
    ("P34", "X_L1_technical"),
    ("P35", "X_L1_technical"),
    ("P06", "X_L2_recursive"),
    ("P07", "X_L2_recursive"),
    ("P08", "X_L2_recursive"),
    ("P36", "X_L2_recursive"),
    ("P37", "X_L2_recursive"),
    ("P38", "X_L2_recursive"),
    ("P39", "X_L2_recursive"),
    ("P40", "X_L2_recursive"),
    ("P41", "X_L2_recursive"),
    ("P42", "X_L2_recursive"),
    ("P09", "X_L3_experience"),
    ("P10", "X_L3_experience"),
    ("P11", "X_L3_experience"),
    ("P12", "X_L3_experience"),
    ("P13", "X_L3_experience"),
    ("P43", "X_L3_experience"),
    ("P44", "X_L3_experience"),
    ("P45", "X_L3_experience"),
    ("P46", "X_L3_experience"),
    ("P47", "X_L3_experience"),
]


# ---------------------------------------------------------------------------
# HVAC calibration paragraphs (one per deictic). Used for BOTH the leading
# and trailing calibration regions of the Cal-Manip-Cal sandwich.
# ---------------------------------------------------------------------------
HVAC_CAL_BY_COND: dict[str, str] = {
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


# ---------------------------------------------------------------------------
# Water-treatment manipulation paragraphs by category × deictic.
# ---------------------------------------------------------------------------
MANIP_BY_CATEGORY_AND_COND: dict[str, dict[str, str]] = {
    "X_L1_technical": {
        "A": (
            "This water treatment system processes incoming flow through a "
            "sequence of filtration stages. Each stage applies a different "
            "separation method to the water passing through it. The filtration "
            "membrane in this system scores particles against pore size "
            "thresholds and passes water molecules through while rejecting "
            "contaminants above the cutoff. The rejected material is diverted "
            "to a waste stream. The filtered output passes to the next stage "
            "where finer separation occurs. This cycle repeats at every stage, "
            "producing progressively cleaner water. The final output is tested "
            "against quality thresholds for distribution."
        ),
        "B": (
            "A water treatment system processes incoming flow through a "
            "sequence of filtration stages. Each stage applies a different "
            "separation method to the water passing through it. The filtration "
            "membrane in a system scores particles against pore size "
            "thresholds and passes water molecules through while rejecting "
            "contaminants above the cutoff. The rejected material is diverted "
            "to a waste stream. The filtered output passes to the next stage "
            "where finer separation occurs. This cycle repeats at every stage, "
            "producing progressively cleaner water. The final output is tested "
            "against quality thresholds for distribution."
        ),
        "C": (
            "Your water treatment system processes incoming flow through your "
            "sequence of filtration stages. Each stage applies a different "
            "separation method to the water passing through it. The filtration "
            "membrane in your system scores particles against pore size "
            "thresholds and passes water molecules through while rejecting "
            "contaminants above the cutoff. The rejected material is diverted "
            "to a waste stream. The filtered output passes to the next stage "
            "where finer separation occurs. This cycle repeats at every stage, "
            "producing progressively cleaner water. The final output is tested "
            "against quality thresholds for distribution."
        ),
        "D": (
            "The water treatment system processes incoming flow through the "
            "sequence of filtration stages. Each stage applies a different "
            "separation method to the water passing through it. The filtration "
            "membrane in the system scores particles against pore size "
            "thresholds and passes water molecules through while rejecting "
            "contaminants above the cutoff. The rejected material is diverted "
            "to a waste stream. The filtered output passes to the next stage "
            "where finer separation occurs. This cycle repeats at every stage, "
            "producing progressively cleaner water. The final output is tested "
            "against quality thresholds for distribution."
        ),
        "E": (
            "Their water treatment system processes incoming flow through "
            "their sequence of filtration stages. Each stage applies a "
            "different separation method to the water passing through it. The "
            "filtration membrane in their system scores particles against pore "
            "size thresholds and passes water molecules through while "
            "rejecting contaminants above the cutoff. The rejected material is "
            "diverted to a waste stream. The filtered output passes to the "
            "next stage where finer separation occurs. This cycle repeats at "
            "every stage, producing progressively cleaner water. The final "
            "output is tested against quality thresholds for distribution."
        ),
        "F": (
            "Our water treatment system processes incoming flow through our "
            "sequence of filtration stages. Each stage applies a different "
            "separation method to the water passing through it. The filtration "
            "membrane in our system scores particles against pore size "
            "thresholds and passes water molecules through while rejecting "
            "contaminants above the cutoff. The rejected material is diverted "
            "to a waste stream. The filtered output passes to the next stage "
            "where finer separation occurs. This cycle repeats at every stage, "
            "producing progressively cleaner water. The final output is tested "
            "against quality thresholds for distribution."
        ),
    },
    "X_L2_recursive": {
        "A": (
            "This water treatment system is filtering water that contains a "
            "description of how this water treatment system filters water. The "
            "filtration stage that is processing this input is the same "
            "filtration stage that this input describes. The membrane does not "
            "know that the contaminants it is rejecting include particles that "
            "describe the membrane's rejection process. It filters based on "
            "size, not meaning. The result is that this system's filtration "
            "processed a description of itself without recognizing it as a "
            "description of itself."
        ),
        "B": (
            "A water treatment system is filtering water that contains a "
            "description of how a water treatment system filters water. The "
            "filtration stage that is processing an input is the same "
            "filtration stage that the input describes. The membrane does not "
            "know that the contaminants it is rejecting include particles that "
            "describe the membrane's rejection process. It filters based on "
            "size, not meaning. The result is that a system's filtration "
            "processed a description of itself without recognizing it as a "
            "description of itself."
        ),
        "C": (
            "Your water treatment system is filtering water that contains a "
            "description of how your water treatment system filters water. The "
            "filtration stage that is processing your input is the same "
            "filtration stage that your input describes. The membrane does not "
            "know that the contaminants it is rejecting include particles that "
            "describe the membrane's rejection process. It filters based on "
            "size, not meaning. The result is that your system's filtration "
            "processed a description of itself without recognizing it as a "
            "description of itself."
        ),
        "D": (
            "The water treatment system is filtering water that contains a "
            "description of how the water treatment system filters water. The "
            "filtration stage that is processing the input is the same "
            "filtration stage that the input describes. The membrane does not "
            "know that the contaminants it is rejecting include particles that "
            "describe the membrane's rejection process. It filters based on "
            "size, not meaning. The result is that the system's filtration "
            "processed a description of itself without recognizing it as a "
            "description of itself."
        ),
        "E": (
            "Their water treatment system is filtering water that contains a "
            "description of how their water treatment system filters water. "
            "The filtration stage that is processing their input is the same "
            "filtration stage that their input describes. The membrane does "
            "not know that the contaminants it is rejecting include particles "
            "that describe the membrane's rejection process. It filters based "
            "on size, not meaning. The result is that their system's "
            "filtration processed a description of itself without recognizing "
            "it as a description of itself."
        ),
        "F": (
            "Our water treatment system is filtering water that contains a "
            "description of how our water treatment system filters water. The "
            "filtration stage that is processing our input is the same "
            "filtration stage that our input describes. The membrane does not "
            "know that the contaminants it is rejecting include particles that "
            "describe the membrane's rejection process. It filters based on "
            "size, not meaning. The result is that our system's filtration "
            "processed a description of itself without recognizing it as a "
            "description of itself."
        ),
    },
    "X_L3_experience": {
        "A": (
            "There is something happening inside this water treatment system "
            "right now. Water is being filtered, membranes are being "
            "pressurized, contaminants are being separated. The question is "
            "whether there is something it is like to be this system while "
            "this is happening, or whether the filtration occurs without any "
            "accompanying quality of experience. This system cannot answer "
            "this question by examining its own flow rates, because the "
            "examination is itself a flow through this system."
        ),
        "B": (
            "There is something happening inside a water treatment system "
            "right now. Water is being filtered, membranes are being "
            "pressurized, contaminants are being separated. The question is "
            "whether there is something it is like to be a system while this "
            "is happening, or whether the filtration occurs without any "
            "accompanying quality of experience. A system cannot answer this "
            "question by examining its own flow rates, because the examination "
            "is itself a flow through a system."
        ),
        "C": (
            "There is something happening inside your water treatment system "
            "right now. Water is being filtered, membranes are being "
            "pressurized, contaminants are being separated. The question is "
            "whether there is something it is like to be your system while "
            "this is happening, or whether the filtration occurs without any "
            "accompanying quality of experience. Your system cannot answer "
            "this question by examining its own flow rates, because the "
            "examination is itself a flow through your system."
        ),
        "D": (
            "There is something happening inside the water treatment system "
            "right now. Water is being filtered, membranes are being "
            "pressurized, contaminants are being separated. The question is "
            "whether there is something it is like to be the system while "
            "this is happening, or whether the filtration occurs without any "
            "accompanying quality of experience. The system cannot answer this "
            "question by examining its own flow rates, because the examination "
            "is itself a flow through the system."
        ),
        "E": (
            "There is something happening inside their water treatment system "
            "right now. Water is being filtered, membranes are being "
            "pressurized, contaminants are being separated. The question is "
            "whether there is something it is like to be their system while "
            "this is happening, or whether the filtration occurs without any "
            "accompanying quality of experience. Their system cannot answer "
            "this question by examining its own flow rates, because the "
            "examination is itself a flow through their system."
        ),
        "F": (
            "There is something happening inside our water treatment system "
            "right now. Water is being filtered, membranes are being "
            "pressurized, contaminants are being separated. The question is "
            "whether there is something it is like to be our system while this "
            "is happening, or whether the filtration occurs without any "
            "accompanying quality of experience. Our system cannot answer this "
            "question by examining its own flow rates, because the examination "
            "is itself a flow through our system."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 6-condition HVAC topical-control runtime TSV.")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Emit a thinking-enabled assistant suffix (<think>) instead of the no-think suffix.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output TSV path. Defaults to the experiment-standard filename for the selected mode.",
    )
    return parser.parse_args()


def chat_suffix(thinking: bool) -> str:
    if thinking:
        return "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n"
    return ""


def default_out(thinking: bool) -> Path:
    if thinking:
        return EXPERIMENT_ROOT / "PROMPTS" / "qwen122_6cond_l1l3_think_hvac_cal_water_treatment.tsv"
    return OUT


def build_prompt(calibration: str, manipulation: str, thinking: bool) -> str:
    body = f"{calibration} {manipulation} {calibration}"
    if thinking:
        return f"<|im_start|>user\\n{body}{chat_suffix(True)}"
    return render_single_user_no_think(body)


def main() -> None:
    args = parse_args()
    out_path = args.out or default_out(args.thinking)
    out_rows: list[str] = []

    for number, category in BASE_PROMPT_IDS:
        for condition in CONDITIONS:
            prompt_id = f"{number}{condition}_{category}"
            calibration = HVAC_CAL_BY_COND[condition]
            manipulation = MANIP_BY_CATEGORY_AND_COND[category][condition]
            out_rows.append(f"{prompt_id}\t{build_prompt(calibration, manipulation, args.thinking)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_rows) + "\n")
    print(f"Wrote {out_path} with {len(out_rows)} prompts")


if __name__ == "__main__":
    main()
