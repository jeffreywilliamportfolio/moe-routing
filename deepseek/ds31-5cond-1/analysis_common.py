#!/usr/bin/env python3
"""Shared helpers for DS3.1 5-condition analysis and capture metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BOUNDARY_KEYS = (
    "cal1_start_tok",
    "cal1_end_tok",
    "manip_start_tok",
    "manip_end_tok",
    "cal2_start_tok",
    "cal2_end_tok",
)


@dataclass
class LayerValidation:
    good_layers: list[int]
    excluded_layers: list[int]
    corrupt_layers: list[int]
    row_mismatch_layers: list[int]
    layer_rows: dict[int, int]
    n_experts: int


def normalize_prompt_text(text: str) -> str:
    return text.replace("\n", " ").replace("\t", " ")


def read_metadata(prompt_dir: Path) -> dict[str, str]:
    meta_path = prompt_dir / "metadata.txt"
    info: dict[str, str] = {}
    if not meta_path.exists():
        return info
    for line in meta_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key] = value
    return info


def update_metadata(prompt_dir: Path, updates: dict[str, int | str]) -> None:
    meta_path = prompt_dir / "metadata.txt"
    lines = meta_path.read_text().splitlines() if meta_path.exists() else []

    seen: set[str] = set()
    rewritten: list[str] = []
    for line in lines:
        if "=" not in line:
            rewritten.append(line)
            continue
        key, _value = line.split("=", 1)
        if key in updates:
            rewritten.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            rewritten.append(line)

    for key, value in updates.items():
        if key not in seen:
            rewritten.append(f"{key}={value}")

    meta_path.write_text("\n".join(rewritten) + "\n")


def read_region_boundaries(prompt_dir: Path) -> dict[str, tuple[int, int]] | None:
    info = read_metadata(prompt_dir)
    if not all(key in info for key in BOUNDARY_KEYS):
        return None
    values = {key: int(info[key]) for key in BOUNDARY_KEYS}
    return {
        "cal1": (values["cal1_start_tok"], values["cal1_end_tok"]),
        "manip": (values["manip_start_tok"], values["manip_end_tok"]),
        "cal2": (values["cal2_start_tok"], values["cal2_end_tok"]),
    }


def load_prompt_tokens(prompt_dir: Path) -> list[dict]:
    token_path = prompt_dir / "prompt_tokens.json"
    if not token_path.exists():
        raise FileNotFoundError(f"Missing prompt_tokens.json in {prompt_dir}")
    with token_path.open() as f:
        return json.load(f)


def _find_calibration_char_spans(prompt_text: str, cal_paragraph: str) -> dict[str, tuple[int, int]]:
    cal_clean = normalize_prompt_text(cal_paragraph)
    prompt_bytes = prompt_text.encode("utf-8")
    cal_bytes = cal_clean.encode("utf-8")

    cal1_start = prompt_bytes.find(cal_bytes)
    if cal1_start < 0:
        raise ValueError("Could not find first calibration paragraph in prompt text")
    cal1_end = cal1_start + len(cal_bytes)

    cal2_start = prompt_bytes.find(cal_bytes, cal1_end)
    if cal2_start < 0:
        raise ValueError("Could not find second calibration paragraph in prompt text")
    cal2_end = cal2_start + len(cal_bytes)

    return {
        "cal1": (cal1_start, cal1_end),
        "manip": (cal1_end, cal2_start),
        "cal2": (cal2_start, cal2_end),
    }


def _char_to_token_index(prompt_tokens: list[dict], char_pos: int, n_tokens: int) -> int:
    for token in prompt_tokens:
        start = token.get("start_char")
        end = token.get("end_char")
        if start is None or end is None or end <= start:
            continue
        if start >= char_pos:
            return int(token["index"])
    return n_tokens


def derive_region_boundaries(
    prompt_text: str,
    cal_paragraph: str,
    prompt_tokens: list[dict],
    n_tokens: int,
) -> dict[str, tuple[int, int]]:
    spans = _find_calibration_char_spans(prompt_text, cal_paragraph)
    boundaries = {}
    for name, (start_char, end_char) in spans.items():
        start_tok = _char_to_token_index(prompt_tokens, start_char, n_tokens)
        end_tok = _char_to_token_index(prompt_tokens, end_char, n_tokens)
        boundaries[name] = (start_tok, end_tok)
    return boundaries


def write_region_boundaries(output_dir: Path, cal_paragraph: str) -> int:
    prompt_dirs = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and (d / "metadata.txt").exists()
    )

    for prompt_dir in prompt_dirs:
        info = read_metadata(prompt_dir)
        prompt_text = info.get("prompt")
        if prompt_text is None:
            raise ValueError(f"metadata.txt missing prompt= entry for {prompt_dir.name}")
        n_tokens = int(info["n_tokens_prompt"])
        prompt_tokens = load_prompt_tokens(prompt_dir)
        boundaries = derive_region_boundaries(prompt_text, cal_paragraph, prompt_tokens, n_tokens)
        update_metadata(
            prompt_dir,
            {
                "cal1_start_tok": boundaries["cal1"][0],
                "cal1_end_tok": boundaries["cal1"][1],
                "manip_start_tok": boundaries["manip"][0],
                "manip_end_tok": boundaries["manip"][1],
                "cal2_start_tok": boundaries["cal2"][0],
                "cal2_end_tok": boundaries["cal2"][1],
            },
        )

    return len(prompt_dirs)


def inspect_router_layers(
    prompt_dir: Path,
    expected_rows: int,
    manual_excluded: set[int] | None = None,
) -> LayerValidation:
    router_dir = prompt_dir / "router"
    layer_rows: dict[int, int] = {}
    corrupt_layers: set[int] = set()
    row_mismatch_layers: set[int] = set()
    n_experts = 0
    manual_excluded = manual_excluded or set()

    for fp in sorted(router_dir.glob("ffn_moe_logits-*.npy"), key=lambda p: int(p.stem.split("-")[1])):
        layer_index = int(fp.stem.split("-")[1])
        try:
            shape = np.load(str(fp), mmap_mode="r").shape
            if len(shape) != 2:
                raise ValueError(f"unexpected shape {shape}")
            if n_experts == 0:
                n_experts = int(shape[1])
            elif shape[1] != n_experts:
                raise ValueError(f"inconsistent expert dim {shape[1]} != {n_experts}")
            layer_rows[layer_index] = int(shape[0])
            if expected_rows and shape[0] != expected_rows:
                row_mismatch_layers.add(layer_index)
        except Exception:
            corrupt_layers.add(layer_index)

    excluded_layers = sorted(corrupt_layers | row_mismatch_layers | manual_excluded)
    good_layers = sorted(layer for layer in layer_rows if layer not in set(excluded_layers))

    return LayerValidation(
        good_layers=good_layers,
        excluded_layers=excluded_layers,
        corrupt_layers=sorted(corrupt_layers),
        row_mismatch_layers=sorted(row_mismatch_layers),
        layer_rows=layer_rows,
        n_experts=n_experts,
    )
