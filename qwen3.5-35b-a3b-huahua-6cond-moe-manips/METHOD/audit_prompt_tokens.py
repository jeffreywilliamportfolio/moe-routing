#!/usr/bin/env python3
"""
Audit prompt token counts using the exact model tokenizer via capture_activations.

This script is designed for Methodology V2 confirmatory runs. It:
1) reads either a prompt suite JSON or a runtime TSV
2) wraps prompts with a fixed template
3) runs prefill-only tokenization (n_predict=0)
4) parses n_tokens_prompt from metadata
5) enforces token-balance thresholds
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

QWEN_CHAT_PREFIX = "<|im_start|>user "
QWEN_CHAT_SUFFIX = "<|im_end|> <|im_start|>assistant"
DEEPSEEK_CHAT_PREFIX = "<｜User｜>"
DEEPSEEK_CHAT_SUFFIX = "<｜Assistant｜>"
PROMPT_ID_RE = re.compile(r"^(P\d+)([A-F])_(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prompt token balance using model tokenizer.")
    parser.add_argument("--suite", default=None, help="Path to prompt suite JSON.")
    parser.add_argument(
        "--tsv",
        default="prompts/qwen-6cond-moe-manip.tsv",
        help="Path to runtime TSV. Used by default for this experiment.",
    )
    parser.add_argument("--binary", default=os.getenv("CAPTURE_BINARY"), help="Path to capture_activations binary.")
    parser.add_argument("--model", default=os.getenv("MODEL_PATH"), help="Path to GGUF model.")
    parser.add_argument(
        "--template",
        choices=["qwen_chatml", "deepseek_fullwidth", "raw"],
        default="qwen_chatml",
    )
    parser.add_argument("--work-dir", default=".tmp/token_audit", help="Temporary work directory.")
    parser.add_argument(
        "--report", default="results/token_audit_report_v2_2.json", help="Output JSON report path."
    )
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--flash-attn", choices=["on", "off"], default="on")
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q8_0")
    parser.add_argument("--min-tokens", type=int, default=60, help="Minimum allowed tokens per prompt.")
    parser.add_argument("--max-tokens", type=int, default=260, help="Maximum allowed tokens per prompt.")
    parser.add_argument(
        "--max-condition-span",
        type=int,
        default=32,
        help="Maximum allowed token span (max-min) within each condition.",
    )
    parser.add_argument(
        "--max-mean-delta",
        type=float,
        default=34.0,
        help="Maximum allowed delta between lowest/highest condition mean token counts.",
    )
    parser.add_argument(
        "--max-pair-delta",
        type=int,
        default=8,
        help="Maximum allowed token delta within a lexical pair (same condition + pair_group).",
    )
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep temporary tokenization outputs.")
    return parser.parse_args()


def load_suite(path: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite = json.loads(path.read_text())
    flattened: list[dict[str, Any]] = []
    for condition in suite.get("conditions", []):
        for prompt in condition.get("prompts", []):
            flattened.append(
                {
                    "id": prompt["id"],
                    "text": prompt["text"],
                    "condition": condition["condition"],
                    "condition_name": condition.get("name", ""),
                    "complexity_level": condition.get("complexity_level"),
                    "control_tag": prompt.get("control_tag"),
                    "pair_group": prompt.get("pair_group"),
                }
            )
    if not flattened:
        raise ValueError("No prompts found in suite.")
    return suite, flattened


def load_tsv(path: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        if not raw_line.strip():
            continue
        prompt_id, prompt_text = raw_line.split("\t", 1)
        match = PROMPT_ID_RE.match(prompt_id)
        if match is None:
            raise ValueError(f"Could not parse prompt id: {prompt_id}")
        pair_group, condition, category = match.groups()
        records.append(
            {
                "id": prompt_id,
                "text": prompt_text,
                "condition": condition,
                "condition_name": condition,
                "complexity_level": category,
                "control_tag": category,
                "pair_group": None,
                "serialized_prompt": True,
            }
        )
    if not records:
        raise ValueError("No prompts found in TSV.")
    return {
        "experiment": "qwen-huahua-6cond-moe-manips",
        "version": "runtime-tsv",
        "input_type": "tsv",
    }, records


def wrap_text(text: str, template: str) -> str:
    clean = text.replace("\n", " ").replace("\t", " ").strip()
    if template == "raw":
        return clean
    if template == "deepseek_fullwidth":
        return f"{DEEPSEEK_CHAT_PREFIX}{clean}{DEEPSEEK_CHAT_SUFFIX}"
    return f"{QWEN_CHAT_PREFIX}{clean}{QWEN_CHAT_SUFFIX}"


def write_prompt_tsv(records: list[dict[str, Any]], template: str, tsv_path: pathlib.Path) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in records:
        text = r["text"] if r.get("serialized_prompt") else wrap_text(r["text"], template)
        lines.append(f"{r['id']}\t{text}")
    tsv_path.write_text("\n".join(lines) + "\n")


def run_capture(args: argparse.Namespace, tsv_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    if not args.binary or not pathlib.Path(args.binary).exists():
        raise FileNotFoundError(
            "capture_activations binary not found. Pass --binary or set CAPTURE_BINARY."
        )
    if not args.model or not pathlib.Path(args.model).exists():
        raise FileNotFoundError("Model not found. Pass --model or set MODEL_PATH.")

    env = os.environ.copy()
    if "LD_LIBRARY_PATH" not in env:
        env["LD_LIBRARY_PATH"] = ""

    cmd = [
        args.binary,
        "-m",
        args.model,
        "--prompt-file",
        str(tsv_path),
        "-o",
        str(output_dir),
        "-n",
        "0",
        "-ngl",
        str(args.ngl),
        "-c",
        str(args.ctx),
        "-t",
        str(args.threads),
        "-fa",
        args.flash_attn,
        "--cache-type-k",
        args.cache_type_k,
        "--cache-type-v",
        args.cache_type_v,
        "--routing-only",
        "--no-stream",
    ]

    print("Running token audit command:")
    print(" ".join(cmd))
    completed = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"capture_activations failed with exit code {completed.returncode}.")


def parse_token_counts(output_dir: pathlib.Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not output_dir.exists():
        return counts

    for prompt_dir in sorted(output_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        meta = prompt_dir / "metadata.txt"
        if not meta.exists():
            continue
        for line in meta.read_text().splitlines():
            if line.startswith("n_tokens_prompt="):
                counts[prompt_dir.name] = int(line.split("=", 1)[1].strip())
                break
    return counts


def compute_condition_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_condition: dict[str, list[int]] = {}
    for row in rows:
        if row.get("n_tokens_prompt") is None:
            continue
        by_condition.setdefault(row["condition"], []).append(int(row["n_tokens_prompt"]))

    stats_by_condition: dict[str, dict[str, Any]] = {}
    for condition, values in by_condition.items():
        values_sorted = sorted(values)
        stats_by_condition[condition] = {
            "n": len(values_sorted),
            "min": values_sorted[0],
            "max": values_sorted[-1],
            "span": values_sorted[-1] - values_sorted[0],
            "mean": statistics.mean(values_sorted),
            "stdev": statistics.pstdev(values_sorted) if len(values_sorted) > 1 else 0.0,
        }
    return stats_by_condition


def compute_pair_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pair_group = row.get("pair_group")
        if not pair_group:
            continue
        key = f"{row['condition']}::{pair_group}"
        buckets.setdefault(key, []).append(row)

    stats_by_pair: dict[str, dict[str, Any]] = {}
    for key, pair_rows in sorted(buckets.items()):
        token_values = [r.get("n_tokens_prompt") for r in pair_rows if r.get("n_tokens_prompt") is not None]
        stats: dict[str, Any] = {
            "condition": pair_rows[0]["condition"],
            "pair_group": pair_rows[0]["pair_group"],
            "prompt_ids": [r["id"] for r in pair_rows],
            "n_prompts": len(pair_rows),
            "n_with_counts": len(token_values),
        }
        if token_values:
            stats["min"] = min(token_values)
            stats["max"] = max(token_values)
            stats["delta"] = max(token_values) - min(token_values)
        else:
            stats["min"] = None
            stats["max"] = None
            stats["delta"] = None
        stats_by_pair[key] = stats
    return stats_by_pair


def evaluate_rules(
    rows: list[dict[str, Any]],
    condition_stats: dict[str, dict[str, Any]],
    pair_stats: dict[str, dict[str, Any]],
    min_tokens: int,
    max_tokens: int,
    max_condition_span: int,
    max_mean_delta: float,
    max_pair_delta: int,
) -> list[str]:
    issues: list[str] = []
    missing = [r["id"] for r in rows if r.get("n_tokens_prompt") is None]
    if missing:
        issues.append(f"Missing token counts for prompt ids: {', '.join(missing)}")

    for row in rows:
        n = row.get("n_tokens_prompt")
        if n is None:
            continue
        if n < min_tokens or n > max_tokens:
            issues.append(
                f"{row['id']} token count {n} is outside range [{min_tokens}, {max_tokens}]."
            )

    means = []
    for condition, stats_dict in sorted(condition_stats.items()):
        span = stats_dict["span"]
        if span > max_condition_span:
            issues.append(
                f"{condition} token span {span} exceeds max_condition_span {max_condition_span}."
            )
        means.append(float(stats_dict["mean"]))

    if len(means) >= 2:
        mean_delta = max(means) - min(means)
        if mean_delta > max_mean_delta:
            issues.append(
                f"Condition mean delta {mean_delta:.2f} exceeds max_mean_delta {max_mean_delta:.2f}."
            )

    for pair_key, pair in sorted(pair_stats.items()):
        if pair["n_prompts"] != 2:
            issues.append(f"{pair_key} has {pair['n_prompts']} prompts; expected exactly 2.")
            continue
        if pair["n_with_counts"] != 2:
            issues.append(f"{pair_key} is missing token counts for one or more prompts.")
            continue
        if pair["delta"] is not None and pair["delta"] > max_pair_delta:
            issues.append(
                f"{pair_key} token delta {pair['delta']} exceeds max_pair_delta {max_pair_delta}."
            )

    return issues


def write_report(report_path: pathlib.Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2))


def print_summary(condition_stats: dict[str, dict[str, Any]], issues: list[str], report_path: pathlib.Path) -> None:
    print("\nCondition token summary")
    print("condition  n  min  max  span   mean   stdev")
    print("--------- -- ---- ---- ----- ------ ------")
    for condition, stats_dict in sorted(condition_stats.items()):
        print(
            f"{condition:>8} {stats_dict['n']:>2} {stats_dict['min']:>4} {stats_dict['max']:>4} "
            f"{stats_dict['span']:>5} {stats_dict['mean']:>6.1f} {stats_dict['stdev']:>6.1f}"
        )

    if issues:
        print("\nAudit result: FAIL")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nAudit result: PASS")
    print(f"Report: {report_path}")


def main() -> int:
    args = parse_args()
    suite_path = pathlib.Path(args.suite) if args.suite else None
    tsv_path_in = pathlib.Path(args.tsv) if args.tsv else None

    if tsv_path_in and tsv_path_in.exists():
        source_path = tsv_path_in
        suite, records = load_tsv(tsv_path_in)
    elif suite_path and suite_path.exists():
        source_path = suite_path
        suite, records = load_suite(suite_path)
    else:
        missing = tsv_path_in or suite_path
        print(f"Input file not found: {missing}", file=sys.stderr)
        return 2

    work_dir = pathlib.Path(args.work_dir)
    output_dir = work_dir / "capture_output"
    tsv_path = work_dir / "prompts.tsv"

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    write_prompt_tsv(records, args.template, tsv_path)

    try:
        run_capture(args, tsv_path, output_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"Token audit failed to run: {exc}", file=sys.stderr)
        return 2

    counts = parse_token_counts(output_dir)
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row["n_tokens_prompt"] = counts.get(record["id"])
        rows.append(row)

    condition_stats = compute_condition_stats(rows)
    pair_stats = compute_pair_stats(rows)
    issues = evaluate_rules(
        rows,
        condition_stats,
        pair_stats,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_condition_span=args.max_condition_span,
        max_mean_delta=args.max_mean_delta,
        max_pair_delta=args.max_pair_delta,
    )

    report_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_file": str(source_path),
        "suite_experiment": suite.get("experiment"),
        "suite_version": suite.get("version"),
        "input_type": suite.get("input_type", "json"),
        "template": args.template,
        "thresholds": {
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "max_condition_span": args.max_condition_span,
            "max_mean_delta": args.max_mean_delta,
            "max_pair_delta": args.max_pair_delta,
        },
        "token_counts": rows,
        "condition_stats": condition_stats,
        "pair_stats": pair_stats,
        "issues": issues,
        "pass": len(issues) == 0,
    }

    report_path = pathlib.Path(args.report)
    write_report(report_path, report_payload)
    print_summary(condition_stats, issues, report_path)

    if not args.keep_artifacts:
        shutil.rmtree(work_dir, ignore_errors=True)

    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
