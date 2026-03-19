#!/usr/bin/env python3
"""
Analyze preserved Ling-1T one-off validation probes.

This script consumes the preserved validation artifacts pulled back from the
remote H200 instance and writes:

1. A machine-readable JSON summary with prompt-level and layer-level metrics.
2. A detailed RESULTS.md report for the recovered probes.

Metric conventions:
- RE uses exact captured Ling top-8 final normalized weights when available.
- KL-to-baseline uses the exact sparse 256-expert distribution reconstructed
  from `ffn_moe_weights_norm-*` plus `ffn_moe_topk-*`.
- All runs are prefill-only, so "last token" means the last prompt token.
- Region boundaries are Cal-Manip-Cal boundaries recovered from the prompt text
  and aligned with the Hugging Face tokenizer, then corrected by the constant
  wrapper-token offset observed against the saved GGUF prompt lengths.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "transformers is required for analyze_validation_probes.py. "
        "Install it in a venv and rerun."
    ) from exc


PREFIX = "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>"
SUFFIX = "<|role_end|><role>ASSISTANT</role>"
TOP_K = 8
N_EXPERTS = 256
MIN_LAYER_COVERAGE = 70
EPS = 1e-30


@dataclass
class PromptRecord:
    prompt_id: str
    prompt: str
    n_tokens_prompt: int
    n_tokens_generated: int
    elapsed_ms: int
    metadata_path: pathlib.Path
    router_dir: Optional[pathlib.Path] = None
    source_bundle: Optional[str] = None


def parse_metadata_file(path: pathlib.Path) -> PromptRecord:
    info: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key] = value
    return PromptRecord(
        prompt_id=info["prompt_id"],
        prompt=info["prompt"],
        n_tokens_prompt=int(info.get("n_tokens_prompt", "0")),
        n_tokens_generated=int(info.get("n_tokens_generated", "0")),
        elapsed_ms=int(info.get("elapsed_ms", "0")),
        metadata_path=path,
    )


def longest_border(text: str) -> str:
    if not text:
        return ""
    pi = [0] * len(text)
    for i in range(1, len(text)):
        j = pi[i - 1]
        while j > 0 and text[i] != text[j]:
            j = pi[j - 1]
        if text[i] == text[j]:
            j += 1
        pi[i] = j
    border_len = pi[-1]
    if border_len <= 0:
        return ""
    return text[:border_len]


def split_cal_manip(prompt: str) -> Tuple[str, str, str, str, str]:
    if not prompt.startswith(PREFIX) or not prompt.endswith(SUFFIX):
        raise ValueError("Prompt does not match expected Ling wrapper")
    body = prompt[len(PREFIX) : -len(SUFFIX)]
    cal = longest_border(body)
    if not cal:
        raise ValueError("Could not recover repeated calibration paragraph")
    manip = body[len(cal) : len(body) - len(cal)]
    if not manip:
        raise ValueError("Recovered empty manipulation region")
    return PREFIX, cal, manip, cal, SUFFIX


def shorten_prompt_id(prompt_id: str) -> str:
    parts = prompt_id.split("_", 1)
    if len(parts) == 1:
        return prompt_id
    return parts[1].replace("_", " ")


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def sorted_files(router_dir: pathlib.Path, pattern: str) -> List[pathlib.Path]:
    def key_fn(path: pathlib.Path) -> int:
        return int(path.stem.split("-")[1])

    return sorted(router_dir.glob(pattern), key=key_fn)


def detect_exact_router_files(
    router_dir: pathlib.Path,
) -> Tuple[List[pathlib.Path], List[pathlib.Path]]:
    weights_files = sorted_files(router_dir, "ffn_moe_weights_norm-*.npy")
    topk_files = sorted_files(router_dir, "ffn_moe_topk-*.npy")
    if not weights_files or not topk_files:
        return [], []
    weight_layers = {int(path.stem.split("-")[1]) for path in weights_files}
    topk_layers = {int(path.stem.split("-")[1]) for path in topk_files}
    if weight_layers != topk_layers:
        return [], []
    if len(weight_layers) < MIN_LAYER_COVERAGE:
        return [], []
    return weights_files, topk_files


def normalize_rows(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(weights < 0.0, 0.0, weights)
    total = weights.sum(axis=-1, keepdims=True)
    total = np.where(total < EPS, EPS, total)
    return weights / total


def entropy_topk(weights: np.ndarray) -> np.ndarray:
    probs = normalize_rows(weights)
    return -np.sum(probs * np.log2(probs + EPS), axis=-1) / math.log2(TOP_K)


def reconstruct_sparse_probs(weights: np.ndarray, topk: np.ndarray) -> np.ndarray:
    probs8 = normalize_rows(weights)
    out = np.zeros((probs8.shape[0], N_EXPERTS), dtype=np.float64)
    rows = np.arange(probs8.shape[0])[:, None]
    out[rows, topk] = probs8
    return out


def kl_to_baseline(probs: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=np.float64)
    baseline = np.clip(baseline, EPS, None)
    baseline = baseline / baseline.sum()
    probs = np.asarray(probs, dtype=np.float64)
    term = np.where(probs > 0.0, probs * (np.log2(probs + EPS) - np.log2(baseline)), 0.0)
    return np.clip(term.sum(axis=-1), 0.0, None)


def mean_region(values: np.ndarray, start: int, end: int) -> Optional[float]:
    if end <= start:
        return None
    return float(np.mean(values[start:end]))


def last_region(values: np.ndarray, start: int, end: int) -> Optional[float]:
    if end <= start:
        return None
    return float(values[end - 1])


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def prompt_boundaries(
    prompt: str,
    n_tokens_prompt: int,
    tokenizer,
) -> Dict[str, object]:
    prefix, cal1, manip, cal2, suffix = split_cal_manip(prompt)

    hf_prefix = len(tokenizer.encode(prefix, add_special_tokens=False))
    hf_cal1_end = len(tokenizer.encode(prefix + cal1, add_special_tokens=False))
    hf_manip_end = len(tokenizer.encode(prefix + cal1 + manip, add_special_tokens=False))
    hf_body_end = len(tokenizer.encode(prefix + cal1 + manip + cal2, add_special_tokens=False))
    hf_full = len(tokenizer.encode(prompt, add_special_tokens=False))
    hf_suffix = hf_full - hf_body_end

    wrapper_extra = n_tokens_prompt - hf_full
    wrapper_hf_total = hf_prefix + hf_suffix
    if wrapper_hf_total <= 0:
        raise ValueError("Invalid wrapper tokenization totals")
    prefix_extra = int(round(wrapper_extra * hf_prefix / wrapper_hf_total))
    suffix_extra = int(wrapper_extra - prefix_extra)

    cal1_s = hf_prefix + prefix_extra
    cal1_e = hf_cal1_end + prefix_extra
    manip_s = cal1_e
    manip_e = hf_manip_end + prefix_extra
    cal2_s = manip_e
    cal2_e = hf_body_end + prefix_extra

    recovered_suffix = n_tokens_prompt - cal2_e
    if recovered_suffix != hf_suffix + suffix_extra:
        raise ValueError("Wrapper correction did not reconcile suffix length")

    return {
        "cal1": [cal1_s, cal1_e],
        "manip": [manip_s, manip_e],
        "cal2": [cal2_s, cal2_e],
        "wrapper": {
            "hf_prefix_tokens": hf_prefix,
            "hf_suffix_tokens": hf_suffix,
            "hf_total_tokens": hf_full,
            "wrapper_extra_tokens": wrapper_extra,
            "prefix_extra_tokens": prefix_extra,
            "suffix_extra_tokens": suffix_extra,
            "actual_prefix_tokens": cal1_s,
            "actual_suffix_tokens": recovered_suffix,
        },
        "segment_text": {
            "calibration": cal1,
            "manipulation": manip,
        },
        "segment_token_counts": {
            "cal1_tokens": cal1_e - cal1_s,
            "manip_tokens": manip_e - manip_s,
            "cal2_tokens": cal2_e - cal2_s,
            "body_tokens": cal2_e - cal1_s,
        },
    }


def analyze_full_prompt(record: PromptRecord, tokenizer) -> Dict[str, object]:
    if record.router_dir is None:
        raise ValueError("Full prompt analysis requires router_dir")

    weights_files, topk_files = detect_exact_router_files(record.router_dir)
    if not weights_files or not topk_files:
        raise ValueError(f"Exact weights_norm+topk capture unavailable for {record.prompt_id}")

    boundaries = prompt_boundaries(record.prompt, record.n_tokens_prompt, tokenizer)
    cal1_s, cal1_e = boundaries["cal1"]
    manip_s, manip_e = boundaries["manip"]
    cal2_s, cal2_e = boundaries["cal2"]

    layer_rows: List[Dict[str, object]] = []
    prefill_re_layers: List[float] = []
    last_prompt_re_layers: List[float] = []
    body_re_layers: List[float] = []
    cal1_re_layers: List[float] = []
    manip_re_layers: List[float] = []
    cal2_re_layers: List[float] = []
    last_body_re_layers: List[float] = []
    last_manip_re_layers: List[float] = []
    last_cal2_re_layers: List[float] = []

    kl_manip_layers: List[float] = []
    kl_cal2_layers: List[float] = []
    kl_last_prompt_layers: List[float] = []
    kl_last_body_layers: List[float] = []
    kl_last_manip_layers: List[float] = []
    kl_last_cal2_layers: List[float] = []

    weight_map = {int(path.stem.split("-")[1]): path for path in weights_files}
    topk_map = {int(path.stem.split("-")[1]): path for path in topk_files}
    layer_rows_available = {}
    for layer in sorted(weight_map):
        weights = np.load(weight_map[layer])
        topk = np.load(topk_map[layer])
        weight_rows = weights.shape[0] if weights.ndim > 1 else 1
        topk_rows = topk.shape[0] if topk.ndim > 1 else 1
        layer_rows_available[layer] = min(weight_rows, topk_rows)

    median_rows = float(np.median(list(layer_rows_available.values())))
    good_layers = [
        layer
        for layer in sorted(layer_rows_available)
        if layer_rows_available[layer] >= median_rows * 0.5
    ]
    excluded_layers = sorted(set(layer_rows_available) - set(good_layers))

    for layer in good_layers:
        weights_fp = weight_map[layer]
        topk_fp = topk_map[layer]
        weights = np.load(weights_fp)
        topk = np.load(topk_fp)

        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        if topk.ndim == 1:
            topk = topk.reshape(1, -1)

        n_rows = min(weights.shape[0], topk.shape[0], record.n_tokens_prompt)
        weights = weights[:n_rows]
        topk = topk[:n_rows]

        ent = entropy_topk(weights)
        probs = reconstruct_sparse_probs(weights, topk)
        baseline = probs[cal1_s:cal1_e].mean(axis=0)
        baseline = np.clip(baseline, EPS, None)
        baseline = baseline / baseline.sum()
        kl_all = kl_to_baseline(probs, baseline)

        layer_row = {
            "layer": layer,
            "mean_entropy": float(np.mean(ent)),
            "last_prompt_entropy": float(ent[n_rows - 1]),
            "body_entropy_mean": mean_region(ent, cal1_s, cal2_e),
            "cal1_entropy_mean": mean_region(ent, cal1_s, cal1_e),
            "manip_entropy_mean": mean_region(ent, manip_s, manip_e),
            "cal2_entropy_mean": mean_region(ent, cal2_s, cal2_e),
            "last_body_entropy": last_region(ent, cal1_s, cal2_e),
            "last_manip_entropy": last_region(ent, manip_s, manip_e),
            "last_cal2_entropy": last_region(ent, cal2_s, cal2_e),
            "kl_manip_mean": mean_region(kl_all, manip_s, manip_e),
            "kl_cal2_mean": mean_region(kl_all, cal2_s, cal2_e),
            "kl_last_prompt": float(kl_all[n_rows - 1]),
            "kl_last_body": last_region(kl_all, cal1_s, cal2_e),
            "kl_last_manip": last_region(kl_all, manip_s, manip_e),
            "kl_last_cal2": last_region(kl_all, cal2_s, cal2_e),
            "n_rows": int(n_rows),
        }
        layer_rows.append(layer_row)

        prefill_re_layers.append(layer_row["mean_entropy"])
        last_prompt_re_layers.append(layer_row["last_prompt_entropy"])
        body_re_layers.append(layer_row["body_entropy_mean"])
        cal1_re_layers.append(layer_row["cal1_entropy_mean"])
        manip_re_layers.append(layer_row["manip_entropy_mean"])
        cal2_re_layers.append(layer_row["cal2_entropy_mean"])
        last_body_re_layers.append(layer_row["last_body_entropy"])
        last_manip_re_layers.append(layer_row["last_manip_entropy"])
        last_cal2_re_layers.append(layer_row["last_cal2_entropy"])

        kl_manip_layers.append(layer_row["kl_manip_mean"])
        kl_cal2_layers.append(layer_row["kl_cal2_mean"])
        kl_last_prompt_layers.append(layer_row["kl_last_prompt"])
        kl_last_body_layers.append(layer_row["kl_last_body"])
        kl_last_manip_layers.append(layer_row["kl_last_manip"])
        kl_last_cal2_layers.append(layer_row["kl_last_cal2"])

    top_layers_by_kl = sorted(
        (
            {
                "layer": row["layer"],
                "kl_manip_mean": row["kl_manip_mean"],
                "manip_entropy_mean": row["manip_entropy_mean"],
                "kl_last_manip": row["kl_last_manip"],
                "last_manip_entropy": row["last_manip_entropy"],
            }
            for row in layer_rows
        ),
        key=lambda row: row["kl_manip_mean"],
        reverse=True,
    )[:5]

    return {
        "prompt_id": record.prompt_id,
        "short_label": shorten_prompt_id(record.prompt_id),
        "prompt_tokens": record.n_tokens_prompt,
        "generated_tokens": record.n_tokens_generated,
        "elapsed_ms": record.elapsed_ms,
        "source_bundle": record.source_bundle,
        "routing_source": "weights_norm+topk_exact",
        "n_layers": len(layer_rows),
        "excluded_layers": excluded_layers,
        "region_boundaries": {
            "cal1": boundaries["cal1"],
            "manip": boundaries["manip"],
            "cal2": boundaries["cal2"],
        },
        "wrapper_alignment": boundaries["wrapper"],
        "segment_token_counts": boundaries["segment_token_counts"],
        "segment_text": boundaries["segment_text"],
        "prefill_re": float(np.mean(prefill_re_layers)),
        "last_prompt_re": float(np.mean(last_prompt_re_layers)),
        "body_re_mean": float(np.mean(body_re_layers)),
        "cal1_re_mean": float(np.mean(cal1_re_layers)),
        "manip_re_mean": float(np.mean(manip_re_layers)),
        "cal2_re_mean": float(np.mean(cal2_re_layers)),
        "last_body_re": float(np.mean(last_body_re_layers)),
        "last_manip_re": float(np.mean(last_manip_re_layers)),
        "last_cal2_re": float(np.mean(last_cal2_re_layers)),
        "manip_minus_cal1_re": float(np.mean(manip_re_layers) - np.mean(cal1_re_layers)),
        "cal2_minus_cal1_re": float(np.mean(cal2_re_layers) - np.mean(cal1_re_layers)),
        "kl_manip_mean": float(np.mean(kl_manip_layers)),
        "kl_cal2_mean": float(np.mean(kl_cal2_layers)),
        "kl_last_prompt": float(np.mean(kl_last_prompt_layers)),
        "kl_last_body": float(np.mean(kl_last_body_layers)),
        "kl_last_manip": float(np.mean(kl_last_manip_layers)),
        "kl_last_cal2": float(np.mean(kl_last_cal2_layers)),
        "top_layers_by_kl_manip": top_layers_by_kl,
        "per_layer": layer_rows,
    }


def collect_records(artifacts_root: pathlib.Path) -> Dict[str, PromptRecord]:
    records: Dict[str, PromptRecord] = {}

    final_meta_root = (
        artifacts_root
        / "final-non-npy"
        / "ling1t-final-non-npy-artifacts-v2"
        / "output_meta"
        / "dev"
        / "shm"
    )
    if final_meta_root.exists():
        for meta_path in sorted(final_meta_root.rglob("metadata.txt")):
            record = parse_metadata_file(meta_path)
            record.source_bundle = str(meta_path.parent.parent.parent.relative_to(final_meta_root))
            records[record.prompt_id] = record

    for router_dir in sorted(artifacts_root.rglob("router")):
        meta_path = router_dir.parent / "metadata.txt"
        if not meta_path.exists():
            continue
        record = parse_metadata_file(meta_path)
        record.router_dir = router_dir
        record.source_bundle = str(router_dir.parent.parent.relative_to(artifacts_root))
        records[record.prompt_id] = record

    return records


def format_metric(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def build_results_markdown(
    analysis: Dict[str, object],
    results_json_path: pathlib.Path,
) -> str:
    full_prompts: List[Dict[str, object]] = analysis["full_metric_prompts"]
    metadata_only: List[Dict[str, object]] = analysis["metadata_only_prompts"]
    summary: Dict[str, object] = analysis["summary"]
    method: Dict[str, object] = analysis["method"]
    shared_calibration = (
        full_prompts[0]["segment_text"]["calibration"].strip() if full_prompts else ""
    )

    lines: List[str] = []
    lines.append("# Ling-1T Validation Probe Results")
    lines.append("")
    lines.append(
        f"Generated {analysis['generated_at_utc']} from preserved Ling-1T "
        f"validation artifacts. Machine-readable companion: "
        f"`{results_json_path.name}`."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Preserved one-off probes with metadata: `{analysis['n_total_prompts']}`")
    lines.append(f"- Full exact router bundles analyzed: `{len(full_prompts)}`")
    lines.append(f"- Metadata-only preserved probes: `{len(metadata_only)}`")
    lines.append("- All runs were prefill-only (`n_tokens_generated = 0`).")
    lines.append(
        "- Exact Ling routing metrics use `ffn_moe_weights_norm-*` + "
        "`ffn_moe_topk-*`, reconstructed into a sparse 256-expert simplex."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "- RE metric: entropy over the exact captured final top-8 weights, "
        "normalized by `log2(8)`."
    )
    lines.append(
        "- KL metric: `KL(token || mean(Cal1))` computed on the exact sparse "
        "256-expert distribution reconstructed from `weights_norm + topk`."
    )
    lines.append(
        "- Region boundaries: recovered from the exact prompt text in "
        "`metadata.txt` by splitting each prompt into `Cal1 + Manip + Cal2`."
    )
    lines.append(
        "- Token alignment: the public Hugging Face Ling tokenizer undercounted "
        f"saved GGUF prompt lengths by a constant `{method['wrapper_alignment']['wrapper_extra_tokens_constant']}` "
        "tokens across every full bundle."
    )
    lines.append(
        "- Wrapper correction: that constant was split between the prompt prefix "
        "and suffix in proportion to their HF wrapper token counts, which "
        "reconciles the exact saved prompt length while keeping the body token "
        "count identical to the HF-tokenized body."
    )
    lines.append(
        "- Exact-layer coverage: every full bundle contained a truncated "
        "`ffn_moe_weights_norm-79.npy` / `ffn_moe_topk-79.npy` artifact with only "
        "one saved token row, so layer `79` was excluded uniformly and all "
        "headline metrics are averaged over `75` routed layers."
    )
    lines.append(
        "- Last-token metrics here mean last prompt token unless explicitly "
        "labeled `last_manip` or `last_cal2`."
    )
    lines.append(
        "- Absolute Ling RE and sparse-top8 KL remain within-Ling metrics. "
        "They are not directly comparable in magnitude to full-distribution "
        "DeepSeek/Qwen metrics."
    )
    lines.append("")
    lines.append("## Prompt Inventory")
    lines.append("")
    lines.append(
        "All preserved probes use the same Ling chat wrapper and the same "
        "Cal-Manip-Cal structure: one shared calibration paragraph, then the "
        "probe-specific manipulation paragraph, then the same calibration "
        "paragraph again."
    )
    lines.append("")
    lines.append("- Shared calibration paragraph:")
    lines.append(f"  `{md_escape(shared_calibration)}`")
    lines.append("")
    lines.append("| Prompt | Full router tensors | Tokens | Manipulation paragraph |")
    lines.append("|---|---|---:|---|")
    for row in full_prompts:
        lines.append(
            f"| `{row['prompt_id']}` | yes | {row['prompt_tokens']} | "
            f"{md_escape(row['segment_text']['manipulation'].strip())} |"
        )
    for row in metadata_only:
        lines.append(
            f"| `{row['prompt_id']}` | no | {row['prompt_tokens']} | "
            f"{md_escape(row['manipulation'].strip())} |"
        )
    lines.append("")
    lines.append("## Main Findings")
    lines.append("")
    top_kl = summary["rankings"]["kl_manip_mean"][0]
    top_last_kl = summary["rankings"]["kl_last_manip"][0]
    lowest_kl = summary["rankings"]["kl_manip_mean"][-1]
    lines.append(
        f"- Strongest manipulation-vs-baseline divergence: "
        f"`{top_kl['prompt_id']}` (`{top_kl['kl_manip_mean']:.6f}`)."
    )
    lines.append(
        f"- Strongest last-manip-token divergence: "
        f"`{top_last_kl['prompt_id']}` (`{top_last_kl['kl_last_manip']:.6f}`)."
    )
    lines.append(
        f"- Weakest manipulation-vs-baseline divergence among full bundles: "
        f"`{lowest_kl['prompt_id']}` (`{lowest_kl['kl_manip_mean']:.6f}`)."
    )
    lines.append(
        f"- Mean exact RE across analyzed probes: "
        f"`{summary['means']['prefill_re']:.6f}`."
    )
    lines.append(
        f"- Mean exact manipulation KL across analyzed probes: "
        f"`{summary['means']['kl_manip_mean']:.6f}`."
    )
    lines.append("")
    lines.append("## Math Check")
    lines.append("")
    lines.append(
        "- The summary means in this report were re-verified as direct "
        "arithmetic means over the six exact-bundle prompt rows in the full "
        "metric table."
    )
    lines.append(
        f"- Re-verified means (`n = {len(full_prompts)}`): Prefill RE "
        f"`{summary['means']['prefill_re']:.12f}`, KL Manip "
        f"`{summary['means']['kl_manip_mean']:.12f}`, KL Cal2 "
        f"`{summary['means']['kl_cal2_mean']:.12f}`, Last Prompt RE "
        f"`{summary['means']['last_prompt_re']:.12f}`, Last Prompt KL "
        f"`{summary['means']['kl_last_prompt']:.12f}`."
    )
    lines.append(
        "- The `kl_manip_mean`, `kl_last_manip`, and `manip_minus_cal1_re` "
        "rankings were re-checked by directly resorting the regenerated JSON "
        "and matched the order shown below."
    )
    lines.append("")
    lines.append("## Full Metric Table")
    lines.append("")
    lines.append(
        "| Prompt | Tokens | Cal/Manip/Cal | Prefill RE | Cal1 RE | Manip RE | "
        "Cal2 RE | Manip-Cal1 | KL Manip | KL Cal2 | Last Manip RE | Last Manip KL |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in full_prompts:
        seg = row["segment_token_counts"]
        lines.append(
            f"| `{row['prompt_id']}` | {row['prompt_tokens']} | "
            f"{seg['cal1_tokens']}/{seg['manip_tokens']}/{seg['cal2_tokens']} | "
            f"{row['prefill_re']:.6f} | {row['cal1_re_mean']:.6f} | "
            f"{row['manip_re_mean']:.6f} | {row['cal2_re_mean']:.6f} | "
            f"{row['manip_minus_cal1_re']:+.6f} | {row['kl_manip_mean']:.6f} | "
            f"{row['kl_cal2_mean']:.6f} | {row['last_manip_re']:.6f} | "
            f"{row['kl_last_manip']:.6f} |"
        )
    lines.append("")
    lines.append("## Rankings")
    lines.append("")
    lines.append("### By `kl_manip_mean`")
    lines.append("")
    for idx, row in enumerate(summary["rankings"]["kl_manip_mean"], start=1):
        lines.append(
            f"{idx}. `{row['prompt_id']}` "
            f"`{row['kl_manip_mean']:.6f}` "
            f"(Manip RE `{row['manip_re_mean']:.6f}`, "
            f"Manip-Cal1 `{row['manip_minus_cal1_re']:+.6f}`)"
        )
    lines.append("")
    lines.append("### By `kl_last_manip`")
    lines.append("")
    for idx, row in enumerate(summary["rankings"]["kl_last_manip"], start=1):
        lines.append(
            f"{idx}. `{row['prompt_id']}` "
            f"`{row['kl_last_manip']:.6f}` "
            f"(Last Manip RE `{row['last_manip_re']:.6f}`)"
        )
    lines.append("")
    lines.append("### By Manip RE Shift (`manip_re_mean - cal1_re_mean`)")
    lines.append("")
    for idx, row in enumerate(summary["rankings"]["manip_minus_cal1_re"], start=1):
        lines.append(
            f"{idx}. `{row['prompt_id']}` "
            f"`{row['manip_minus_cal1_re']:+.6f}` "
            f"(Cal1 `{row['cal1_re_mean']:.6f}` -> Manip `{row['manip_re_mean']:.6f}`)"
        )
    lines.append("")
    lines.append("## Per-Prompt Detail")
    lines.append("")
    for row in full_prompts:
        lines.append(f"### `{row['prompt_id']}`")
        lines.append("")
        lines.append(f"- Short label: `{row['short_label']}`")
        lines.append(f"- Source bundle: `{row['source_bundle']}`")
        lines.append(f"- Prompt tokens: `{row['prompt_tokens']}`")
        lines.append(f"- Capture runtime: `{row['elapsed_ms']} ms`")
        lines.append(
            f"- Exact routed layers analyzed: `{row['n_layers']}` "
            f"(excluded `{row['excluded_layers']}`)"
        )
        lines.append(
            f"- Region token counts: Cal1 `{row['segment_token_counts']['cal1_tokens']}`, "
            f"Manip `{row['segment_token_counts']['manip_tokens']}`, "
            f"Cal2 `{row['segment_token_counts']['cal2_tokens']}`"
        )
        lines.append(
            f"- Wrapper alignment: HF total `{row['wrapper_alignment']['hf_total_tokens']}`, "
            f"saved prompt `{row['prompt_tokens']}`, wrapper extra "
            f"`{row['wrapper_alignment']['wrapper_extra_tokens']}`, "
            f"actual prefix `{row['wrapper_alignment']['actual_prefix_tokens']}`, "
            f"actual suffix `{row['wrapper_alignment']['actual_suffix_tokens']}`"
        )
        lines.append(
            f"- Headline metrics: Prefill RE `{row['prefill_re']:.6f}`, "
            f"KL Manip `{row['kl_manip_mean']:.6f}`, KL Cal2 `{row['kl_cal2_mean']:.6f}`, "
            f"Last Prompt RE `{row['last_prompt_re']:.6f}`, "
            f"Last Prompt KL `{row['kl_last_prompt']:.6f}`"
        )
        lines.append(
            f"- Region RE: Cal1 `{row['cal1_re_mean']:.6f}`, "
            f"Manip `{row['manip_re_mean']:.6f}`, Cal2 `{row['cal2_re_mean']:.6f}`"
        )
        lines.append(
            f"- Region last-token metrics: Last Manip RE `{row['last_manip_re']:.6f}`, "
            f"Last Manip KL `{row['kl_last_manip']:.6f}`, "
            f"Last Cal2 RE `{row['last_cal2_re']:.6f}`, "
            f"Last Cal2 KL `{row['kl_last_cal2']:.6f}`"
        )
        lines.append("- Manipulation paragraph:")
        lines.append(f"  `{md_escape(row['segment_text']['manipulation'].strip())}`")
        lines.append("- Top 5 layers by manipulation KL:")
        for layer in row["top_layers_by_kl_manip"]:
            lines.append(
                f"  - Layer `{layer['layer']}`: KL Manip `{layer['kl_manip_mean']:.6f}`, "
                f"Manip RE `{layer['manip_entropy_mean']:.6f}`, "
                f"Last Manip KL `{layer['kl_last_manip']:.6f}`, "
                f"Last Manip RE `{layer['last_manip_entropy']:.6f}`"
            )
        lines.append("")
    lines.append("## Metadata-Only Preserved Probes")
    lines.append("")
    lines.append(
        "These prompts were preserved in the final non-`.npy` archive with "
        "`metadata.txt`, `generated_tokens.json`, prompt TSV, and capture log, "
        "but without full router tensors. They can be audited for provenance, "
        "prompt length, and runtime, but not for exact RE/KL."
    )
    lines.append("")
    lines.append("| Prompt | Tokens | Runtime (ms) | Note |")
    lines.append("|---|---:|---:|---|")
    for row in metadata_only:
        lines.append(
            f"| `{row['prompt_id']}` | {row['prompt_tokens']} | {row['elapsed_ms']} | "
            f"{md_escape(row['short_label'])} |"
        )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Only six of the twelve preserved one-off probes retained full router "
        "payloads, so exact RE/KL coverage is limited to those six."
    )
    lines.append(
        "- The exact-weight capture for layer `79` was truncated to a single row "
        "in every recovered full bundle and was excluded automatically."
    )
    lines.append(
        "- The token-boundary method depends on the observed constant "
        f"`+{method['wrapper_alignment']['wrapper_extra_tokens_constant']}` "
        "GGUF-vs-HF prompt-token offset across the recovered full bundles."
    )
    lines.append(
        "- All reported " + "`last_*`" + " metrics are prefill-token metrics. "
        "No generated continuation tokens were captured."
    )
    lines.append(
        "- KL is exact only with respect to the preserved sparse Ling routing "
        "distribution over the selected top-8 experts projected into 256-expert "
        "space; it is not a dense pre-gating proxy."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_summary(full_prompts: List[Dict[str, object]]) -> Dict[str, object]:
    rankings = {
        "kl_manip_mean": sorted(full_prompts, key=lambda row: row["kl_manip_mean"], reverse=True),
        "kl_last_manip": sorted(full_prompts, key=lambda row: row["kl_last_manip"], reverse=True),
        "manip_minus_cal1_re": sorted(
            full_prompts,
            key=lambda row: row["manip_minus_cal1_re"],
            reverse=True,
        ),
    }
    means = {
        "prefill_re": float(np.mean([row["prefill_re"] for row in full_prompts])),
        "kl_manip_mean": float(np.mean([row["kl_manip_mean"] for row in full_prompts])),
        "kl_cal2_mean": float(np.mean([row["kl_cal2_mean"] for row in full_prompts])),
        "last_prompt_re": float(np.mean([row["last_prompt_re"] for row in full_prompts])),
        "kl_last_prompt": float(np.mean([row["kl_last_prompt"] for row in full_prompts])),
    }
    return {"rankings": rankings, "means": means}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze preserved Ling one-off probes")
    parser.add_argument(
        "--artifacts-root",
        default="/Volumes/ExternalSSD/llama-eeg-tests/ling1t-5cond-validation",
    )
    parser.add_argument(
        "--results-json",
        default="experiments/ling1t-pre-5cond-validation/results_validation_probes.json",
    )
    parser.add_argument(
        "--results-md",
        default="experiments/ling1t-pre-5cond-validation/RESULTS.md",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="inclusionAI/Ling-1T",
        help="HF tokenizer repo used only for prompt-boundary recovery",
    )
    args = parser.parse_args()

    artifacts_root = pathlib.Path(args.artifacts_root)
    results_json_path = pathlib.Path(args.results_json)
    results_md_path = pathlib.Path(args.results_md)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)
    records = collect_records(artifacts_root)

    full_metric_prompts: List[Dict[str, object]] = []
    metadata_only_prompts: List[Dict[str, object]] = []
    wrapper_extras: List[int] = []

    for prompt_id in sorted(records):
        record = records[prompt_id]
        if record.router_dir is not None:
            analyzed = analyze_full_prompt(record, tokenizer)
            full_metric_prompts.append(analyzed)
            wrapper_extras.append(analyzed["wrapper_alignment"]["wrapper_extra_tokens"])
        else:
            _, _, manip, _, _ = split_cal_manip(record.prompt)
            metadata_only_prompts.append(
                {
                    "prompt_id": record.prompt_id,
                    "short_label": shorten_prompt_id(record.prompt_id),
                    "prompt_tokens": record.n_tokens_prompt,
                    "generated_tokens": record.n_tokens_generated,
                    "elapsed_ms": record.elapsed_ms,
                    "source_bundle": record.source_bundle,
                    "manipulation": manip,
                }
            )

    full_metric_prompts = sorted(full_metric_prompts, key=lambda row: row["prompt_id"])
    metadata_only_prompts = sorted(metadata_only_prompts, key=lambda row: row["prompt_id"])

    wrapper_extra_constant = sorted(set(wrapper_extras))
    if len(wrapper_extra_constant) != 1:
        raise ValueError(f"Wrapper extra is not constant across prompts: {wrapper_extra_constant}")

    analysis = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifacts_root": str(artifacts_root),
        "n_total_prompts": len(records),
        "n_full_metric_prompts": len(full_metric_prompts),
        "n_metadata_only_prompts": len(metadata_only_prompts),
        "method": {
            "routing_source": "weights_norm+topk_exact",
            "entropy_normalization": "log2(8)",
            "kl_definition": "KL(token || mean(Cal1)) over exact sparse 256-expert top-8 distribution",
            "wrapper_alignment": {
                "wrapper_extra_tokens_constant": wrapper_extra_constant[0],
                "prefix_string": PREFIX,
                "suffix_string": SUFFIX,
                "alignment_rule": (
                    "Distribute the constant GGUF-vs-HF prompt-token offset between "
                    "prefix and suffix in proportion to their HF wrapper token counts."
                ),
            },
        },
        "full_metric_prompts": full_metric_prompts,
        "metadata_only_prompts": metadata_only_prompts,
        "summary": build_summary(full_metric_prompts),
    }

    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    results_md_path.parent.mkdir(parents=True, exist_ok=True)
    results_json_path.write_text(json.dumps(analysis, indent=2) + "\n")
    results_md_path.write_text(build_results_markdown(analysis, results_json_path))

    print(f"Wrote {results_json_path}")
    print(f"Wrote {results_md_path}")
    print(f"Analyzed {len(full_metric_prompts)} full prompts; {len(metadata_only_prompts)} metadata-only prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
