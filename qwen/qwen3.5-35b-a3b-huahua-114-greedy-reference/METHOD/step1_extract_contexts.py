#!/usr/bin/env python3
"""
Step 1 of the E114 verbalizer pipeline: extract per-token contexts for every captured prompt.

Inputs  (per run_id):
  raw/<run_id>/capture_manifest.json      — authoritative inventory of succeeded prompts
  raw/<run_id>/<prompt_cell>/
    prompt_tokens.json       — tokenized prompt, with char spans
    generated_tokens.json    — greedy/sampled generated tokens (includes hallucinated tail)
    router/ffn_moe_logits-{13,14,15}.npy
    metadata.txt             — includes n_tokens_prompt, n_tokens_generated
    generated_text.txt

Outputs (per run_id):
  analysis/<run_id>/step1/
    contexts.jsonl           — one object per prompt: token-level W/S/Q/rank at L13/L14/L15
                               with a ±N token context snippet per token (default N=20)
    summary.json             — per-prompt trim diagnostics + pooled W/S/Q over TRIMMED tokens
    contexts_preview.tsv     — human-readable sampler of the first 60 rows per prompt

Applies the HauhauCS `<|im_end|>` TRIM rule (CLAUDE.md § Verbalizer methodology): trim generated
tokens at the first occurrence of the literal 6-token sequence [27, 91, 316, 6018, 91, 29]
(the spelled-out `<|im_end|>`). Everything past that boundary is hallucinated continuation and
MUST NOT feed the labeler.

Does NOT do sampling or ranking — that's Step 2. This step just produces the full per-token
context inventory so Step 2 can decile-stratify over a clean (trimmed) token set.

Iterates ONLY over capture_manifest.json entries with status == "succeeded" — per the
invariant declared in CLAUDE.md, `ls raw/<run_id>/*/` is NOT authoritative.

Target expert is hard-coded to E114 because this repo is scoped to that single expert (see
PLAN.md and CLAUDE.md purpose). A later run for a different expert forks this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TARGET_EXPERT = 114
TARGET_LAYERS = (13, 14, 15)
PRIMARY_LAYER = 14  # formation layer per CLAUDE.md § "Why layer 14 (not 26)"

# HauhauCS hallucinated-<|im_end|> spelled-out token sequence.
# Any later occurrence of this exact 6-token tuple in generated_tokens_ids marks the boundary
# past which the model starts hallucinating fresh turns. Source: CLAUDE.md § Verbalizer
# methodology — "HauhauCS `<|im_end|>` hallucination — TRIM before sampling".
HAUHAU_IMEND_SEQUENCE: tuple[int, ...] = (27, 91, 316, 6018, 91, 29)


def load_reconstruct_probs():
    """Import qwen_router.reconstruct_probs from scripts/ without a package install."""
    script_dir = Path(__file__).resolve().parent
    path = script_dir / "qwen_router.py"
    spec = importlib.util.spec_from_file_location("qwen_router", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load qwen_router from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reconstruct_probs


def find_trim_point(generated_token_ids: list[int]) -> int:
    """
    Return the index of the first token position where the HauhauCS `<|im_end|>` 6-token
    sequence BEGINS in generated_token_ids, or -1 if it does not appear.

    Trimmed generation = generated_token_ids[:trim_point]. Tokens at [trim_point:] are
    the spelled-out `<|im_end|>` plus hallucinated continuation; discarded for all
    downstream ranking and labeling per CLAUDE.md.
    """
    seq = HAUHAU_IMEND_SEQUENCE
    L = len(seq)
    for i in range(0, len(generated_token_ids) - L + 1):
        if tuple(generated_token_ids[i : i + L]) == seq:
            return i
    return -1


@dataclass
class TrackRow:
    """Lightweight per-token record before it becomes a dict for JSON serialization."""

    global_idx: int
    track: str  # "prefill" | "generation"
    track_idx: int
    token_id: int
    piece: str


def build_track_rows(
    prompt_tokens: list[dict[str, Any]],
    generated_tokens_trimmed: list[dict[str, Any]],
) -> list[TrackRow]:
    """Unified flat list: prefill rows first, then trimmed-generation rows."""
    rows: list[TrackRow] = []
    for i, tok in enumerate(prompt_tokens):
        rows.append(
            TrackRow(
                global_idx=i,
                track="prefill",
                track_idx=i,
                token_id=int(tok["token_id"]),
                piece=str(tok["piece"]),
            )
        )
    offset = len(prompt_tokens)
    for i, tok in enumerate(generated_tokens_trimmed):
        rows.append(
            TrackRow(
                global_idx=offset + i,
                track="generation",
                track_idx=i,
                token_id=int(tok["token_id"]),
                piece=str(tok["piece"]),
            )
        )
    return rows


def context_snippet(rows: list[TrackRow], center: int, radius: int) -> tuple[str, str]:
    """
    Return (pre, post) strings: the `radius` tokens immediately before `center` concatenated,
    and the `radius` tokens immediately after `center` concatenated. Pieces are joined with
    no separator (token pieces are byte-accurate renderings including leading spaces).
    """
    pre_start = max(0, center - radius)
    post_end = min(len(rows), center + 1 + radius)
    pre = "".join(rows[i].piece for i in range(pre_start, center))
    post = "".join(rows[i].piece for i in range(center + 1, post_end))
    return pre, post


def expert_rank(logits_1d: np.ndarray, expert_idx: int) -> int:
    """
    Rank of `expert_idx` among 256 experts, descending by dense softmax probability.
    0 = highest-probability expert. Computed directly on logits (argsort on logits is equivalent
    to argsort on softmax). Ties are broken by numpy's default argsort (stable, index-ascending).
    """
    order = np.argsort(-logits_1d, kind="stable")
    pos = np.where(order == expert_idx)[0]
    if pos.size == 0:
        return -1
    return int(pos[0])


def analyze_prompt(
    prompt_cell: Path,
    run_id: str,
    reconstruct_probs,
    context_radius: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Process one prompt cell. Returns (per-prompt context record, per-prompt summary)."""

    # Metadata
    meta: dict[str, str] = {}
    for line in (prompt_cell / "metadata.txt").read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            meta[k.strip()] = v.strip()
    n_tok_prompt = int(meta["n_tokens_prompt"])
    n_tok_gen = int(meta["n_tokens_generated"])
    prompt_id = meta.get("prompt_id", prompt_cell.name)

    prompt_tokens = json.loads((prompt_cell / "prompt_tokens.json").read_text())
    generated_tokens = json.loads((prompt_cell / "generated_tokens.json").read_text())

    assert len(prompt_tokens) == n_tok_prompt, (
        f"{prompt_cell}: prompt_tokens.json has {len(prompt_tokens)} rows, "
        f"metadata says n_tokens_prompt={n_tok_prompt}"
    )
    assert len(generated_tokens) == n_tok_gen, (
        f"{prompt_cell}: generated_tokens.json has {len(generated_tokens)} rows, "
        f"metadata says n_tokens_generated={n_tok_gen}"
    )

    gen_ids = [int(t["token_id"]) for t in generated_tokens]
    trim_idx = find_trim_point(gen_ids)
    trim_found = trim_idx >= 0
    if trim_found:
        gen_trimmed = generated_tokens[:trim_idx]
    else:
        gen_trimmed = generated_tokens  # no hallucination detected; use full gen

    n_gen_trimmed = len(gen_trimmed)
    total_rows = n_tok_prompt + n_gen_trimmed
    rows = build_track_rows(prompt_tokens, gen_trimmed)

    # Load router logits per layer and compute per-token probs + E114 metrics.
    # The logits .npy rows are [prompt || full generation], so we slice:
    #   rows [0 : n_tok_prompt]                                   — prefill
    #   rows [n_tok_prompt : n_tok_prompt + n_gen_trimmed]        — trimmed generation
    per_layer: dict[int, dict[str, np.ndarray]] = {}
    for il in TARGET_LAYERS:
        logits_path = prompt_cell / "router" / f"ffn_moe_logits-{il}.npy"
        logits = np.load(logits_path)
        assert logits.shape[0] == n_tok_prompt + n_tok_gen, (
            f"{logits_path}: n_rows={logits.shape[0]}, "
            f"expected {n_tok_prompt + n_tok_gen}"
        )
        sliced = np.vstack([logits[:n_tok_prompt], logits[n_tok_prompt : n_tok_prompt + n_gen_trimmed]])
        assert sliced.shape[0] == total_rows
        probs = reconstruct_probs(sliced)  # [total_rows, 256]
        W = probs[:, TARGET_EXPERT].astype(np.float64)
        S = (W > 0).astype(np.int8)
        ranks = np.array([expert_rank(sliced[i], TARGET_EXPERT) for i in range(total_rows)], dtype=np.int32)
        per_layer[il] = {"W": W, "S": S, "rank": ranks}

    # Build per-token JSON records. Residual tensor paths stored as pointers (not loaded);
    # the labeler does not need them, and loading would bloat the artifact by ~30 MB per prompt.
    token_records: list[dict[str, Any]] = []
    for r in rows:
        rec: dict[str, Any] = {
            "global_idx": r.global_idx,
            "track": r.track,
            "track_idx": r.track_idx,
            "token_id": r.token_id,
            "piece": r.piece,
        }
        pre, post = context_snippet(rows, r.global_idx, context_radius)
        rec["context_pre"] = pre
        rec["context_post"] = post
        for il in TARGET_LAYERS:
            W_val = float(per_layer[il]["W"][r.global_idx])
            S_val = int(per_layer[il]["S"][r.global_idx])
            rank_val = int(per_layer[il]["rank"][r.global_idx])
            # Q = renormalized weight when selected; undefined otherwise. Store as null so
            # downstream decile binning treats non-selected tokens as "zero-activation decile".
            Q_val = W_val if S_val == 1 else None
            rec[f"W_114_L{il}"] = W_val
            rec[f"S_114_L{il}"] = S_val
            rec[f"Q_114_L{il}"] = Q_val
            rec[f"rank_114_L{il}"] = rank_val
        token_records.append(rec)

    # Per-prompt summary: pooled W/S/Q over trimmed tokens per layer per track.
    def pool(mask: np.ndarray, W: np.ndarray, S: np.ndarray) -> dict[str, float]:
        n = int(mask.sum())
        if n == 0:
            return {"W_mean": 0.0, "S_mean": 0.0, "Q_mean": 0.0, "n_tokens": 0, "n_selected": 0}
        Wm = W[mask]
        Sm = S[mask].astype(np.float64)
        sel_mask = Sm > 0
        n_sel = int(sel_mask.sum())
        Q_mean = float(Wm[sel_mask].mean()) if n_sel > 0 else 0.0
        return {
            "W_mean": float(Wm.mean()),
            "S_mean": float(Sm.mean()),
            "Q_mean": Q_mean,
            "n_tokens": n,
            "n_selected": n_sel,
        }

    prefill_mask = np.zeros(total_rows, dtype=bool)
    prefill_mask[:n_tok_prompt] = True
    gen_mask = ~prefill_mask

    pooled: dict[int, dict[str, Any]] = {}
    for il in TARGET_LAYERS:
        W = per_layer[il]["W"]
        S = per_layer[il]["S"]
        pooled[il] = {
            "prefill": pool(prefill_mask, W, S),
            "generation_trimmed": pool(gen_mask, W, S),
        }

    # W_114 = S_114 * Q_114 identity residual at machine epsilon — defensive check.
    max_resid = 0.0
    for il in TARGET_LAYERS:
        row = pooled[il]["generation_trimmed"]
        if row["n_tokens"] > 0:
            max_resid = max(max_resid, abs(row["W_mean"] - row["S_mean"] * row["Q_mean"]))

    per_prompt_record = {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "context_trim_mode": "trim_at_literal_imend",
        "context_radius_tokens": context_radius,
        "n_tokens_prompt": n_tok_prompt,
        "n_tokens_generated_raw": n_tok_gen,
        "n_tokens_generated_trimmed": n_gen_trimmed,
        "hauhau_imend_trim_found": trim_found,
        "hauhau_imend_trim_idx": trim_idx,
        "target_expert": TARGET_EXPERT,
        "target_layers": list(TARGET_LAYERS),
        "tokens": token_records,
    }

    per_prompt_summary = {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "n_tokens_prompt": n_tok_prompt,
        "n_tokens_generated_raw": n_tok_gen,
        "n_tokens_generated_trimmed": n_gen_trimmed,
        "hauhau_imend_trim_found": trim_found,
        "hauhau_imend_trim_idx": trim_idx,
        "pooled_WSQ_E114": pooled,
        "WSQ_identity_residual_max": max_resid,
    }

    return per_prompt_record, per_prompt_summary


def write_preview_tsv(per_prompt: dict[str, Any], path: Path, n_rows: int = 60) -> None:
    """Writable sampler so a human can eyeball the output before Step 2 runs."""
    fields = [
        "global_idx",
        "track",
        "track_idx",
        "token_id",
        "piece",
        "W_114_L13",
        "W_114_L14",
        "W_114_L15",
        "rank_114_L14",
        "context_pre_tail",
        "context_post_head",
    ]
    with path.open("w") as f:
        f.write("# prompt_id=" + per_prompt["prompt_id"] + "\n")
        f.write("\t".join(fields) + "\n")
        for rec in per_prompt["tokens"][:n_rows]:
            row = [
                str(rec["global_idx"]),
                rec["track"],
                str(rec["track_idx"]),
                str(rec["token_id"]),
                repr(rec["piece"]),
                f"{rec['W_114_L13']:.6f}",
                f"{rec['W_114_L14']:.6f}",
                f"{rec['W_114_L15']:.6f}",
                str(rec["rank_114_L14"]),
                repr(rec["context_pre"][-40:]),
                repr(rec["context_post"][:40]),
            ]
            f.write("\t".join(row) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", required=True, help="raw/<run_id>/ directory containing capture_manifest.json")
    p.add_argument("--analysis-dir", required=True, help="analysis/<run_id>/ directory to write step1 output into")
    p.add_argument("--context-radius", type=int, default=20, help="tokens before and after each center token for context snippet")
    args = p.parse_args()

    raw = Path(args.raw_dir)
    manifest_path = raw / "capture_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no manifest at {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())

    run_id = raw.name  # by convention
    out = Path(args.analysis_dir) / "step1"
    out.mkdir(parents=True, exist_ok=True)

    succeeded = [p for p in manifest["prompts"] if p["status"] == "succeeded"]
    if not succeeded:
        print(f"ERROR: no succeeded prompts in manifest", file=sys.stderr)
        return 2
    print(f"Step 1: processing {len(succeeded)} succeeded prompt(s) from {raw}")

    reconstruct_probs = load_reconstruct_probs()

    contexts_path = out / "contexts.jsonl"
    summary_path = out / "summary.json"
    per_prompt_summaries = []
    with contexts_path.open("w") as ctx_f:
        for entry in succeeded:
            safe_id = entry["safe_id"]
            cell = raw / safe_id
            if not cell.exists():
                print(f"  WARN: manifest lists safe_id={safe_id} but {cell} is missing; skipping", file=sys.stderr)
                continue
            print(f"  processing {safe_id}")
            rec, summary = analyze_prompt(cell, run_id, reconstruct_probs, args.context_radius)
            ctx_f.write(json.dumps(rec) + "\n")
            per_prompt_summaries.append(summary)
            preview_path = out / f"contexts_preview_{safe_id}.tsv"
            write_preview_tsv(rec, preview_path)

    summary = {
        "run_id": run_id,
        "context_trim_mode": "trim_at_literal_imend",
        "context_radius_tokens": args.context_radius,
        "target_expert": TARGET_EXPERT,
        "target_layers": list(TARGET_LAYERS),
        "n_prompts_processed": len(per_prompt_summaries),
        "per_prompt": per_prompt_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nwrote:")
    print(f"  {contexts_path}  ({contexts_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {summary_path}")
    for entry in succeeded:
        preview = out / f"contexts_preview_{entry['safe_id']}.tsv"
        if preview.exists():
            print(f"  {preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
