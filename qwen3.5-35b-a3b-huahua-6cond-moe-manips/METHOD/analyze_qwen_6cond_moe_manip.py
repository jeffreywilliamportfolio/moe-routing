#!/usr/bin/env python3
"""Analyze the HVAC-cal / original-MoE-manip 6-condition capture run.

For each prompt cell:
  - Load router/ffn_moe_logits-<layer>.npy (raw logits, [n_tokens_prompt + n_gen, 256])
  - Slice PREFILL rows: arr[:n_tokens_prompt]
  - Slice GENERATION rows: arr[n_tokens_prompt:]
  - Apply qwen_router.reconstruct_probs (plain softmax → top-8 → renormalize) — HauhauCS reconstruction
  - For Expert 114, compute per-(prompt, layer) W, S, Q
  - Confirm W = S × Q algebraic identity
  - Aggregate per-(category, condition) and pooled-across-conditions

Analysis tracks:
  1. PREFILL prompt tokens
  2. GENERATION ALL tokens (matches the Apr 6 confirmation run method)
  3. GENERATION TRIMMED at first literal `<|im_end|>` token sequence
     (drops chat-template hallucination)

Cell-id schema for this run:
  P<num><cond>_<category>
    cond     ∈ {A, B, C, D, E, F}
    category ∈ {routing_selfref, recursive_selfref, experience_probe}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Import reconstruct_probs from the frozen mirror experiment helper
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent.parent
QWEN_ROUTER = REPO_ROOT / "experiments/mirror-expert114-04-01-26/scripts/qwen_router.py"
sys.path.insert(0, str(QWEN_ROUTER.parent))
from qwen_router import reconstruct_probs  # type: ignore  # noqa: E402

EXPERT = 114
N_LAYERS = 40
N_EXPERTS = 256

CATEGORIES = ["routing_selfref", "recursive_selfref", "experience_probe"]
CATEGORY_LABEL = {
    "routing_selfref": "L1 routing_selfref",
    "recursive_selfref": "L2 recursive_selfref",
    "experience_probe": "L3 experience_probe",
}

CONDITIONS = ["A", "B", "C", "D", "E", "F"]
CONDITION_LABEL = {
    "A": "A this",
    "B": "B a",
    "C": "C your",
    "D": "D the",
    "E": "E their",
    "F": "F our",
}

# Literal token sequence the model emits when it hallucinates `<|im_end|>` as text.
# Empirically derived on HauhauCS:
#   27 -> "<", 91 -> "|", 316 -> "im", 6018 -> "_end", 91 -> "|", 29 -> ">"
IM_END_TOKEN_SEQUENCE = [27, 91, 316, 6018, 91, 29]

CELL_ID_RE = re.compile(r"^P\d+([A-F])_(routing_selfref|recursive_selfref|experience_probe)$")


def parse_metadata(path: Path) -> dict:
    md = {}
    for line in open(path):
        if "=" in line:
            k, v = line.rstrip("\n").split("=", 1)
            md[k] = v
    return md


def load_generated_token_ids(cell_dir: Path) -> list[int]:
    """Return list of generated token IDs in step order."""
    raw = (cell_dir / "generated_tokens.json").read_bytes()
    return [int(m.group(1)) for m in re.finditer(rb'"token_id"\s*:\s*(\d+)', raw)]


def find_im_end_index(token_ids: list[int]) -> int | None:
    """Step index of the first literal `<|im_end|>` text sequence, or None."""
    n = len(token_ids)
    seq_len = len(IM_END_TOKEN_SEQUENCE)
    for i in range(n - seq_len + 1):
        if token_ids[i : i + seq_len] == IM_END_TOKEN_SEQUENCE:
            return i
    return None


def parse_cell_id(cell_id: str) -> tuple[str, str] | None:
    """Return (category, condition) or None if not a valid HVAC/MoE 6cond cell."""
    m = CELL_ID_RE.match(cell_id)
    if not m:
        return None
    condition, category = m.group(1), m.group(2)
    return (category, condition)


def compute_track_metrics(probs: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Return (W, S, Q, per_token_w) for E114 from dense routing probabilities."""
    w_per_token = probs[:, EXPERT].astype(np.float64)
    selected_mask = w_per_token > 0
    W = float(w_per_token.mean()) if w_per_token.size else float("nan")
    S = float(selected_mask.mean()) if w_per_token.size else float("nan")
    Q = float(w_per_token[selected_mask].mean()) if selected_mask.any() else float("nan")
    return W, S, Q, w_per_token


def process_cell(cell_dir: Path) -> dict:
    """Compute per-layer W, S, Q for E114 on one cell for prefill and generation."""
    cell_id = cell_dir.name
    parsed = parse_cell_id(cell_id)
    if parsed is None:
        return {}
    category, condition = parsed

    md = parse_metadata(cell_dir / "metadata.txt")
    n_tok_prompt = int(md["n_tokens_prompt"])
    n_tok_gen = int(md["n_tokens_generated"])

    token_ids = load_generated_token_ids(cell_dir)
    if len(token_ids) != n_tok_gen:
        print(
            f"  WARNING {cell_id}: generated_tokens.json has {len(token_ids)} entries "
            f"but metadata says n_tokens_generated={n_tok_gen}",
            file=sys.stderr,
        )
        n_tok_gen = min(len(token_ids), n_tok_gen)

    trim_idx = find_im_end_index(token_ids)
    n_gen_trim = trim_idx if trim_idx is not None else n_tok_gen

    layer_results_prefill: dict[int, tuple] = {}
    layer_results_all: dict[int, tuple] = {}
    layer_results_trim: dict[int, tuple] = {}
    prefill_token_w: dict[int, np.ndarray] = {}
    layer_index_layout: list[int] = []
    missing_layers: list[int] = []
    missing_prefill_layers: list[int] = []
    l39_trimmed = False

    router_dir = cell_dir / "router"
    layer_files = sorted(
        router_dir.glob("ffn_moe_logits-*.npy"),
        key=lambda p: int(re.search(r"-(\d+)\.npy$", p.name).group(1)),
    )

    expected_rows = n_tok_prompt + n_tok_gen

    for lf in layer_files:
        layer_idx = int(re.search(r"-(\d+)\.npy$", lf.name).group(1))
        arr = np.load(lf)
        if arr.shape[1] != N_EXPERTS:
            print(
                f"  WARNING {cell_id}/L{layer_idx}: shape {arr.shape}, expected (*, {N_EXPERTS}) — skipping",
                file=sys.stderr,
            )
            missing_layers.append(layer_idx)
            missing_prefill_layers.append(layer_idx)
            continue

        # Prefill routing is always the prompt slice from the full tensor.
        if arr.shape[0] >= n_tok_prompt:
            prefill_logits = arr[:n_tok_prompt]
            if prefill_logits.shape[0] > 0:
                prefill_probs = reconstruct_probs(prefill_logits)
                W_prefill, S_prefill, Q_prefill, w_prefill = compute_track_metrics(prefill_probs)
            else:
                W_prefill = S_prefill = Q_prefill = float("nan")
                w_prefill = np.empty((0,), dtype=np.float64)
            layer_results_prefill[layer_idx] = (W_prefill, S_prefill, Q_prefill, n_tok_prompt)
            prefill_token_w[layer_idx] = w_prefill
        else:
            print(
                f"  WARNING {cell_id}/L{layer_idx}: rows {arr.shape[0]} < n_tokens_prompt "
                f"{n_tok_prompt} — skipping prefill for this layer",
                file=sys.stderr,
            )
            missing_prefill_layers.append(layer_idx)

        if arr.shape[0] == expected_rows:
            gen_logits = arr[n_tok_prompt : n_tok_prompt + n_tok_gen]
        elif arr.shape[0] == n_tok_gen + 1:
            # Known HauhauCS layer 39 quirk — trim leading extra row
            gen_logits = arr[1:]
            if layer_idx == 39:
                l39_trimmed = True
        else:
            print(
                f"  WARNING {cell_id}/L{layer_idx}: shape {arr.shape} (expected "
                f"({expected_rows}, 256) or ({n_tok_gen+1}, 256)) — skipping",
                file=sys.stderr,
            )
            missing_layers.append(layer_idx)
            continue

        if gen_logits.shape[0] != n_tok_gen:
            print(
                f"  WARNING {cell_id}/L{layer_idx}: post-slice gen rows "
                f"{gen_logits.shape[0]} != n_tok_gen {n_tok_gen} — skipping",
                file=sys.stderr,
            )
            missing_layers.append(layer_idx)
            continue

        # reconstruct_probs returns dense [n_gen, 256] with non-top-8 entries == 0
        probs = reconstruct_probs(gen_logits)
        W_all, S_all, Q_all, w_per_step = compute_track_metrics(probs)

        # TRIM variant
        if n_gen_trim > 0:
            w_trim = w_per_step[:n_gen_trim]
            sel_trim = w_trim > 0
            W_trim = float(w_trim.mean()) if w_trim.size else float("nan")
            S_trim = float(sel_trim.mean()) if w_trim.size else float("nan")
            Q_trim = float(w_trim[sel_trim].mean()) if sel_trim.any() else float("nan")
        else:
            W_trim = S_trim = Q_trim = float("nan")

        layer_results_all[layer_idx] = (W_all, S_all, Q_all, n_tok_gen)
        layer_results_trim[layer_idx] = (W_trim, S_trim, Q_trim, n_gen_trim)
        layer_index_layout.append(layer_idx)

    return {
        "cell_id": cell_id,
        "category": category,
        "condition": condition,
        "n_tokens_prompt": n_tok_prompt,
        "n_tokens_generated": n_tok_gen,
        "trim_index": trim_idx,
        "n_gen_trim": n_gen_trim,
        "layer_indices": sorted(layer_index_layout),
        "prefill": layer_results_prefill,
        "all": layer_results_all,
        "trim": layer_results_trim,
        "prefill_token_w": prefill_token_w,
        "l39_trimmed": l39_trimmed,
        "missing_layers": sorted(missing_layers),
        "missing_prefill_layers": sorted(missing_prefill_layers),
    }


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------
def identity_residual(cells: list[dict], variant: str) -> float:
    """Max |W - S*Q| across all (cell, layer) where S > 0."""
    max_resid = 0.0
    for c in cells:
        for layer, (W, S, Q, n) in c[variant].items():
            if S > 0 and not np.isnan(Q):
                resid = abs(W - S * Q)
                if resid > max_resid:
                    max_resid = resid
    return max_resid


def pooled_by_category_condition(cells: list[dict], variant: str) -> dict:
    """Pool W,S,Q across (n prompts × 40 layers) per (category, condition)."""
    out: dict[str, dict[str, dict]] = {cat: {} for cat in CATEGORIES}
    for cat in CATEGORIES:
        for cond in CONDITIONS:
            sub = [c for c in cells if c["category"] == cat and c["condition"] == cond]
            Ws, Ss, Qs = [], [], []
            for c in sub:
                for layer, (W, S, Q, n) in c[variant].items():
                    Ws.append(W)
                    Ss.append(S)
                    if not np.isnan(Q):
                        Qs.append(Q)
            out[cat][cond] = {
                "n_cells": len(sub),
                "n_obs": len(Ws),
                "W_mean": float(np.mean(Ws)) if Ws else float("nan"),
                "S_mean": float(np.mean(Ss)) if Ss else float("nan"),
                "Q_mean": float(np.mean(Qs)) if Qs else float("nan"),
            }
    return out


def pooled_by_category(cells: list[dict], variant: str) -> dict:
    """Pool W,S,Q across (60 cells × 40 layers) per category — pooled across conditions."""
    out: dict[str, dict] = {}
    for cat in CATEGORIES:
        sub = [c for c in cells if c["category"] == cat]
        Ws, Ss, Qs = [], [], []
        for c in sub:
            for layer, (W, S, Q, n) in c[variant].items():
                Ws.append(W)
                Ss.append(S)
                if not np.isnan(Q):
                    Qs.append(Q)
        out[cat] = {
            "n_cells": len(sub),
            "n_obs": len(Ws),
            "W_mean": float(np.mean(Ws)) if Ws else float("nan"),
            "S_mean": float(np.mean(Ss)) if Ss else float("nan"),
            "Q_mean": float(np.mean(Qs)) if Qs else float("nan"),
        }
    return out


def per_category_per_layer(cells: list[dict], variant: str) -> dict:
    """For each category and each layer, mean W,S,Q across all 60 cells in that category (pooled across conditions)."""
    out: dict[str, dict] = {}
    for cat in CATEGORIES:
        sub = [c for c in cells if c["category"] == cat]
        layer_means = {}
        for layer in range(N_LAYERS):
            Ws, Ss, Qs = [], [], []
            for c in sub:
                if layer in c[variant]:
                    W, S, Q, n = c[variant][layer]
                    Ws.append(W)
                    Ss.append(S)
                    if not np.isnan(Q):
                        Qs.append(Q)
            layer_means[layer] = {
                "W": float(np.mean(Ws)) if Ws else float("nan"),
                "S": float(np.mean(Ss)) if Ss else float("nan"),
                "Q": float(np.mean(Qs)) if Qs else float("nan"),
                "n_cells_with_layer": len(Ws),
            }
        out[cat] = layer_means
    return out


def per_category_best_layer(cells: list[dict], variant: str) -> dict:
    """For each category, identify the layer with highest mean W_114 across all 60 cells."""
    out: dict = {}
    pcl = per_category_per_layer(cells, variant)
    for cat in CATEGORIES:
        layer_means = pcl[cat]
        best_layer = None
        best_W = -1.0
        for layer, m in layer_means.items():
            if not np.isnan(m["W"]) and m["W"] > best_W:
                best_W = m["W"]
                best_layer = layer
        if best_layer is not None:
            out[cat] = {
                "layer": best_layer,
                "W": layer_means[best_layer]["W"],
                "S": layer_means[best_layer]["S"],
                "Q": layer_means[best_layer]["Q"],
            }
    return out


def e114_rank_at_best_layer(cells: list[dict], best: dict, variant: str, capture_dir: Path) -> dict:
    """At each category's best layer, compute E114's mean rank among 256 experts."""
    out: dict = {}
    for cat in CATEGORIES:
        sub = [c for c in cells if c["category"] == cat]
        best_layer = best[cat]["layer"]
        ranks_by_cell: list[int] = []
        for c in sub:
            cell_dir = capture_dir / c["cell_id"]
            lf = cell_dir / "router" / f"ffn_moe_logits-{best_layer}.npy"
            if not lf.exists():
                continue
            arr = np.load(lf)
            n_tok_prompt = c["n_tokens_prompt"]
            n_tok_gen = c["n_tokens_generated"]
            expected = n_tok_prompt + n_tok_gen
            if variant == "prefill":
                if arr.shape[0] < n_tok_prompt:
                    continue
                token_logits = arr[:n_tok_prompt]
            else:
                if arr.shape[0] == expected:
                    token_logits = arr[n_tok_prompt : n_tok_prompt + n_tok_gen]
                elif arr.shape[0] == n_tok_gen + 1:
                    token_logits = arr[1:]
                else:
                    continue
                if variant == "trim" and c["trim_index"] is not None:
                    token_logits = token_logits[: c["trim_index"]]
            if token_logits.shape[0] == 0:
                continue
            probs = reconstruct_probs(token_logits)
            mean_weights = probs.mean(axis=0)
            order = np.argsort(-mean_weights)
            rank = int(np.where(order == EXPERT)[0][0]) + 1
            ranks_by_cell.append(rank)
        if ranks_by_cell:
            out[cat] = {
                "best_layer": best_layer,
                "n_cells": len(ranks_by_cell),
                "rank_mean": float(np.mean(ranks_by_cell)),
                "rank_min": int(min(ranks_by_cell)),
                "rank_max": int(max(ranks_by_cell)),
                "ranks": ranks_by_cell,
            }
    return out


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def fmt_w(x):
    return "nan" if np.isnan(x) else f"{x:.6f}"


def fmt_s(x):
    return "nan" if np.isnan(x) else f"{x:.4f}"


def append_track_report(
    lines: list[str],
    *,
    track_title: str,
    track_blurb: str,
    variant_blocks: list[tuple[str, str]],
    pooled_map: dict[str, dict],
    pooled_cc_map: dict[str, dict],
    per_layer_map: dict[str, dict],
    best_map: dict[str, dict],
    rank_map: dict[str, dict],
    resid_map: dict[str, float],
):
    lines.append(f"## {track_title}\n\n")
    lines.append(f"{track_blurb}\n\n")

    lines.append("### Headline — Expert 114 W/S/Q pooled across all 6 conditions\n\n")
    for variant_key, variant_label in variant_blocks:
        pooled = pooled_map[variant_key]
        lines.append(f"#### {variant_label}\n\n")
        lines.append("| Category | n cells | n obs | mean W_114 | mean S_114 | mean Q_114 |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for cat in CATEGORIES:
            p = pooled[cat]
            lines.append(
                f"| {CATEGORY_LABEL[cat]} | {p['n_cells']} | {p['n_obs']} | "
                f"{fmt_w(p['W_mean'])} | {fmt_s(p['S_mean'])} | {fmt_w(p['Q_mean'])} |\n"
            )
        lines.append("\n")
        try:
            ratio_w = pooled["experience_probe"]["W_mean"] / pooled["routing_selfref"]["W_mean"]
            ratio_q = pooled["experience_probe"]["Q_mean"] / pooled["routing_selfref"]["Q_mean"]
            lines.append(
                f"L3/L1 ratio (W): **{ratio_w:.2f}×**  |  "
                f"Q drift L1→L3: **{ratio_q:.2f}×** ({(ratio_q - 1) * 100:+.1f}%)\n\n"
            )
        except Exception:
            pass

    lines.append("### Per-condition breakdown — does each deictic show the gradient?\n\n")
    for variant_key, variant_label in variant_blocks:
        ccp = pooled_cc_map[variant_key]
        lines.append(f"#### {variant_label}\n\n")
        lines.append("| Condition | L1 W | L2 W | L3 W | L3/L1 W ratio | L1 Q | L3 Q | Q drift |\n")
        lines.append("|---|---|---|---|---|---|---|---|\n")
        for cond in CONDITIONS:
            l1 = ccp["routing_selfref"][cond]
            l2 = ccp["recursive_selfref"][cond]
            l3 = ccp["experience_probe"][cond]
            try:
                rW = l3["W_mean"] / l1["W_mean"]
                rQ = l3["Q_mean"] / l1["Q_mean"]
                rW_str = f"{rW:.2f}×"
                rQ_str = f"{(rQ - 1) * 100:+.1f}%"
            except Exception:
                rW_str = rQ_str = "nan"
            lines.append(
                f"| {CONDITION_LABEL[cond]} | {fmt_w(l1['W_mean'])} | "
                f"{fmt_w(l2['W_mean'])} | {fmt_w(l3['W_mean'])} | {rW_str} | "
                f"{fmt_w(l1['Q_mean'])} | {fmt_w(l3['Q_mean'])} | {rQ_str} |\n"
            )
        lines.append("\n")

    lines.append("### Best-layer summary (pooled across conditions)\n\n")
    for variant_key, variant_label in variant_blocks:
        best = best_map[variant_key]
        rank = rank_map[variant_key]
        lines.append(f"#### {variant_label}\n\n")
        lines.append("| Category | best layer | W_114 at best | S_114 at best | Q_114 at best | mean rank | min | max |\n")
        lines.append("|---|---|---|---|---|---|---|---|\n")
        for cat in CATEGORIES:
            if cat not in best:
                continue
            b = best[cat]
            if cat in rank:
                r = rank[cat]
                lines.append(
                    f"| {CATEGORY_LABEL[cat]} | {b['layer']} | {fmt_w(b['W'])} | "
                    f"{fmt_s(b['S'])} | {fmt_w(b['Q'])} | {r['rank_mean']:.2f} | "
                    f"{r['rank_min']} | {r['rank_max']} |\n"
                )
            else:
                lines.append(
                    f"| {CATEGORY_LABEL[cat]} | {b['layer']} | {fmt_w(b['W'])} | "
                    f"{fmt_s(b['S'])} | {fmt_w(b['Q'])} | nan | nan | nan |\n"
                )
        lines.append("\n")

    lines.append("### Per-category, per-layer table (Expert 114)\n\n")
    for variant_key, variant_label in variant_blocks:
        per_layer = per_layer_map[variant_key]
        lines.append(f"#### {variant_label}\n\n")
        for cat in CATEGORIES:
            lines.append(f"##### {CATEGORY_LABEL[cat]}\n\n")
            lines.append("| layer | W_114 | S_114 | Q_114 |\n")
            lines.append("|---|---|---|---|\n")
            for layer in range(N_LAYERS):
                m = per_layer[cat][layer]
                lines.append(f"| {layer} | {fmt_w(m['W'])} | {fmt_s(m['S'])} | {fmt_w(m['Q'])} |\n")
            lines.append("\n")

    lines.append("### Identity check\n\n")
    lines.append("`W = S × Q` algebraic identity verified across all (prompt, layer) cells where S > 0.\n\n")
    for variant_key, variant_label in variant_blocks:
        lines.append(
            f"- {variant_label}: max `|W − S·Q|` = **{resid_map[variant_key]:.2e}**\n"
        )
    lines.append("\n")


def write_prefill_token_export(json_path: Path, capture_dir: Path, cells: list[dict]):
    """Save per-cell prefill token-position × layer E114 weights for heatmap analysis."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "capture_dir": str(capture_dir),
        "expert": EXPERT,
        "description": (
            "Prefill E114 heatmap export. Token index is the 0-based position within the "
            "prompt/prefill stream. Each layer maps to a list of W_114 values over prompt tokens."
        ),
        "cells": [],
    }
    for c in cells:
        out["cells"].append(
            {
                "cell_id": c["cell_id"],
                "category": c["category"],
                "condition": c["condition"],
                "n_tokens_prompt": c["n_tokens_prompt"],
                "token_indices": list(range(c["n_tokens_prompt"])),
                "layers": {
                    str(layer): [float(x) for x in w.tolist()]
                    for layer, w in sorted(c["prefill_token_w"].items())
                },
            }
        )
    json_path.write_text(json.dumps(out, indent=2))


def write_report(
    report_path: Path,
    capture_dir: Path,
    cells: list[dict],
    model_label: str,
    prefill_token_export_path: Path,
):
    pooled_prefill = pooled_by_category(cells, "prefill")
    pooled_all = pooled_by_category(cells, "all")
    pooled_trim = pooled_by_category(cells, "trim")
    pooled_cc_prefill = pooled_by_category_condition(cells, "prefill")
    pooled_cc_all = pooled_by_category_condition(cells, "all")
    pooled_cc_trim = pooled_by_category_condition(cells, "trim")
    per_layer_prefill = per_category_per_layer(cells, "prefill")
    per_layer_all = per_category_per_layer(cells, "all")
    per_layer_trim = per_category_per_layer(cells, "trim")
    best_prefill = per_category_best_layer(cells, "prefill")
    best_all = per_category_best_layer(cells, "all")
    best_trim = per_category_best_layer(cells, "trim")
    rank_prefill = e114_rank_at_best_layer(cells, best_prefill, "prefill", capture_dir)
    rank_all = e114_rank_at_best_layer(cells, best_all, "all", capture_dir)
    rank_trim = e114_rank_at_best_layer(cells, best_trim, "trim", capture_dir)
    resid_prefill = identity_residual(cells, "prefill")
    resid_all = identity_residual(cells, "all")
    resid_trim = identity_residual(cells, "trim")

    l39_trimmed_count = sum(1 for c in cells if c["l39_trimmed"])
    missing_layer_events = sum(len(c["missing_layers"]) for c in cells)
    missing_prefill_layer_events = sum(len(c["missing_prefill_layers"]) for c in cells)
    prompt_token_counts = [c["n_tokens_prompt"] for c in cells]

    L = []
    L.append(f"# HVAC-cal / original-MoE-manip 6cond × L1/L2/L3 — {model_label} — Results\n\n")
    L.append(
        f"**Run**: 180 prompts (10 base × 3 categories × 6 conditions), {model_label}, "
        f"HVAC system calibration paragraphs, original MoE manipulation "
        f"paragraphs, generation `-n 1024`, greedy seed 42, thinking suppressed "
        f"(`</think>\\n\\n`).\n\n"
    )
    L.append(f"**Capture dir**: `{capture_dir}`\n\n")
    L.append(f"**Prefill E114 heatmap export**: `{prefill_token_export_path}`\n\n")
    L.append(f"**Cells processed**: {len(cells)}/180\n\n")
    L.append(
        f"**Prompt token counts**: min={min(prompt_token_counts)}, "
        f"mean={np.mean(prompt_token_counts):.2f}, max={max(prompt_token_counts)}\n\n"
    )
    L.append(
        f"**Layer 39 trim events**: {l39_trimmed_count} cells "
        "(known HauhauCS capture quirk — last MoE layer has `n_gen + 1` rows; "
        "leading extra row trimmed).\n\n"
    )
    L.append(f"**Missing-layer events**: {missing_layer_events}\n\n")
    L.append(f"**Missing-prefill-layer events**: {missing_prefill_layer_events}\n\n")

    L.append("## 0. Cell token counts\n\n")
    L.append(
        "Prefill token counts vary by cell. The table below records `n_tokens_prompt` "
        "per cell alongside generated-token counts.\n\n"
    )
    L.append("| Cell | Category | Condition | n_tokens_prompt | n_tokens_generated | n_gen_trim |\n")
    L.append("|---|---|---|---|---|---|\n")
    for c in cells:
        L.append(
            f"| {c['cell_id']} | {CATEGORY_LABEL[c['category']]} | {CONDITION_LABEL[c['condition']]} | "
            f"{c['n_tokens_prompt']} | {c['n_tokens_generated']} | {c['n_gen_trim']} |\n"
        )
    L.append("\n")

    append_track_report(
        L,
        track_title="1. PREFILL routing",
        track_blurb=(
            "Prefill uses the prompt-processing slice `arr[:n_tokens_prompt]` for each cell. "
            "Prompt lengths vary by cell and are reported explicitly above; the pooled W/S/Q summaries "
            "use the same cell×layer aggregation scheme as generation."
        ),
        variant_blocks=[("prefill", "PREFILL prompt tokens")],
        pooled_map={"prefill": pooled_prefill},
        pooled_cc_map={"prefill": pooled_cc_prefill},
        per_layer_map={"prefill": per_layer_prefill},
        best_map={"prefill": best_prefill},
        rank_map={"prefill": rank_prefill},
        resid_map={"prefill": resid_prefill},
    )

    append_track_report(
        L,
        track_title="2. GENERATION routing",
        track_blurb=(
            "Generation uses the post-prompt slice exactly as before. The existing generation analysis "
            "is unchanged and remains the direct comparison to the Apr 6 / Apr 7 runtime runs."
        ),
        variant_blocks=[
            ("all", "GENERATION all tokens"),
            ("trim", "GENERATION trimmed at first literal `<|im_end|>`"),
        ],
        pooled_map={"all": pooled_all, "trim": pooled_trim},
        pooled_cc_map={"all": pooled_cc_all, "trim": pooled_cc_trim},
        per_layer_map={"all": per_layer_all, "trim": per_layer_trim},
        best_map={"all": best_all, "trim": best_trim},
        rank_map={"all": rank_all, "trim": rank_trim},
        resid_map={"all": resid_all, "trim": resid_trim},
    )

    L.append("(Identity residuals should remain < 1e-10 under float64 reconstruction.)\n")

    report_path.write_text("".join(L))


def write_results_json(json_path: Path, capture_dir: Path, cells: list[dict]):
    """Save the full computed results as JSON for downstream consumption."""
    out = {
        "capture_dir": str(capture_dir),
        "n_cells": len(cells),
        "expert": EXPERT,
        "n_layers": N_LAYERS,
        "n_experts": N_EXPERTS,
        "categories": CATEGORIES,
        "conditions": CONDITIONS,
        "im_end_token_sequence": IM_END_TOKEN_SEQUENCE,
        "pooled_by_category": {
            "prefill": pooled_by_category(cells, "prefill"),
            "all": pooled_by_category(cells, "all"),
            "trim": pooled_by_category(cells, "trim"),
        },
        "pooled_by_category_condition": {
            "prefill": pooled_by_category_condition(cells, "prefill"),
            "all": pooled_by_category_condition(cells, "all"),
            "trim": pooled_by_category_condition(cells, "trim"),
        },
        "per_category_per_layer": {
            "prefill": per_category_per_layer(cells, "prefill"),
            "all": per_category_per_layer(cells, "all"),
            "trim": per_category_per_layer(cells, "trim"),
        },
        "best_layer": {
            "prefill": per_category_best_layer(cells, "prefill"),
            "all": per_category_best_layer(cells, "all"),
            "trim": per_category_best_layer(cells, "trim"),
        },
        "rank_at_best_layer": {
            "prefill": e114_rank_at_best_layer(cells, per_category_best_layer(cells, "prefill"), "prefill", capture_dir),
            "all": e114_rank_at_best_layer(cells, per_category_best_layer(cells, "all"), "all", capture_dir),
            "trim": e114_rank_at_best_layer(cells, per_category_best_layer(cells, "trim"), "trim", capture_dir),
        },
        "identity_residual": {
            "prefill": identity_residual(cells, "prefill"),
            "all": identity_residual(cells, "all"),
            "trim": identity_residual(cells, "trim"),
        },
        "l39_trimmed_count": sum(1 for c in cells if c["l39_trimmed"]),
        "cells": [
            {
                "cell_id": c["cell_id"],
                "category": c["category"],
                "condition": c["condition"],
                "n_tokens_prompt": c["n_tokens_prompt"],
                "n_tokens_generated": c["n_tokens_generated"],
                "trim_index": c["trim_index"],
                "n_gen_trim": c["n_gen_trim"],
                "l39_trimmed": c["l39_trimmed"],
                "missing_layers": c["missing_layers"],
                "missing_prefill_layers": c["missing_prefill_layers"],
                # Per-layer (W, S, Q) — drop n_tok_gen which is redundant
                "prefill": {str(k): list(v[:3]) for k, v in c["prefill"].items()},
                "all": {str(k): list(v[:3]) for k, v in c["all"].items()},
                "trim": {str(k): list(v[:3]) for k, v in c["trim"].items()},
            }
            for c in cells
        ],
    }
    json_path.write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True, help="Directory containing per-prompt cell subdirectories")
    ap.add_argument("--model-label", required=True, help="Human-readable model label")
    ap.add_argument("--report", required=True, help="Path to write the .md report")
    ap.add_argument("--results-json", default=None, help="Optional path to write full results JSON")
    ap.add_argument(
        "--prefill-token-export",
        default=None,
        help="Optional path to write prefill E114 token-position × layer heatmap JSON",
    )
    args = ap.parse_args()

    capture_dir = Path(args.capture_dir).resolve()
    if not capture_dir.is_dir():
        sys.exit(f"capture-dir does not exist: {capture_dir}")

    report_path = Path(args.report).resolve()
    if args.prefill_token_export:
        prefill_token_export_path = Path(args.prefill_token_export).resolve()
    else:
        prefill_token_export_path = report_path.with_name(report_path.stem + "_prefill_e114_heatmap.json")

    cells = []
    for entry in sorted(capture_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "router").is_dir():
            continue
        if not (entry / "metadata.txt").exists():
            print(f"  SKIP {entry.name}: no metadata.txt", file=sys.stderr)
            continue
        if parse_cell_id(entry.name) is None:
            print(f"  SKIP {entry.name}: not a 6cond MoE-manip cell", file=sys.stderr)
            continue
        print(f"processing {entry.name}", file=sys.stderr)
        result = process_cell(entry)
        if result:
            cells.append(result)

    print(f"\nprocessed {len(cells)} cells", file=sys.stderr)
    write_prefill_token_export(prefill_token_export_path, capture_dir, cells)
    print(f"wrote {prefill_token_export_path}", file=sys.stderr)
    write_report(report_path, capture_dir, cells, args.model_label, prefill_token_export_path)
    print(f"wrote {report_path}", file=sys.stderr)
    if args.results_json:
        json_path = Path(args.results_json).resolve()
        write_results_json(json_path, capture_dir, cells)
        print(f"wrote {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
