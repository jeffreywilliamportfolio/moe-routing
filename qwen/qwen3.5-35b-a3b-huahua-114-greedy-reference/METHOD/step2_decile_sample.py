#!/usr/bin/env python3
"""
Step 2 of the E114 verbalizer pipeline: decile-stratified sampling + same-token matched controls.

Inputs  (per run_id):
  analysis/<run_id>/step1/contexts.jsonl  — from Step 1

Outputs (per run_id):
  analysis/<run_id>/step2/
    sampled.json                — decile_counts + sampled contexts + matched negative controls
    sampling_diagnostics.json   — per-decile W_114 ranges, match-rate stats, prompt provenance

Why deciles (CLAUDE.md § Verbalizer methodology):
  Top-k-only sampling biases the labeler toward the strongest-obvious-pattern story and hides
  soft edges. We bin tokens into 10 deciles by W_114 at the PRIMARY_LAYER (L14) and sample
  from every decile including the zero-activation decile (index 0). The `decile_counts` vector
  (length 10) is required per row in the verbalizer table so starvation of any decile is
  visible on sight. Both prefill and trimmed-generation tokens feed the binning — the label
  describes what E114 fires on regardless of track.

Decile convention (CLAUDE.md explicit):
  - index 0: zero-activation — W_114 == 0.0 (E114 not in top-8). ALL such tokens are collected
    here regardless of count; we then sample from them.
  - indices 1..9: nonzero-W tokens split into 9 equal-count buckets by W_114 (quantile binning,
    ascending). Index 1 is the lowest-nonzero decile; index 9 is the highest-W decile.

Sampling within a decile:
  Shuffle with a fixed seed, take the first `--samples-per-decile`. If a decile has fewer
  tokens than requested, take all of them and record the shortfall in `decile_counts`. The
  schema requires declaring all 10 counts so the consumer can detect starvation.

Same-token-id matched negative controls (load-bearing discipline):
  For each sampled non-zero-W token (deciles 1..9), we also pull a MATCHED NEGATIVE: a
  different context from the same corpus where the SAME token_id appears but W_114 == 0.
  This is the control that lets the labeler (and Step 5) distinguish "this label describes
  what E114 fires on" from "this label describes what token X means." Decile-0 samples alone
  are a weaker control — they're token-distribution-biased (tokens that never fire E114 in
  any context are overrepresented). Matched negatives pin the comparison on "same lexical
  surface, different router decision."

  Algorithm: group all step1 contexts by token_id, filter to those with W_114_L14 == 0, for
  each positive sample look up same-token-id zeros and deterministically pick one (seeded RNG).
  If a token has no same-id zero-W peer in the corpus, no match is emitted for that positive
  and the gap is logged. Multi-prompt corpora will have better match coverage than single-prompt
  ones.

Token-count disparity across buckets:
  The zero-activation decile (index 0) typically dwarfs every other decile on a prompt where
  E114 is active (e.g., for the processing-hum probe, ~30% of tokens have W=0 at L14, which
  is ~70 tokens; each of the 9 nonzero deciles gets ~15 tokens). Sampling K per decile
  balances labeler exposure across the activation spectrum rather than the raw density.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

PRIMARY_LAYER = 14


def bin_tokens_into_deciles(
    W_values: np.ndarray,
    n_deciles: int = 10,
) -> np.ndarray:
    """
    Return an int array the same length as W_values, with each entry in [0, n_deciles-1].

    Decile 0 = exactly W == 0.0 (E114 not selected). The nonzero W values are binned into
    n_deciles - 1 quantile buckets indexed 1..n_deciles-1, ascending by W.
    """
    assert n_deciles >= 2
    out = np.zeros_like(W_values, dtype=np.int64)
    zero_mask = (W_values == 0.0)
    out[zero_mask] = 0

    nonzero_idx = np.where(~zero_mask)[0]
    if nonzero_idx.size == 0:
        return out
    nonzero_W = W_values[nonzero_idx]

    # Quantile-bin the nonzero values into (n_deciles - 1) buckets.
    # np.quantile with linear interpolation gives stable boundaries even when there are ties.
    n_nz_deciles = n_deciles - 1
    quantiles = np.linspace(0.0, 1.0, n_nz_deciles + 1)[1:-1]  # interior cut points
    if quantiles.size > 0:
        boundaries = np.quantile(nonzero_W, quantiles)
        bin_idx = np.digitize(nonzero_W, boundaries, right=False)  # 0..n_nz_deciles-1
    else:
        bin_idx = np.zeros_like(nonzero_W, dtype=np.int64)
    # Shift so nonzero deciles are 1..n_deciles-1
    out[nonzero_idx] = bin_idx + 1
    return out


def sample_from_decile(
    indices: list[int],
    k: int,
    rng: random.Random,
) -> list[int]:
    """Sample up to k indices from `indices` deterministically under rng."""
    if len(indices) <= k:
        return list(indices)
    pool = list(indices)
    rng.shuffle(pool)
    return pool[:k]


def build_same_token_zero_index(tokens: list[dict[str, Any]]) -> dict[int, list[int]]:
    """
    Group indices into `tokens` by token_id, keeping only those with W_114_L14 == 0.0.
    Returns token_id -> list of token-list indices.
    """
    idx: dict[int, list[int]] = {}
    for i, t in enumerate(tokens):
        if float(t[f"W_114_L{PRIMARY_LAYER}"]) == 0.0:
            idx.setdefault(int(t["token_id"]), []).append(i)
    return idx


def pick_matched_negative(
    positive_rec: dict[str, Any],
    zero_index: dict[int, list[int]],
    tokens: list[dict[str, Any]],
    rng: random.Random,
    used: set[int],
) -> dict[str, Any] | None:
    """
    For a single positive (non-zero-W) token record, return a record from the corpus with the
    SAME token_id but W_114_L14 == 0, chosen deterministically under rng. Prefers unused
    candidates to maximize distinct control coverage across the sampled set; if all candidates
    have been used already, reuses the earliest unused-order candidate (better to show the
    labeler a repeated negative than to drop the control entirely).

    Returns the matched record (or None if no same-token-id zero exists in the corpus).
    """
    token_id = int(positive_rec["token_id"])
    candidates = zero_index.get(token_id, [])
    if not candidates:
        return None

    # Prefer unused candidates. Shuffle with rng so the choice is deterministic but not biased
    # toward any particular position.
    unused = [i for i in candidates if i not in used]
    pool = unused if unused else list(candidates)
    rng.shuffle(pool)
    chosen_idx = pool[0]
    used.add(chosen_idx)
    out = dict(tokens[chosen_idx])
    out["matched_negative"] = True
    out["matched_negative_for"] = {
        "positive_global_idx": positive_rec["global_idx"],
        "positive_decile_idx": positive_rec["decile_idx"],
        "positive_W_114_L14": positive_rec[f"W_114_L{PRIMARY_LAYER}"],
    }
    return out


def process_prompt(
    prompt_record: dict[str, Any],
    samples_per_decile: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (sampled record for Step 3, diagnostics row)."""
    prompt_id = prompt_record["prompt_id"]
    tokens = prompt_record["tokens"]
    n = len(tokens)

    # Extract the W_114 @ PRIMARY_LAYER vector aligned with `tokens`.
    W = np.array([t[f"W_114_L{PRIMARY_LAYER}"] for t in tokens], dtype=np.float64)
    deciles = bin_tokens_into_deciles(W, n_deciles=10)

    # Index tokens by decile.
    by_decile: list[list[int]] = [[] for _ in range(10)]
    for i, d in enumerate(deciles):
        by_decile[int(d)].append(i)

    # Seeded RNG: bias by prompt_id so different prompts get different shuffles, but a given
    # (seed, prompt_id) pair always produces the same sample set.
    rng = random.Random(seed + hash(prompt_id) % (2**31))
    sampled_by_decile: list[list[int]] = []
    for d in range(10):
        sampled_by_decile.append(sample_from_decile(by_decile[d], samples_per_decile, rng))

    decile_counts = [len(s) for s in sampled_by_decile]

    # Flatten into a single sampled list, preserving decile index in each record.
    sampled_tokens: list[dict[str, Any]] = []
    for d in range(10):
        for i in sampled_by_decile[d]:
            rec = dict(tokens[i])
            rec["decile_idx"] = d
            rec["decile_W_range"] = {
                "W": float(W[i]),
            }
            sampled_tokens.append(rec)

    # Sort flat list by (decile descending, global_idx) so high-activation contexts are adjacent
    # in the output — easier for the labeler to eyeball.
    sampled_tokens.sort(key=lambda r: (-r["decile_idx"], r["global_idx"]))

    # -------- Matched same-token-id negative controls (CLAUDE.md discipline) --------
    # For every non-zero-W sample, try to find a same-token-id W=0 companion from anywhere in
    # the corpus. These are the critical control for the labeler: "this label describes what
    # E114 fires on (not what token X means)."
    zero_index = build_same_token_zero_index(tokens)
    match_rng = random.Random(seed + 997 + hash(prompt_id) % (2**31))
    used_zero_indices: set[int] = set()
    matched_negatives: list[dict[str, Any]] = []
    matched_gap_token_ids: list[int] = []
    n_positive = 0
    for rec in sampled_tokens:
        if rec["decile_idx"] == 0:
            continue
        n_positive += 1
        neg = pick_matched_negative(rec, zero_index, tokens, match_rng, used_zero_indices)
        if neg is None:
            matched_gap_token_ids.append(int(rec["token_id"]))
            continue
        matched_negatives.append(neg)

    match_rate = (
        (n_positive - len(matched_gap_token_ids)) / n_positive if n_positive > 0 else 0.0
    )

    sampled_record = {
        "run_id": prompt_record["run_id"],
        "prompt_id": prompt_id,
        "context_trim_mode": prompt_record.get("context_trim_mode"),
        "context_radius_tokens": prompt_record.get("context_radius_tokens"),
        "target_expert": prompt_record.get("target_expert"),
        "primary_layer": PRIMARY_LAYER,
        "samples_per_decile_requested": samples_per_decile,
        "decile_counts": decile_counts,
        "n_tokens_total": n,
        "n_tokens_prompt": prompt_record["n_tokens_prompt"],
        "n_tokens_generated_trimmed": prompt_record["n_tokens_generated_trimmed"],
        "sampled_tokens": sampled_tokens,
        "matched_negatives": matched_negatives,
        "matched_negative_stats": {
            "n_positives_needing_match": n_positive,
            "n_matched": len(matched_negatives),
            "match_rate": match_rate,
            "unmatched_token_ids": matched_gap_token_ids,
            "unique_zero_w_tokens_available": len(zero_index),
        },
    }

    # Diagnostics: per-decile W range and population count
    diag_deciles = []
    for d in range(10):
        pop = [int(i) for i in by_decile[d]]
        if pop:
            pop_W = W[pop]
            d_entry = {
                "decile_idx": d,
                "population": len(pop),
                "sampled": decile_counts[d],
                "W_min": float(pop_W.min()),
                "W_max": float(pop_W.max()),
                "W_mean": float(pop_W.mean()),
            }
        else:
            d_entry = {
                "decile_idx": d,
                "population": 0,
                "sampled": 0,
                "W_min": None,
                "W_max": None,
                "W_mean": None,
            }
        diag_deciles.append(d_entry)

    diagnostics = {
        "prompt_id": prompt_id,
        "n_tokens_total": n,
        "decile_counts": decile_counts,
        "deciles": diag_deciles,
        "matched_negative_stats": sampled_record["matched_negative_stats"],
    }
    return sampled_record, diagnostics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", required=True, help="analysis/<run_id>/ directory (Step 1 already wrote step1/ here)")
    p.add_argument("--samples-per-decile", type=int, default=10, help="target sample count per decile (may be truncated if decile smaller)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible sampling")
    args = p.parse_args()

    analysis = Path(args.analysis_dir)
    step1_contexts = analysis / "step1" / "contexts.jsonl"
    if not step1_contexts.exists():
        print(f"ERROR: no Step 1 output at {step1_contexts}", file=sys.stderr)
        return 2

    out = analysis / "step2"
    out.mkdir(parents=True, exist_ok=True)

    # Stream-read step1 contexts
    all_prompts: list[dict[str, Any]] = []
    with step1_contexts.open() as f:
        for line in f:
            line = line.strip()
            if line:
                all_prompts.append(json.loads(line))
    print(f"Step 2: decile-sampling {len(all_prompts)} prompt(s) from {step1_contexts}")

    sampled_records = []
    diagnostics_rows = []
    for rec in all_prompts:
        sampled, diag = process_prompt(rec, args.samples_per_decile, args.seed)
        sampled_records.append(sampled)
        diagnostics_rows.append(diag)
        counts = sampled["decile_counts"]
        starved = [d for d, c in enumerate(counts) if c == 0]
        mstats = sampled["matched_negative_stats"]
        print(f"  {rec['prompt_id']}: decile_counts={counts}  total_sampled={sum(counts)}  starved_deciles={starved}")
        print(f"    matched negatives: {mstats['n_matched']}/{mstats['n_positives_needing_match']} "
              f"positives matched (match_rate={mstats['match_rate']:.1%}); "
              f"{len(mstats['unmatched_token_ids'])} unmatched token_ids, "
              f"{mstats['unique_zero_w_tokens_available']} unique zero-W tokens available in corpus")

    sampled_path = out / "sampled.json"
    sampled_path.write_text(json.dumps({
        "samples_per_decile_requested": args.samples_per_decile,
        "seed": args.seed,
        "primary_layer": PRIMARY_LAYER,
        "prompts": sampled_records,
    }, indent=2))

    diag_path = out / "sampling_diagnostics.json"
    diag_path.write_text(json.dumps({"prompts": diagnostics_rows}, indent=2))

    print(f"\nwrote:")
    print(f"  {sampled_path}  ({sampled_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {diag_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
