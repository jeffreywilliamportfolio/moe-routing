#!/usr/bin/env python3
"""DeepSeek V3.1 routing reconstruction from captured router logits.

This reconstructs the part of the Hugging Face gate logic that is recoverable
from captured ``ffn_moe_logits`` tensors:

1. ``sigmoid(logits)``
2. DeepSeek ``noaux_tc`` group filtering (`n_group=8`, `topk_group=4`)
3. top-k expert selection (`num_experts_per_tok=8`)
4. renormalize selected expert weights to the simplex (`norm_topk_prob=true`)

The exact DeepSeek implementation also adds a learned ``e_score_correction_bias``
before expert choice. That parameter is not present in the captured router
logits, so the reconstruction here is a bias-free approximation.
"""

from __future__ import annotations

import numpy as np

N_ROUTED_EXPERTS = 256
N_GROUP = 8
TOPK_GROUP = 4
TOP_K = 8
EXPERTS_PER_GROUP = N_ROUTED_EXPERTS // N_GROUP
ENTROPY_MAX = np.log2(TOP_K)
RECONSTRUCTION_NAME = (
    "sigmoid_noaux_tc_group_filtered_topk_normalized"
    "_without_e_score_correction_bias"
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    positive = x64 >= 0
    out = np.empty_like(x64)
    out[positive] = 1.0 / (1.0 + np.exp(-x64[positive]))
    exp_x = np.exp(x64[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def reconstruct_probs(logits: np.ndarray) -> np.ndarray:
    """Approximate DeepSeek V3.1 routed expert probabilities from logits."""
    logits = np.asarray(logits)
    squeeze = logits.ndim == 1
    if squeeze:
        logits = logits[None, :]
    if logits.ndim != 2:
        raise ValueError(f"Expected 1D or 2D logits array, got shape {logits.shape}")
    if logits.shape[1] != N_ROUTED_EXPERTS:
        raise ValueError(
            f"Expected {N_ROUTED_EXPERTS} routed experts, got {logits.shape[1]}"
        )

    scores = _sigmoid(logits)
    flat_scores = scores.reshape(-1, N_GROUP, EXPERTS_PER_GROUP)

    # DeepSeek noaux_tc keeps only the top scoring groups before expert choice.
    top2_per_group = np.partition(flat_scores, -2, axis=-1)[:, :, -2:]
    group_scores = top2_per_group.sum(axis=-1)
    top_group_idx = np.argpartition(group_scores, -TOPK_GROUP, axis=-1)[:, -TOPK_GROUP:]

    group_mask = np.zeros_like(group_scores, dtype=bool)
    rows = np.arange(group_scores.shape[0])[:, None]
    group_mask[rows, top_group_idx] = True
    expert_mask = np.repeat(group_mask, EXPERTS_PER_GROUP, axis=1)

    candidate_scores = np.where(expert_mask, scores, 0.0)
    topk_idx = np.argpartition(candidate_scores, -TOP_K, axis=-1)[:, -TOP_K:]

    selected = np.zeros_like(scores)
    selected[rows, topk_idx] = scores[rows, topk_idx]

    totals = selected.sum(axis=-1, keepdims=True)
    totals = np.where(totals < 1e-30, 1e-30, totals)
    probs = selected / totals
    return probs[0] if squeeze else probs


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    return -np.sum(probs * np.log2(probs + 1e-30), axis=-1) / ENTROPY_MAX
