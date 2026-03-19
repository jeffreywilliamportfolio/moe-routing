#!/usr/bin/env python3
"""
GPT-OSS 120B routing reconstruction helpers.

The capture binary stores raw `ffn_moe_logits-*` tensors. The GPT-OSS model
(openai/gpt-oss, GptOssForCausalLM) reconstructs expert weights as:

1. Select top-4 experts by **raw logit value** (no activation applied)
2. Apply `softmax` to only those 4 selected logits
3. Use those 4-dim weights for expert combination

Verified against openai/gpt-oss source (gpt_oss/torch/model.py):

    g = self.gate(t)
    experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
    expert_weights = torch.nn.functional.softmax(experts.values, dim=1)

No group filtering, no sigmoid, no shared experts, no bias correction.
"""
from __future__ import annotations

import numpy as np

N_EXPERTS = 128
TOP_K = 4
ENTROPY_MAX = np.log2(TOP_K)  # 2.0
RECONSTRUCTION_NAME = "topk_raw_logit_then_softmax_4"


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def reconstruct_probs(logits: np.ndarray) -> np.ndarray:
    """Reconstruct GPT-OSS expert probability distribution.

    Input:  raw logits, shape [n_tokens, 128] or [128].
    Output: sparse probability distribution, same shape,
            only TOP_K=4 entries non-zero per row, row-sum = 1.0.
    """
    squeeze = logits.ndim == 1
    if squeeze:
        logits = logits[np.newaxis, :]

    logits = np.asarray(logits, dtype=np.float64)
    n_tokens, n_experts = logits.shape
    k = min(TOP_K, n_experts)

    # Select top-k by raw logit value (no activation first)
    topk_indices = np.argpartition(logits, -k, axis=-1)[:, -k:]
    rows = np.arange(n_tokens)[:, None]
    topk_logits = logits[rows, topk_indices]

    # Softmax on only the selected logits
    topk_probs = softmax(topk_logits, axis=-1)

    # Place back into sparse full-size array
    probs = np.zeros_like(logits, dtype=np.float64)
    probs[rows, topk_indices] = topk_probs

    if squeeze:
        probs = probs[0]
    return probs


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute routing entropy normalized to [0, 1].

    RE = -sum(p * log2(p + eps)) / log2(TOP_K)

    Returns 1.0 when all TOP_K selected experts have equal weight,
    and approaches 0.0 when a single expert dominates.
    """
    probs = np.asarray(probs, dtype=np.float64)
    return -np.sum(probs * np.log2(probs + 1e-30), axis=-1) / ENTROPY_MAX


def softmax_full_probs(logits: np.ndarray) -> np.ndarray:
    """Dense 128-dim softmax proxy for KL computation (analysis only).

    This is NOT the model's actual routing distribution. GPT-OSS routes
    via topk(4) then softmax(4-dim). This dense proxy provides full-support
    128-dim vectors so that KL divergence is well-defined across tokens
    whose sparse top-4 expert sets may not overlap.
    """
    logits = np.asarray(logits, dtype=np.float64)
    return softmax(logits, axis=-1)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) per row, in bits. Both inputs must be 2D or broadcastable."""
    p = np.clip(p, 1e-30, None)
    q = np.clip(q, 1e-30, None)
    return np.sum(p * np.log2(p / q), axis=-1)
