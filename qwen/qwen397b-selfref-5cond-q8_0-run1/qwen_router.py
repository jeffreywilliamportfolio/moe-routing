#!/usr/bin/env python3
"""
Qwen3.5-397B-A17B routing reconstruction helpers.

The capture binary stores raw `ffn_moe_logits-*` tensors (PRE-softmax).
The Qwen3.5-MoE model reconstructs expert weights as:

1. Apply `softmax` to the full 512-dim raw logits
2. Select top-10 experts by **post-softmax probability**
3. **Renormalize** the selected 10 probabilities to sum to 1.0

Verified against:
  - HuggingFace transformers source (Qwen3_5MoeTopKRouter.forward):
      router_logits = F.linear(hidden_states, self.weight)
      router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)
      router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
      router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
  - llama.cpp b8123 src/models/qwen35moe.cpp:
      build_moe_ffn(..., LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX)
  - llama.cpp b8123 src/llama-graph.cpp confirms ffn_moe_logits is named
    BEFORE softmax is applied in the compute graph.

Shared expert: always active via sigmoid gate, independent of router.
Not captured in ffn_moe_logits. Our entropy measures routed decisions only.
"""
from __future__ import annotations

import numpy as np

N_EXPERTS = 512
TOP_K = 10
ENTROPY_MAX = np.log2(TOP_K)  # log2(10) = 3.3219...
RECONSTRUCTION_NAME = "softmax_then_topk10_renorm"


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def reconstruct_probs(logits: np.ndarray) -> np.ndarray:
    """Reconstruct Qwen3.5-MoE expert probability distribution.

    Input:  raw pre-softmax logits, shape [n_tokens, 512] or [512].
    Output: sparse probability distribution, same shape,
            only TOP_K=10 entries non-zero per row, row-sum = 1.0.
    """
    squeeze = logits.ndim == 1
    if squeeze:
        logits = logits[np.newaxis, :]

    logits = np.asarray(logits, dtype=np.float64)
    n_tokens, n_experts = logits.shape
    k = min(TOP_K, n_experts)

    # Step 1: softmax over full 512-dim
    full_probs = softmax(logits, axis=-1)

    # Step 2: select top-k by post-softmax probability
    topk_indices = np.argpartition(full_probs, -k, axis=-1)[:, -k:]
    rows = np.arange(n_tokens)[:, None]
    topk_probs = full_probs[rows, topk_indices]

    # Step 3: renormalize top-k to sum to 1.0
    topk_probs = topk_probs / topk_probs.sum(axis=-1, keepdims=True)

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
    """Dense 512-dim softmax proxy for KL computation (analysis only).

    This is NOT the model's actual routing distribution. Qwen3.5-MoE routes
    via softmax(512) -> topk(10) -> renormalize. This dense proxy provides
    full-support 512-dim vectors so that KL divergence is well-defined across
    tokens whose sparse top-10 expert sets may not overlap.
    """
    logits = np.asarray(logits, dtype=np.float64)
    return softmax(logits, axis=-1)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) per row, in bits. Both inputs must be 2D or broadcastable."""
    p = np.clip(p, 1e-30, None)
    q = np.clip(q, 1e-30, None)
    return np.sum(p * np.log2(p / q), axis=-1)
