#!/usr/bin/env python3
"""
Qwen3.5-35B-A3B routing reconstruction helpers.

The captured tensors are raw router logits over 256 routed experts. Qwen routes by:

1. Dense softmax over all routed experts
2. Top-k select the 8 largest probabilities
3. Renormalize the selected support to sum to 1

Entropy is therefore computed on the sparse top-8 routing distribution and
normalized by log2(8), not log2(256).
"""
from __future__ import annotations

import numpy as np

N_EXPERTS = 256
TOP_K = 8
ENTROPY_MAX = np.log2(TOP_K)
RECONSTRUCTION_NAME = "softmax_then_topk8_renorm"


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_full_probs(logits: np.ndarray) -> np.ndarray:
    """Dense full-support softmax proxy used for analysis and KL."""
    return softmax(logits, axis=-1)


def reconstruct_probs(logits: np.ndarray) -> np.ndarray:
    """Reconstruct sparse Qwen routing probabilities from raw logits."""
    squeeze = logits.ndim == 1
    if squeeze:
        logits = logits[np.newaxis, :]

    logits = np.asarray(logits, dtype=np.float64)
    dense_probs = softmax_full_probs(logits)
    n_tokens, n_experts = dense_probs.shape
    k = min(TOP_K, n_experts)

    topk_indices = np.argpartition(dense_probs, -k, axis=-1)[:, -k:]
    rows = np.arange(n_tokens)[:, None]
    topk_probs = dense_probs[rows, topk_indices]
    topk_probs /= np.sum(topk_probs, axis=-1, keepdims=True)

    probs = np.zeros_like(dense_probs, dtype=np.float64)
    probs[rows, topk_indices] = topk_probs

    if squeeze:
        probs = probs[0]
    return probs


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    return -np.sum(probs * np.log2(probs + 1e-30), axis=-1) / ENTROPY_MAX


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-30, None)
    q = np.clip(q, 1e-30, None)
    return np.sum(p * np.log2(p / q), axis=-1)
