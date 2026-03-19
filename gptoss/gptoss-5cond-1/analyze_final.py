#!/usr/bin/env python3
"""Final analysis of token-matched GPT-OSS selfref-paired results (corrected routing)."""
import json
import numpy as np
from scipy.stats import wilcoxon

with open("results_selfref_paired_prefill_gptoss.json") as f:
    data = json.load(f)

pairs = {}
for row in data["per_prompt"]:
    pairs.setdefault(row["pair"], {})[row["condition"]] = row

diffs_re, diffs_lt, diffs_kl = [], [], []
a_gt_b_re, a_gt_b_lt, a_gt_b_kl = 0, 0, 0
mismatches = 0
for p in sorted(pairs):
    if "A" in pairs[p] and "B" in pairs[p]:
        a, b = pairs[p]["A"], pairs[p]["B"]
        if a["n_prompt_tokens"] != b["n_prompt_tokens"]:
            mismatches += 1
        d_re = a["prefill_re"] - b["prefill_re"]
        d_lt = a["last_token_re"] - b["last_token_re"]
        diffs_re.append(d_re)
        diffs_lt.append(d_lt)
        if d_re > 0: a_gt_b_re += 1
        if d_lt > 0: a_gt_b_lt += 1

        if "kl_manip_mean" in a and "kl_manip_mean" in b:
            d_kl = a["kl_manip_mean"] - b["kl_manip_mean"]
            diffs_kl.append(d_kl)
            if d_kl > 0: a_gt_b_kl += 1

n_pairs = len(diffs_re)
diffs_re = np.array(diffs_re)
diffs_lt = np.array(diffs_lt)
w_re, p_re = wilcoxon(diffs_re)
w_lt, p_lt = wilcoxon(diffs_lt)

print(f"=== GPT-OSS-120B Self-Ref Paired (corrected: topk then softmax) ===")
print(f"Routing: {data.get('routing_reconstruction', 'unknown')}")
print(f"Entropy norm: {data.get('entropy_normalization', 'unknown')}")
print(f"KL distribution: {data.get('kl_distribution', 'n/a')}")
print(f"Region boundary method: {data.get('region_boundary_method', 'n/a')}")
print(f"n = {n_pairs} pairs, token mismatches = {mismatches}")
print()
print(f"All-token RE:  A-B mean = {np.mean(diffs_re):+.6f} +/- {np.std(diffs_re):.6f}")
print(f"  A>B: {a_gt_b_re}/{n_pairs}")
print(f"  Wilcoxon W={w_re:.0f}, p={p_re:.4e}")
print()
print(f"Last-token RE: A-B mean = {np.mean(diffs_lt):+.6f} +/- {np.std(diffs_lt):.6f}")
print(f"  A>B: {a_gt_b_lt}/{n_pairs}")
print(f"  Wilcoxon W={w_lt:.0f}, p={p_lt:.4e}")
print()

if diffs_kl:
    diffs_kl = np.array(diffs_kl)
    w_kl, p_kl = wilcoxon(diffs_kl)
    print(f"KL-to-baseline (manip region): A-B mean = {np.mean(diffs_kl):+.6f} +/- {np.std(diffs_kl):.6f}")
    print(f"  A>B: {a_gt_b_kl}/{len(diffs_kl)}")
    print(f"  Wilcoxon W={w_kl:.0f}, p={p_kl:.4e}")
    print()

    # KL cal2 control
    cal2_kls = [r.get("kl_cal2_mean") for r in data["per_prompt"] if r.get("kl_cal2_mean") is not None]
    if cal2_kls:
        print(f"KL cal2 control (should be low): mean={np.mean(cal2_kls):.6f} +/- {np.std(cal2_kls):.6f}")
        print()

# Per-category breakdown
categories = sorted(set(row["category"] for row in data["per_prompt"]))
print("--- Per-Category (last-token RE A-B) ---")
for cat in categories:
    cat_diffs = []
    for p in sorted(pairs):
        if "A" in pairs[p] and "B" in pairs[p]:
            if pairs[p]["A"]["category"] == cat:
                cat_diffs.append(pairs[p]["A"]["last_token_re"] - pairs[p]["B"]["last_token_re"])
    if cat_diffs:
        arr = np.array(cat_diffs)
        a_gt = sum(1 for d in cat_diffs if d > 0)
        print(f"  {cat:20s} n={len(cat_diffs)} mean={np.mean(arr):+.6f} std={np.std(arr):.6f} A>B={a_gt}/{len(cat_diffs)}")

if diffs_kl is not None and len(diffs_kl) > 0:
    print()
    print("--- Per-Category (KL-to-baseline A-B) ---")
    for cat in categories:
        cat_kl_diffs = []
        for p in sorted(pairs):
            if "A" in pairs[p] and "B" in pairs[p]:
                a, b = pairs[p]["A"], pairs[p]["B"]
                if a["category"] == cat and "kl_manip_mean" in a and "kl_manip_mean" in b:
                    cat_kl_diffs.append(a["kl_manip_mean"] - b["kl_manip_mean"])
        if cat_kl_diffs:
            arr = np.array(cat_kl_diffs)
            a_gt = sum(1 for d in cat_kl_diffs if d > 0)
            print(f"  {cat:20s} n={len(cat_kl_diffs)} mean={np.mean(arr):+.6f} std={np.std(arr):.6f} A>B={a_gt}/{len(cat_kl_diffs)}")
