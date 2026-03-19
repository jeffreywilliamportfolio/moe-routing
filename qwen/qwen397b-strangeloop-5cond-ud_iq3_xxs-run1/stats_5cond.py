#!/usr/bin/env python3
"""
Local stats analysis for qwen397b-strangeloop-5cond-ud_iq3_xxs-run1.

NOTE: The canonical Holm-Bonferroni p_holm values are those stored in
results_strangeloop_5cond_prefill_qwen.json under each pair's 'p_holm' field.
Those were computed by analyze_5cond.py at run time. The recomputed values
from this script use a different sort ordering and should NOT be used as the
source of truth. RESULTS.md draws from the JSON, not from this script.
"""
import json
import itertools
import numpy as np
from scipy.stats import wilcoxon

RESULTS_FILE = "results_strangeloop_5cond_prefill_qwen.json"
CONDITIONS = list("ABCDE")
COND_LABELS = {"A": "this", "B": "a", "C": "your", "D": "the", "E": "their"}

with open(RESULTS_FILE) as f:
    data = json.load(f)

rows = data["per_prompt"]
print(f"Loaded {len(rows)} prompts\n")

# Group by condition
by_cond = {c: [] for c in CONDITIONS}
for r in rows:
    by_cond[r["condition"]].append(r)

# ── Condition means ──────────────────────────────────────────────────────────
print("=== Condition Means ===")
print(f"{'Cond':<6} {'Label':<8} {'N':>4}  {'prefill_RE':>10}  {'last_tok_RE':>11}  {'KL_manip':>9}")
for c in CONDITIONS:
    rs = by_cond[c]
    re  = np.mean([r["prefill_re"]    for r in rs])
    lt  = np.mean([r["last_token_re"] for r in rs])
    kl  = np.mean([r["kl_manip_mean"] for r in rs])
    print(f"  {c:<4} {COND_LABELS[c]:<8} {len(rs):>4}  {re:>10.6f}  {lt:>11.6f}  {kl:>9.6f}")

# ── Pairwise Wilcoxon (Holm-Bonferroni) ─────────────────────────────────────
def wilcoxon_pairs(metric_key, label):
    pairs = list(itertools.combinations(CONDITIONS, 2))
    raw = []
    for a, b in pairs:
        xa = np.array([r[metric_key] for r in by_cond[a]])
        xb = np.array([r[metric_key] for r in by_cond[b]])
        stat, p = wilcoxon(xa, xb)
        diff = np.mean(xa) - np.mean(xb)
        n_agree = int(np.sum(xa > xb))
        raw.append((a, b, stat, p, diff, n_agree, len(xa)))

    # Holm-Bonferroni
    raw.sort(key=lambda x: x[3])
    m = len(raw)
    corrected = []
    for rank, (a, b, stat, p, diff, n_agree, n) in enumerate(raw):
        p_holm = min(1.0, p * (m - rank))
        corrected.append((a, b, stat, p, p_holm, diff, n_agree, n))

    print(f"\n=== {label} — Pairwise Wilcoxon (Holm-Bonferroni) ===")
    print(f"{'Pair':<8} {'mean_diff':>10}  {'A>B':>5}  {'p_raw':>10}  {'p_holm':>10}  {'sig':>4}")
    for a, b, stat, p, p_holm, diff, n_agree, n in corrected:
        sig = "***" if p_holm < 0.001 else ("**" if p_holm < 0.01 else ("*" if p_holm < 0.05 else "ns"))
        print(f"  {a}-{b}    {diff:>+10.6f}  {n_agree:>3}/{n}  {p:>10.4e}  {p_holm:>10.4e}  {sig:>4}")

    return corrected

re_tests = wilcoxon_pairs("prefill_re",    "All-token RE")
lt_tests = wilcoxon_pairs("last_token_re", "Last-token RE")
kl_tests = wilcoxon_pairs("kl_manip_mean", "KL-to-baseline (manip region)")

# ── A vs B summary (key comparison) ─────────────────────────────────────────
print("\n=== A vs B (this vs a) — key comparison ===")
for label, tests in [("prefill_re", re_tests), ("last_token_re", lt_tests), ("kl_manip_mean", kl_tests)]:
    row = next((r for r in tests if r[0] == "A" and r[1] == "B"), None)
    if row:
        a, b, stat, p_raw, p_holm, diff, n_agree, n = row
        print(f"  {label:<20} mean_diff={diff:+.6f}  A>B={n_agree}/{n}  p_raw={p_raw:.4e}  p_holm={p_holm:.4e}")

# ── RE ordering ──────────────────────────────────────────────────────────────
print("\n=== RE Ordering (high to low) ===")
means = {c: np.mean([r["prefill_re"] for r in by_cond[c]]) for c in CONDITIONS}
order = sorted(CONDITIONS, key=lambda c: means[c], reverse=True)
print("  prefill_RE:    " + " > ".join(f"{c}({COND_LABELS[c]})" for c in order))
lt_means = {c: np.mean([r["last_token_re"] for r in by_cond[c]]) for c in CONDITIONS}
lt_order = sorted(CONDITIONS, key=lambda c: lt_means[c], reverse=True)
print("  last_token_RE: " + " > ".join(f"{c}({COND_LABELS[c]})" for c in lt_order))
kl_means = {c: np.mean([r["kl_manip_mean"] for r in by_cond[c]]) for c in CONDITIONS}
kl_order = sorted(CONDITIONS, key=lambda c: kl_means[c], reverse=True)
print("  KL_manip:      " + " > ".join(f"{c}({COND_LABELS[c]})" for c in kl_order))

# ── Token counts (sanity check) ──────────────────────────────────────────────
print("\n=== Token Counts ===")
for c in CONDITIONS:
    toks = [r["n_prompt_tokens"] for r in by_cond[c]]
    print(f"  {c} ({COND_LABELS[c]}): mean={np.mean(toks):.1f}  min={min(toks)}  max={max(toks)}")
