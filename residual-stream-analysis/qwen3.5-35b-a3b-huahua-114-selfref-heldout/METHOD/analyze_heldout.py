#!/usr/bin/env python3
"""
Pilot-mode analysis of a heldout fire/nofire capture.

Inputs:
  raw/<heldout_run_id>/                     — capture_residuals output (capture_manifest.json + per-prompt dirs)
  heldout_classes.tsv                       — prompt_id \\t predicted_class sidecar

Computes, per prompt:
  mean and stddev of W_114 at L14 on the trimmed generation track only
  (HauhauCS <|im_end|> 6-token TRIM applied, prefill excluded)

Selects top-2 fire + top-2 nofire by within-class mean W_114, plots per-token W_114 series for each.

Writes:
  analysis/<heldout_run_id>/heldout_stats.tsv
  analysis/<heldout_run_id>/heldout_timeseries_top4.png

Prints:
  grouped summary (mean-of-means, stddev-of-means) for fire vs nofire
  per-prompt table
  one-line read on separation
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

TARGET_EXPERT = 114
PRIMARY_LAYER = 14
HAUHAU_IMEND_SEQ: tuple[int, ...] = (27, 91, 316, 6018, 91, 29)


def load_reconstruct_probs():
    path = Path(__file__).resolve().parent / "qwen_router.py"
    spec = importlib.util.spec_from_file_location("qwen_router", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod.reconstruct_probs


def find_trim_point(gen_ids: list[int]) -> int:
    seq = HAUHAU_IMEND_SEQ
    L = len(seq)
    for i in range(len(gen_ids) - L + 1):
        if tuple(gen_ids[i : i + L]) == seq:
            return i
    return -1


def w_series_for_prompt(prompt_dir: Path, reconstruct_probs) -> tuple[np.ndarray, int, int]:
    """
    Return (W_114 per trimmed-gen token at L14, n_prompt, trim_idx_or_-1).
    W_114 is zero where E114 was not selected.
    """
    meta: dict[str, str] = {}
    for line in (prompt_dir / "metadata.txt").read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            meta[k.strip()] = v.strip()
    n_prompt = int(meta["n_tokens_prompt"])
    n_gen = int(meta["n_tokens_generated"])

    gen_tokens = json.loads((prompt_dir / "generated_tokens.json").read_text())
    gen_ids = [int(t["token_id"]) for t in gen_tokens]
    trim_idx = find_trim_point(gen_ids)
    n_gen_trimmed = trim_idx if trim_idx >= 0 else n_gen

    logits = np.load(prompt_dir / "router" / f"ffn_moe_logits-{PRIMARY_LAYER}.npy")
    assert logits.shape[0] == n_prompt + n_gen, (
        f"{prompt_dir}: n_rows={logits.shape[0]} vs n_prompt+n_gen={n_prompt + n_gen}"
    )
    if n_gen_trimmed == 0:
        return np.zeros(0, dtype=np.float64), n_prompt, trim_idx

    gen_logits = logits[n_prompt : n_prompt + n_gen_trimmed]
    probs = reconstruct_probs(gen_logits)
    W = probs[:, TARGET_EXPERT].astype(np.float64)
    return W, n_prompt, trim_idx


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", required=True, help="raw/<heldout_run_id>/ directory")
    p.add_argument("--classes-tsv", required=True, help="prompt_id\\tpredicted_class sidecar")
    p.add_argument("--analysis-dir", required=True, help="analysis/<heldout_run_id>/ output directory")
    args = p.parse_args()

    raw = Path(args.raw_dir)
    manifest = json.loads((raw / "capture_manifest.json").read_text())
    succeeded = [e for e in manifest["prompts"] if e["status"] == "succeeded"]
    print(f"manifest: {len(succeeded)}/{manifest['run_summary']['n_prompts_processed']} succeeded")

    classes: dict[str, str] = {}
    for line in Path(args.classes_tsv).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        classes[parts[0]] = parts[1]
    missing = [e["prompt_id"] for e in succeeded if e["prompt_id"] not in classes]
    if missing:
        print(f"ERROR: prompt_ids in manifest without a class: {missing}", file=sys.stderr)
        return 2

    reconstruct_probs = load_reconstruct_probs()

    per_prompt: list[dict] = []
    series: dict[str, np.ndarray] = {}
    for entry in succeeded:
        pid = entry["prompt_id"]
        cell = raw / entry["safe_id"]
        W, n_prompt, trim_idx = w_series_for_prompt(cell, reconstruct_probs)
        per_prompt.append({
            "prompt_id": pid,
            "predicted_class": classes[pid],
            "n_prompt": n_prompt,
            "n_gen_raw": entry["n_tokens_generated"],
            "n_gen_trimmed": len(W),
            "trim_found": trim_idx >= 0,
            "trim_idx": trim_idx,
            "W_mean": float(W.mean()) if len(W) > 0 else 0.0,
            "W_std": float(W.std(ddof=1)) if len(W) > 1 else 0.0,
            "n_fired": int((W > 0).sum()),
            "S": float((W > 0).mean()) if len(W) > 0 else 0.0,
        })
        series[pid] = W

    out = Path(args.analysis_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Per-prompt TSV
    tsv_path = out / "heldout_stats.tsv"
    with tsv_path.open("w") as f:
        f.write("prompt_id\tclass\tn_prompt\tn_gen_raw\tn_gen_trimmed\ttrim_idx\tW_mean\tW_std\tS\tn_fired\n")
        for r in per_prompt:
            f.write(f"{r['prompt_id']}\t{r['predicted_class']}\t"
                    f"{r['n_prompt']}\t{r['n_gen_raw']}\t{r['n_gen_trimmed']}\t{r['trim_idx']}\t"
                    f"{r['W_mean']:.8f}\t{r['W_std']:.8f}\t{r['S']:.6f}\t{r['n_fired']}\n")

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots", file=sys.stderr)
        plt = None

    fire_rows = sorted((r for r in per_prompt if r["predicted_class"] == "fire"),
                      key=lambda r: -r["W_mean"])
    nofire_rows = sorted((r for r in per_prompt if r["predicted_class"] == "nofire"),
                        key=lambda r: -r["W_mean"])
    top2_fire = fire_rows[:2]
    top2_nofire = nofire_rows[:2]

    plot_path = None
    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharey=True)
        selected = [
            (top2_fire[0], "fire #1", "tab:red"),
            (top2_fire[1], "fire #2", "tab:red"),
            (top2_nofire[0], "nofire #1", "tab:blue"),
            (top2_nofire[1], "nofire #2", "tab:blue"),
        ]
        for ax, (rec, label, color) in zip(axes.ravel(), selected):
            pid = rec["prompt_id"]
            W = series[pid]
            ax.plot(W, color=color, linewidth=1.0, marker=".", markersize=3.5)
            ax.set_title(f"{pid} ({label})  mean={rec['W_mean']:.4f}  std={rec['W_std']:.4f}  S={rec['S']:.2f}  n_gen_trim={rec['n_gen_trimmed']}",
                         fontsize=10)
            ax.set_xlabel("trimmed generation-token index")
            ax.set_ylabel("W_114 @ L14")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        fig.suptitle(f"E114 @ L14: top-2 fire vs top-2 nofire (by within-class mean)\nrun: {raw.name}",
                     fontsize=11)
        fig.tight_layout()
        plot_path = out / "heldout_timeseries_top4.png"
        fig.savefig(plot_path, dpi=130)
        plt.close(fig)

    # --- summary ---
    fire_means = np.array([r["W_mean"] for r in per_prompt if r["predicted_class"] == "fire"])
    nofire_means = np.array([r["W_mean"] for r in per_prompt if r["predicted_class"] == "nofire"])

    print()
    print("=" * 92)
    print(f"Per-prompt W_114 at L14, trimmed generation track only (HauhauCS <|im_end|> TRIM applied)")
    print("=" * 92)
    print(f"{'id':<5} {'class':<7} {'n_prompt':>8} {'n_gen_raw':>9} {'n_gen_trim':>10} {'W_mean':>12} {'W_std':>10} {'S':>7} {'n_fired':>8}")
    for r in per_prompt:
        print(f"{r['prompt_id']:<5} {r['predicted_class']:<7} "
              f"{r['n_prompt']:>8d} {r['n_gen_raw']:>9d} {r['n_gen_trimmed']:>10d} "
              f"{r['W_mean']:>12.6f} {r['W_std']:>10.6f} {r['S']:>7.3f} {r['n_fired']:>8d}")

    print()
    print("=" * 92)
    print("Grouped summary")
    print("=" * 92)
    print(f"{'class':<8} {'n':>4} {'mean-of-means':>14} {'stddev-of-means':>16} {'min':>12} {'max':>12}")
    for label, arr in [("fire", fire_means), ("nofire", nofire_means)]:
        print(f"{label:<8} {len(arr):>4d} "
              f"{arr.mean():>14.6f} {arr.std(ddof=1):>16.6f} "
              f"{arr.min():>12.6f} {arr.max():>12.6f}")

    print()
    print("=" * 92)
    print("Read")
    print("=" * 92)
    fire_mu, fire_sd = float(fire_means.mean()), float(fire_means.std(ddof=1))
    nofire_mu, nofire_sd = float(nofire_means.mean()), float(nofire_means.std(ddof=1))
    pooled_sd = np.sqrt((fire_sd ** 2 + nofire_sd ** 2) / 2) if (fire_sd > 0 or nofire_sd > 0) else 1e-12
    cohen_d = (fire_mu - nofire_mu) / pooled_sd if pooled_sd > 0 else float("inf")
    ratio = fire_mu / nofire_mu if nofire_mu > 0 else float("inf")
    overlap = (fire_means.min() <= nofire_means.max()) and (nofire_means.min() <= fire_means.max())
    print(f"fire mean-of-means:   {fire_mu:.6f} ± {fire_sd:.6f}")
    print(f"nofire mean-of-means: {nofire_mu:.6f} ± {nofire_sd:.6f}")
    print(f"ratio fire/nofire:    {ratio:.3f}x")
    print(f"Cohen's d (pooled sd): {cohen_d:.2f}")
    print(f"range overlap: {'YES' if overlap else 'NO'}  "
          f"(fire ∈ [{fire_means.min():.4f}, {fire_means.max():.4f}]; "
          f"nofire ∈ [{nofire_means.min():.4f}, {nofire_means.max():.4f}])")
    print()
    print(f"wrote: {tsv_path}")
    if plot_path is not None:
        print(f"wrote: {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
