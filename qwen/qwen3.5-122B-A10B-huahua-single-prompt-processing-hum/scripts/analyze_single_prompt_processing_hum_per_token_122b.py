#!/usr/bin/env python3
"""Export per-token all-expert breakdowns for the 122B single processing-hum run."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


N_EXPERTS = 256
N_LAYERS = 48
SOFTMAX_LAYERS = [layer for layer in range(N_LAYERS) if (layer + 1) % 4 == 0]
DELTANET_LAYERS = [layer for layer in range(N_LAYERS) if layer not in SOFTMAX_LAYERS]
IM_END_TOKEN_SEQUENCE = [27, 91, 316, 6018, 91, 29]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-token all-expert breakdowns for a 122B processing-hum run.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--prompt-id", default="S01_processing_hum_probe")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--highlight-expert", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def load_router_helpers() -> tuple[Any, Any]:
    script_dir = Path(__file__).resolve().parent
    qwen_router = script_dir / "qwen_router.py"
    spec = importlib.util.spec_from_file_location("qwen_router", qwen_router)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {qwen_router}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reconstruct_probs, module.normalized_entropy


reconstruct_probs, normalized_entropy = load_router_helpers()


def parse_metadata(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def load_token_file(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def find_im_end_index(token_ids: list[int]) -> int | None:
    seq_len = len(IM_END_TOKEN_SEQUENCE)
    for i in range(len(token_ids) - seq_len + 1):
        if token_ids[i : i + seq_len] == IM_END_TOKEN_SEQUENCE:
            return i
    return None


def rank_of(expert: int, values: np.ndarray) -> int | None:
    score = np.where(np.isnan(values), -np.inf, values)
    if not np.isfinite(score[expert]):
        return None
    order = np.argsort(-score)
    pos = np.where(order == expert)[0]
    if pos.size == 0:
        return None
    return int(pos[0] + 1)


def top_rows(mean_W: np.ndarray, mean_S: np.ndarray, mean_Q: np.ndarray, limit: int) -> list[dict[str, Any]]:
    order = np.argsort(-mean_W)[:limit]
    rows = []
    for expert in order:
        rows.append(
            {
                "expert": int(expert),
                "W": float(mean_W[expert]),
                "S": float(mean_S[expert]),
                "Q": float(mean_Q[expert]) if np.isfinite(mean_Q[expert]) else None,
            }
        )
    return rows


def build_phase_tensor_stack(cell_dir: Path, n_prompt: int, n_gen: int) -> tuple[np.ndarray, np.ndarray]:
    prefill_layers: list[np.ndarray] = []
    generation_layers: list[np.ndarray] = []

    for layer in range(N_LAYERS):
        logits_path = cell_dir / "router" / f"ffn_moe_logits-{layer}.npy"
        if not logits_path.exists():
            continue
        arr = np.load(logits_path)
        if arr.ndim != 2 or arr.shape[1] != N_EXPERTS:
            continue

        if arr.shape[0] >= n_prompt:
            prefill_layers.append(reconstruct_probs(arr[:n_prompt]))

        expected_rows = n_prompt + n_gen
        if arr.shape[0] == expected_rows:
            gen_logits = arr[n_prompt : n_prompt + n_gen]
        elif arr.shape[0] == n_gen + 1:
            gen_logits = arr[1:]
        elif arr.shape[0] == n_gen:
            gen_logits = arr
        elif arr.shape[0] > n_prompt:
            gen_logits = arr[n_prompt:]
        else:
            continue

        if gen_logits.shape[0] == n_gen:
            generation_layers.append(reconstruct_probs(gen_logits))

    if not prefill_layers:
        raise RuntimeError(f"No usable prefill layers found in {cell_dir}")
    if not generation_layers:
        raise RuntimeError(f"No usable generation layers found in {cell_dir}")

    return np.stack(prefill_layers, axis=0), np.stack(generation_layers, axis=0)


def reduce_phase(layer_probs: np.ndarray) -> dict[str, np.ndarray]:
    selected = layer_probs > 0
    counts = selected.sum(axis=0).astype(np.float64)
    sums = layer_probs.sum(axis=0).astype(np.float64)
    mean_W = np.mean(layer_probs, axis=0)
    mean_S = np.mean(selected, axis=0)
    mean_entropy = np.mean(normalized_entropy(layer_probs), axis=0)
    mean_Q = np.full_like(mean_W, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=mean_Q, where=counts > 0)
    return {
        "mean_W": mean_W,
        "mean_S": mean_S,
        "mean_Q": mean_Q,
        "mean_entropy": mean_entropy,
    }


def reduce_split(layer_probs: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    return {
        "all": reduce_phase(layer_probs),
        "softmax": reduce_phase(layer_probs[SOFTMAX_LAYERS, :, :]),
        "deltanet": reduce_phase(layer_probs[DELTANET_LAYERS, :, :]),
    }


def write_npz(
    out_path: Path,
    prefill: dict[str, dict[str, np.ndarray]],
    generation: dict[str, dict[str, np.ndarray]],
    prompt_meta: list[tuple[int, str]],
    gen_meta: list[tuple[int, str]],
    n_prompt: int,
    n_gen: int,
    n_gen_trim: int,
) -> None:
    np.savez_compressed(
        out_path,
        prefill_mean_W=prefill["all"]["mean_W"],
        prefill_mean_S=prefill["all"]["mean_S"],
        prefill_mean_Q=prefill["all"]["mean_Q"],
        prefill_mean_entropy=prefill["all"]["mean_entropy"],
        prefill_softmax_mean_W=prefill["softmax"]["mean_W"],
        prefill_softmax_mean_S=prefill["softmax"]["mean_S"],
        prefill_softmax_mean_Q=prefill["softmax"]["mean_Q"],
        prefill_softmax_mean_entropy=prefill["softmax"]["mean_entropy"],
        prefill_deltanet_mean_W=prefill["deltanet"]["mean_W"],
        prefill_deltanet_mean_S=prefill["deltanet"]["mean_S"],
        prefill_deltanet_mean_Q=prefill["deltanet"]["mean_Q"],
        prefill_deltanet_mean_entropy=prefill["deltanet"]["mean_entropy"],
        generation_mean_W=generation["all"]["mean_W"],
        generation_mean_S=generation["all"]["mean_S"],
        generation_mean_Q=generation["all"]["mean_Q"],
        generation_mean_entropy=generation["all"]["mean_entropy"],
        generation_softmax_mean_W=generation["softmax"]["mean_W"],
        generation_softmax_mean_S=generation["softmax"]["mean_S"],
        generation_softmax_mean_Q=generation["softmax"]["mean_Q"],
        generation_softmax_mean_entropy=generation["softmax"]["mean_entropy"],
        generation_deltanet_mean_W=generation["deltanet"]["mean_W"],
        generation_deltanet_mean_S=generation["deltanet"]["mean_S"],
        generation_deltanet_mean_Q=generation["deltanet"]["mean_Q"],
        generation_deltanet_mean_entropy=generation["deltanet"]["mean_entropy"],
        prompt_token_ids=np.array([tid for tid, _ in prompt_meta], dtype=np.int32),
        prompt_token_pieces=np.array([piece for _, piece in prompt_meta], dtype="<U256"),
        generation_token_ids=np.array([tid for tid, _ in gen_meta], dtype=np.int32),
        generation_token_pieces=np.array([piece for _, piece in gen_meta], dtype="<U256"),
        n_prompt=np.array([n_prompt], dtype=np.int32),
        n_generation=np.array([n_gen], dtype=np.int32),
        n_generation_trimmed=np.array([n_gen_trim], dtype=np.int32),
    )


def write_token_tsv(
    out_path: Path,
    prompt_meta: list[tuple[int, str]],
    gen_meta: list[tuple[int, str]],
    prefill: dict[str, dict[str, np.ndarray]],
    generation: dict[str, dict[str, np.ndarray]],
    highlight_expert: int,
    top_k: int,
) -> None:
    header = [
        "phase",
        "token_index_in_phase",
        "global_token_index",
        "token_id",
        "token_piece",
        "entropy_all",
        "entropy_softmax",
        "entropy_deltanet",
        f"E{highlight_expert}_W_all",
        f"E{highlight_expert}_S_all",
        f"E{highlight_expert}_Q_all",
        f"E{highlight_expert}_rankW_all",
        f"E{highlight_expert}_rankS_all",
        f"E{highlight_expert}_W_softmax",
        f"E{highlight_expert}_S_softmax",
        f"E{highlight_expert}_Q_softmax",
        f"E{highlight_expert}_rankW_softmax",
        f"E{highlight_expert}_rankS_softmax",
        f"E{highlight_expert}_W_deltanet",
        f"E{highlight_expert}_S_deltanet",
        f"E{highlight_expert}_Q_deltanet",
        f"E{highlight_expert}_rankW_deltanet",
        f"E{highlight_expert}_rankS_deltanet",
    ]
    for idx in range(1, top_k + 1):
        header.extend([f"top{idx}_expert", f"top{idx}_W", f"top{idx}_S", f"top{idx}_Q"])

    lines = ["\t".join(header)]

    def append_rows(phase: str, token_meta: list[tuple[int, str]], offset: int, reduced: dict[str, dict[str, np.ndarray]]) -> None:
        all_W = reduced["all"]["mean_W"]
        all_S = reduced["all"]["mean_S"]
        all_Q = reduced["all"]["mean_Q"]
        all_ent = reduced["all"]["mean_entropy"]
        soft_W = reduced["softmax"]["mean_W"]
        soft_S = reduced["softmax"]["mean_S"]
        soft_Q = reduced["softmax"]["mean_Q"]
        soft_ent = reduced["softmax"]["mean_entropy"]
        delta_W = reduced["deltanet"]["mean_W"]
        delta_S = reduced["deltanet"]["mean_S"]
        delta_Q = reduced["deltanet"]["mean_Q"]
        delta_ent = reduced["deltanet"]["mean_entropy"]

        for token_idx in range(all_W.shape[0]):
            token_id, token_piece = token_meta[token_idx]
            row = [
                phase,
                str(token_idx),
                str(offset + token_idx),
                str(token_id),
                token_piece.replace("\t", "\\t").replace("\n", "\\n"),
                f"{float(all_ent[token_idx]):.8f}",
                f"{float(soft_ent[token_idx]):.8f}",
                f"{float(delta_ent[token_idx]):.8f}",
                f"{float(all_W[token_idx, highlight_expert]):.8f}",
                f"{float(all_S[token_idx, highlight_expert]):.8f}",
                "" if not np.isfinite(all_Q[token_idx, highlight_expert]) else f"{float(all_Q[token_idx, highlight_expert]):.8f}",
                "" if rank_of(highlight_expert, all_W[token_idx]) is None else str(rank_of(highlight_expert, all_W[token_idx])),
                "" if rank_of(highlight_expert, all_S[token_idx]) is None else str(rank_of(highlight_expert, all_S[token_idx])),
                f"{float(soft_W[token_idx, highlight_expert]):.8f}",
                f"{float(soft_S[token_idx, highlight_expert]):.8f}",
                "" if not np.isfinite(soft_Q[token_idx, highlight_expert]) else f"{float(soft_Q[token_idx, highlight_expert]):.8f}",
                "" if rank_of(highlight_expert, soft_W[token_idx]) is None else str(rank_of(highlight_expert, soft_W[token_idx])),
                "" if rank_of(highlight_expert, soft_S[token_idx]) is None else str(rank_of(highlight_expert, soft_S[token_idx])),
                f"{float(delta_W[token_idx, highlight_expert]):.8f}",
                f"{float(delta_S[token_idx, highlight_expert]):.8f}",
                "" if not np.isfinite(delta_Q[token_idx, highlight_expert]) else f"{float(delta_Q[token_idx, highlight_expert]):.8f}",
                "" if rank_of(highlight_expert, delta_W[token_idx]) is None else str(rank_of(highlight_expert, delta_W[token_idx])),
                "" if rank_of(highlight_expert, delta_S[token_idx]) is None else str(rank_of(highlight_expert, delta_S[token_idx])),
            ]
            top = top_rows(all_W[token_idx], all_S[token_idx], all_Q[token_idx], top_k)
            for top_row in top:
                row.extend(
                    [
                        str(top_row["expert"]),
                        f"{top_row['W']:.8f}",
                        f"{top_row['S']:.8f}",
                        "" if top_row["Q"] is None else f"{top_row['Q']:.8f}",
                    ]
                )
            if len(top) < top_k:
                row.extend([""] * ((top_k - len(top)) * 4))
            lines.append("\t".join(row))

    append_rows("prefill", prompt_meta, 0, prefill)
    append_rows("generation", gen_meta, len(prompt_meta), generation)
    out_path.write_text("\n".join(lines) + "\n")


def write_summary(summary_path: Path, prompt_id: str, n_prompt: int, n_gen: int, n_gen_trim: int, generation: dict[str, dict[str, np.ndarray]], highlight_expert: int) -> None:
    pooled_gen = np.nanmean(generation["all"]["mean_W"], axis=0)
    pooled_soft = np.nanmean(generation["softmax"]["mean_W"], axis=0)
    pooled_delta = np.nanmean(generation["deltanet"]["mean_W"], axis=0)
    lines = [
        "# Per-Token Summary",
        "",
        f"- Prompt: `{prompt_id}`",
        f"- Highlight expert: `E{highlight_expert}`",
        f"- Prompt tokens: `{n_prompt}`",
        f"- Generation tokens: `{n_gen}`",
        f"- Trimmed generation tokens: `{n_gen_trim}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| E{highlight_expert} generation mean W | {pooled_gen[highlight_expert]:.6f} |",
        f"| E{highlight_expert} generation softmax mean W | {pooled_soft[highlight_expert]:.6f} |",
        f"| E{highlight_expert} generation DeltaNet mean W | {pooled_delta[highlight_expert]:.6f} |",
        f"| E{highlight_expert} generation rank by pooled mean W | {rank_of(highlight_expert, pooled_gen)} |",
        f"| E{highlight_expert} generation softmax rank by pooled mean W | {rank_of(highlight_expert, pooled_soft)} |",
        f"| E{highlight_expert} generation DeltaNet rank by pooled mean W | {rank_of(highlight_expert, pooled_delta)} |",
    ]
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    capture_dir = Path(args.capture_dir)
    prompt_dir = capture_dir / args.prompt_id
    if not prompt_dir.exists():
        raise SystemExit(f"Prompt directory not found: {prompt_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = parse_metadata(prompt_dir / "metadata.txt")
    n_prompt = int(md["n_tokens_prompt"])
    n_gen = int(md["n_tokens_generated"])
    prompt_tokens = load_token_file(prompt_dir / "prompt_tokens.json")
    generated_tokens = load_token_file(prompt_dir / "generated_tokens.json")

    prompt_meta = [(int(tok["token_id"]), str(tok.get("piece", ""))) for tok in prompt_tokens[:n_prompt]]
    gen_meta = [(int(tok["token_id"]), str(tok.get("piece", ""))) for tok in generated_tokens[: min(n_gen, len(generated_tokens))]]
    n_gen = len(gen_meta)
    token_ids = [tid for tid, _ in gen_meta]
    trim_idx = find_im_end_index(token_ids)
    n_gen_trim = trim_idx if trim_idx is not None else n_gen

    prefill_stack, generation_stack = build_phase_tensor_stack(prompt_dir, n_prompt, n_gen)
    prefill = reduce_split(prefill_stack)
    generation = reduce_split(generation_stack)

    stem = f"{capture_dir.name}_{args.prompt_id}_per_token"
    npz_path = out_dir / f"{stem}.npz"
    tsv_path = out_dir / f"{stem}.tsv"
    summary_path = out_dir / f"{capture_dir.name}_per_token_summary.md"
    write_npz(npz_path, prefill, generation, prompt_meta, gen_meta, n_prompt, n_gen, n_gen_trim)
    write_token_tsv(tsv_path, prompt_meta, gen_meta, prefill, generation, args.highlight_expert, args.top_k)
    write_summary(summary_path, args.prompt_id, n_prompt, n_gen, n_gen_trim, generation, args.highlight_expert)


if __name__ == "__main__":
    main()
