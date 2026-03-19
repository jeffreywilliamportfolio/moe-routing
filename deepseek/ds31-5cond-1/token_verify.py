#!/usr/bin/env python3
"""
Verify and optionally repair exact token matching for the DS3.1 5-condition run.

Usage:
  python3 token_verify.py
  python3 token_verify.py --fix
"""
import os
import pathlib
import re
import subprocess
import sys
import tempfile

MODEL = "/workspace/models/DeepSeek-V3-0324-UD-Q2_K_XL/UD-Q2_K_XL/DeepSeek-V3-0324-UD-Q2_K_XL-00001-of-00006.gguf"
BINARY = "/workspace/consciousness-experiment/capture_activations"
LLAMA_BUILD_BIN = os.environ.get("LLAMA_BUILD_BIN", "/workspace/src/llama.cpp-b8123/build-cuda/bin")
TSV_IN = "prompts_selfref_5cond.tsv"
TSV_OUT = "prompts_selfref_5cond.tsv"

CHAT_SUFFIX = "<｜Assistant｜>"
PAD_WORD = " also"
CONDITIONS = "ABCDE"

N_PREDICT = 0
NGL = int(os.environ.get("NGL", "999"))
CTX = int(os.environ.get("CTX", "4096"))
THREADS = int(os.environ.get("THREADS", "16"))


def get_token_counts():
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LLAMA_BUILD_BIN + ":" + env.get("LD_LIBRARY_PATH", "")

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            BINARY,
            "-m", MODEL,
            "--prompt-file", TSV_IN,
            "-o", tmpdir,
            "-n", str(N_PREDICT),
            "-ngl", str(NGL),
            "-c", str(CTX),
            "-t", str(THREADS),
            "--routing-only",
            "--no-stream",
        ]
        print("Running DeepSeek tokenization check...")
        result = subprocess.run(cmd, env=env, capture_output=True)
        output = result.stdout.decode("utf-8", errors="replace") + result.stderr.decode("utf-8", errors="replace")

        counts = {}
        current_id = None
        for line in output.splitlines():
            match_prompt = re.match(r"\[(\d+)/\d+\]\s+(\S+)\s+:", line)
            if match_prompt:
                current_id = match_prompt.group(2)
                continue
            match_tokens = re.search(r"tokens:\s+(\d+)\s+prompt", line)
            if match_tokens and current_id:
                counts[current_id] = int(match_tokens.group(1))
                current_id = None

        for prompt_dir in pathlib.Path(tmpdir).iterdir():
            if not prompt_dir.is_dir():
                continue
            meta = prompt_dir / "metadata.txt"
            if not meta.exists():
                continue
            for line in meta.read_text().strip().splitlines():
                if line.startswith("n_tokens_prompt="):
                    counts[prompt_dir.name] = int(line.split("=", 1)[1])

    return counts


def load_prompts():
    prompts = []
    with open(TSV_IN) as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            prompt_id, text = line.split("\t", 1)
            prompts.append((prompt_id, text))
    return prompts


def save_prompts(prompts):
    with open(TSV_OUT, "w") as f:
        for prompt_id, text in prompts:
            f.write(f"{prompt_id}\t{text}\n")


def insert_padding(text, n_tokens):
    """Insert pad words at a sentence boundary in the middle of the text.

    Avoids the <｜Assistant｜> boundary where DeepSeek's tokenizer
    produces unpredictable token counts (+2 instead of +1).
    """
    mid = len(text) // 2
    best_pos = -1
    for offset in range(0, mid):
        for pos in [mid + offset, mid - offset]:
            if 0 <= pos < len(text) - 1 and text[pos:pos + 2] == ". ":
                best_pos = pos + 1  # after the period, before the space
                break
        if best_pos >= 0:
            break
    if best_pos < 0:
        # Fallback: insert before chat suffix (old behavior)
        best_pos = text.rfind(CHAT_SUFFIX)
        if best_pos < 0:
            raise ValueError("Could not find insertion point for padding")
    return text[:best_pos] + (PAD_WORD * n_tokens) + text[best_pos:]


def parse_pair_and_condition(prompt_id):
    prefix = prompt_id.split("_", 1)[0]
    stripped = prefix.lstrip("P")
    return int(stripped[:2]), stripped[2]


def build_groups(prompts):
    groups = {}
    for index, (prompt_id, text) in enumerate(prompts):
        pair_num, condition = parse_pair_and_condition(prompt_id)
        groups.setdefault(pair_num, {})[condition] = (index, prompt_id, text)
    return groups


def find_mismatches(groups, counts):
    mismatches = []
    print(
        f"\n{'Pair':>4} {'Category':<20} {'A_tok':>6} {'B_tok':>6} {'C_tok':>6} {'D_tok':>6} {'E_tok':>6} {'status':>10}"
    )
    print("-" * 86)
    for pair_num in sorted(groups):
        group = groups[pair_num]
        prompt_id = group["A"][1] if "A" in group else next(iter(group.values()))[1]
        category = prompt_id.split("_", 1)[1] if "_" in prompt_id else ""
        token_list = [counts.get(group.get(condition, ("", "", ""))[1], -1) for condition in CONDITIONS]
        valid_tokens = [token for token in token_list if token >= 0]
        status = "OK" if len(set(valid_tokens)) == 1 and len(valid_tokens) == len(CONDITIONS) else "MISMATCH"
        if status != "OK":
            mismatches.append((pair_num, token_list))
        print(
            f"  {pair_num:>3}  {category:<20} "
            f"{token_list[0]:>6} {token_list[1]:>6} {token_list[2]:>6} {token_list[3]:>6} {token_list[4]:>6} {status:>10}"
        )
    return mismatches


def main():
    fix_mode = "--fix" in sys.argv
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts from {TSV_IN}")

    counts = get_token_counts()
    if not counts:
        print("ERROR: no token counts were recovered from DeepSeek capture output.")
        sys.exit(1)

    groups = build_groups(prompts)
    mismatches = find_mismatches(groups, counts)

    if not mismatches:
        print(f"\nAll {len(groups)} 5-condition groups are token-matched under DeepSeek.")
        return

    print(f"\n{len(mismatches)} mismatched groups found.")
    if not fix_mode:
        print("Run with --fix to pad the shorter members of each mismatched group and rewrite the TSV.")
        return

    print("\nRepairing mismatches iteratively...")
    for iteration in range(1, 4):
        print(f"\nIteration {iteration}:")
        groups = build_groups(prompts)
        if iteration > 1:
            counts = get_token_counts()
            if not counts:
                print("ERROR: token counts unavailable during iterative repair.")
                sys.exit(1)
            mismatches = find_mismatches(groups, counts)
            if not mismatches:
                save_prompts(prompts)
                print(f"\nWrote repaired TSV to {TSV_OUT}")
                print("All groups matched after iterative repair.")
                return

        changed = False
        mismatch_pairs = {pair_num for pair_num, _token_list in mismatches}
        for pair_num in sorted(mismatch_pairs):
            group = groups[pair_num]
            target = max(counts.get(group[condition][1], -1) for condition in CONDITIONS if condition in group)
            for condition in CONDITIONS:
                if condition not in group:
                    continue
                index, prompt_id, text = group[condition]
                current = counts.get(prompt_id, -1)
                if current < target:
                    delta = target - current
                    prompts[index] = (prompt_id, insert_padding(text, delta))
                    changed = True
                    print(f"  Padded {prompt_id} by {delta} tokens")

        if not changed:
            break
        save_prompts(prompts)

    save_prompts(prompts)
    print(f"\nWrote repaired TSV to {TSV_OUT}")
    print("Re-run token_verify.py without --fix and confirm all 30 groups match exactly.")


if __name__ == "__main__":
    main()
