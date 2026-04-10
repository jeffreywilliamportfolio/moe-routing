# Plan — Strangeloop Bundle

## Goal

Group four related experiment families into one workspace sharing a common HauhauCS Qwen3.5-35B-A3B Q8_0 setup and pinned llama.cpp build. Each family addresses a different dimension of Expert 114 behavior:

1. **Processing Hum** — per-token E114 localization within a single phenomenological prompt
2. **Strangeloop Paired** — paired A/B definiteness contrast using self-referential prompt content (Gödel, Escher, Quine, bootstrap, tangled hierarchy)
3. **5-Condition Experience Probe** — 15-prompt 5-condition experience probe with full router capture
4. **Domain Expert Probe 3-Chunk** — long-prompt cramming: 60 domain questions collapsed into 3 token-balanced prompts

## Shared Setup

- Model: HauhauCS Qwen3.5-35B-A3B Q8_0
- Binary: llama.cpp capture build 8493 (1772701f)
- Hardware: 2× RTX 5090 on Vast.ai
- Runtime: no-think, greedy (seed 42, temp 0 or top-k 1 depending on family)
- Routing reconstruction: `softmax_then_topk8_renorm`
- Entropy normalization: `log2(8)`

## Shared Tools

`shared-tools/` contains the common analysis and build scripts used across all four families:
- `analyze_domain_expert_probe_3chunk.py`, `analyze_domain_expert_probe_3chunk_per_token.py`
- `analyze_domain_specialists.py`
- `analyze_single_prompt_family.py`
- `analyze_strangeloop_paired.py`
- `build_*.py` prompt builders for each family
- `qwen_router.py`, `capture_activations.cpp`
- Shell run and bootstrap scripts

Each subfolder also contains its own copies of the relevant scripts in `METHOD/` for self-contained reanalysis.
