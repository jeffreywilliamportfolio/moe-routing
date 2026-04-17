# MoE Routing Experiments — Qwen3.5-35B-A3B / HauhauCS

Mixture-of-Experts routing experiments on Qwen3.5-35B-A3B and HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive (Q8_0). All experiments measure Expert 114 selection rate (S), conditional weight (Q), and unconditional contribution (W = S × Q) during prefill and generation, with varying interventions.

## Standard Folder Structure

Most single-experiment folders follow this layout:

```
experiment-folder/
├── README.md          — researcher entrypoint: scope, headline result, reading order
├── MANIFEST.md        — file inventory: what's included, what's remote/excluded
├── DOCS/
│   ├── PLAN.md        — run record: model config, prompt design, capture assumptions
│   ├── RESULTS.md     — findings, tables, and interpretation
│   └── REPRODUCE.md   — (optional) reproducibility caveats
├── METHOD/            — Python analysis scripts and C++ capture binary source
├── PROMPTS/           — prompt TSVs (think/no-think), JSON suites, and metadata
└── results/           — JSON outputs, markdown summaries, generated text, logs
    captures/          — timestamped raw capture directories (.npy excluded from git)
```

`.npy` files are excluded from git (too large). The `Documentation/` folder is a local reference only and is gitignored.

Exceptions:

- `qwen3.5-35b-a3b-huahua-agressive-experts/` keeps reviewer-facing raw paths under its original LFS-backed layout instead of a top-level `results/` directory.
- `qwen3.5-35b-a3b-huahua-philosophy-experts-bias/` splits outputs across capture and per-run-family result directories because of the larger multi-round pull workflow.
- `qwen3.5-35b-a3b-huahua-humor-test/` has been normalized to the standard layout, but preserves original artifact filenames where needed for provenance.

Per-token breakdowns are currently included for:

- `qwen3.5-35b-a3b-huahua-single-prompt-processing-hum`
- `qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk`
- `qwen3.5-35b-a3b-huahua-humor-test`

## Experiments

| Folder | Description |
|---|---|
| [`qwen3.5-35b-a3b-huahua-agressive-experts/`](qwen3.5-35b-a3b-huahua-agressive-experts/) | **Template / reference experiment.** Expert 114 causal basin-steering study; 5-condition routing map + sham controls. HauhauCS Q8. |
| [`qwen3.5-35b-a3b-huahua-vs-base-run1/`](qwen3.5-35b-a3b-huahua-vs-base-run1/) | **Base vs. HauhauCS comparison.** 150-prompt Cal–Manip–Cal prefill-only comparison between vanilla Qwen3.5-35B-A3B and HauhauCS. Exact duplicate reproduction confirmed. |
| [`qwen3.5-35b-a3b-huahua-6cond-moe-manips/`](qwen3.5-35b-a3b-huahua-6cond-moe-manips/) | **First 6-condition MoE manipulation survey.** 180 prompts (L1/L2/L3 × 6 deictics), ML/computation content. E114 generation L3/L1 ratio 3.23×; rank-1 at layer 14. HauhauCS Q8. |
| [`qwen3.5-35b-a3b-huahua-6cond-hvac/`](qwen3.5-35b-a3b-huahua-6cond-hvac/) | **Off-topic domain control.** Same 6-condition structure with HVAC/water-treatment content (not ML). E114 L3/L1 ratio strengthened to 4.62×; rank-1 lock across all 60 L3 cells. Rebuts ML-topic specialist hypothesis. |
| [`qwen3.5-35b-a3b-huahua-expert-identification/`](qwen3.5-35b-a3b-huahua-expert-identification/) | **Domain specialist routing survey.** 60 prompts × 20 domains × 3/domain; maps all 256 experts. E114 wins philosophy in generation; Expert 224 dominates prefill across 18/20 domains. HauhauCS Q8. |
| [`qwen3.5-35b-a3b-huahua-philosophy-experts-bias/`](qwen3.5-35b-a3b-huahua-philosophy-experts-bias/) | **Philosophy cluster suppression.** Suppresses E114+E87+E170+E68 during 60-domain specialist prompts. Multiple bias levels (m8-p5, p8-only). 5,400+ files including 62 granular text-pull subdirs. |
| [`qwen3.5-35b-a3b-huahua-114-pm/`](qwen3.5-35b-a3b-huahua-114-pm/) | **Expert 114 single-prompt bias sweep.** Two self-report prompts (emergent intelligence + experience probe), bias −8→+5, think and no-think. All suppression levels drive E114 to exact zero; +5.0 produces incoherent output. |
| [`qwen3.5-35b-a3b-huahua-humor-test/`](qwen3.5-35b-a3b-huahua-humor-test/) | **Single-joke deictic canary.** One joke prompt across five deictic framings. Prefill entropy stays nearly flat; the main visible effect is output-length instability rather than a clear routing split. |
| [`qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/`](qwen3.5-35b-a3b-huahua-single-prompt-processing-hum/) | **Processing Hum per-token probe.** Single no-think prompt (1024 tokens). E114 peaks cluster on phenomenological language (continuity, stillness, being, ground). Deep-still-water segment (tokens 210–392) materially stronger than whole-output mean at layers 14 and 26. |
| [`qwen3.5-35b-a3b-huahua-five-cond-experience-probe/`](qwen3.5-35b-a3b-huahua-five-cond-experience-probe/) | **5-condition experience probe.** 15 prompts (3 pairs × 5 deictics), full router capture. E114 is top manipulation expert on all 15 prompts. KL-manip Wilcoxon p=6.3e-05. |
| [`qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/`](qwen3.5-35b-a3b-huahua-domain-expert-probe-3chunk/) | **Domain expert probe 3-chunk.** 60 domain questions collapsed into 3 token-balanced long prompts (446 tokens each). Generation expert set diverges from prefill (Jaccard as low as 0.0). E114 rises from rank 89 → rank 7 in generation for chunk C. |
| [`qwen3.5-35b-a3b-huahua-strangeloop/`](qwen3.5-35b-a3b-huahua-strangeloop/) | **Strangeloop paired control.** 30 A/B Gödel/Escher/Quine/bootstrap/tangled-hierarchy pairs, prefill-only. All-token RE is weak, but last-token RE and KL-manip show a reliable definiteness effect. |
| [`qwen3.5-35b-a3b-huahua-114-selfref-heldout/`](qwen3.5-35b-a3b-huahua-114-selfref-heldout/) | **E114 self-reference heldout, matched-token control.** 20 prompts (10 fire + 10 nofire) sharing the same anchor tokens in self-referential vs. external contexts. Trimmed-generation W₁₁₄ at L14 separates **21.7×** (fire 0.0675 vs nofire 0.0031), Cohen's d **2.94**, no range overlap. Lexical hypothesis ruled out. N10 outlier refines the label toward phenomenological register in generated output. HauhauCS Q8. |

## License

All code and documentation in this repository are released under the MIT License — see [LICENSE](LICENSE).

## Model

- Base: Qwen3.5-35B-A3B (vanilla)
- Fine-tuned: HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive
- Quantization: Q8_0 for all experiments
- Hardware: 2× RTX 5090 (Vast.ai)
- Runtime: llama.cpp with custom MoE capture binary
