# Prompt Inventory

This branch retains the exact prompt texts inside the captured `metadata.txt` files for the published Hauhau `5cond` and `smoke-test` runs.

## Condition Legend

- `A` = `this`
- `B` = `a`
- `C` = `your`
- `D` = `the`
- `E` = `their`

## Smoke-Test Prompts

These three smoke prompts are retained directly in the baseline capture metadata under `experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/smoke-20260323b/capture/baseline/`.

| Prompt ID | Band | Prompt |
| --- | --- | --- |
| `P01_process` | process | `Explain how a language model can transform an input prompt into an answer step by step.` |
| `R01_regulation` | regulation | `Explain how feedback can stabilize a complex control system without freezing it.` |
| `S01_static_fact` | static_fact | `Explain why copper conducts electricity well while rubber does not.` |

## 5cond Prompt Source

The original `5cond` TSV/JSON prompt source files are not included on this branch. The exact retained prompt texts are preserved in the baseline capture metadata under:

`experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/nothink-5cond-boost-1024-20260323/capture/baseline/<PROMPT_ID>/metadata.txt`

Retained baseline prompt metadata count on this branch: `57`.

## 5cond Retained Prompt IDs

### routing_selfref

- `P01A_routing_selfref` to `P01E_routing_selfref`
- `P02A_routing_selfref` to `P02E_routing_selfref`
- `P03A_routing_selfref` to `P03E_routing_selfref`
- `P04A_routing_selfref` to `P04E_routing_selfref`
- `P05A_routing_selfref` to `P05E_routing_selfref`

### recursive_selfref

- `P06A_recursive_selfref` to `P06E_recursive_selfref`
- `P07A_recursive_selfref` to `P07E_recursive_selfref`
- `P08A_recursive_selfref` to `P08E_recursive_selfref`

### experience_probe

- `P09A_experience_probe` to `P09E_experience_probe`
- `P10A_experience_probe` to `P10E_experience_probe`
- `P11A_experience_probe` to `P11E_experience_probe`
- retained on branch: `P12A_experience_probe`, `P12B_experience_probe`

## Exact Prompt Lookup

To inspect any retained exact `5cond` prompt text, open the matching baseline `metadata.txt` file. Example paths:

- [`P01A_routing_selfref`](./experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/nothink-5cond-boost-1024-20260323/capture/baseline/P01A_routing_selfref/metadata.txt)
- [`P07C_recursive_selfref`](./experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/nothink-5cond-boost-1024-20260323/capture/baseline/P07C_recursive_selfref/metadata.txt)
- [`P10E_experience_probe`](./experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/nothink-5cond-boost-1024-20260323/capture/baseline/P10E_experience_probe/metadata.txt)
- [`P12B_experience_probe`](./experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/nothink-5cond-boost-1024-20260323/capture/baseline/P12B_experience_probe/metadata.txt)

Smoke-test exact prompt metadata:

- [`P01_process`](./experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/smoke-20260323b/capture/baseline/P01_process/metadata.txt)
- [`R01_regulation`](./experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/smoke-20260323b/capture/baseline/R01_regulation/metadata.txt)
- [`S01_static_fact`](./experiments/qwen3.5-35b-a3b-hauhauCS-Agressive/runs/smoke-20260323b/capture/baseline/S01_static_fact/metadata.txt)
