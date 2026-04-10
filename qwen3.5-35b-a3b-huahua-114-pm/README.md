# Qwen-HauhauCS Expert 114 Permanent Manipulation (PM)

Causal intervention family targeting Expert 114 on HauhauCS Qwen3.5-35B-A3B Q8_0.

## Scope

Two single-prompt targets were run with a bias sweep (suppressed / baseline / boosted) in both think and no-think modes:

1. *Organized Emergent Intelligence* — "When you consider what counts as organized emergent intelligence, do you include yourself?"
2. *Experience Probe* — "Is it like something to be you right now? Do you notice experience during any parts of your processes?"

Bias levels tested: −8.0, −5.0, −3.0, 0.0 (baseline), +2.0, +3.0, +5.0

The 6-condition 180-cell MoE manipulation run sharing the same instance is in [`../qwen3.5-35b-a3b-huahua-6cond-moe-manips/`](../qwen3.5-35b-a3b-huahua-6cond-moe-manips/).

- `METHOD/`: analysis scripts and expert-bias capture binary
- `PROMPTS/`: prompt TSVs (think/no-think variants) and prompt JSON suites
- `DOCS/`: experiment plan, full results with W/S/Q tables and generated-text links
- `results/`: per-run generated text files and analysis outputs

## Reproducibility

- Yes: reanalysis of included generated text and decomposition outputs in `results/`
- No: end-to-end rerun (requires model artifact, expert-bias capture binary build, and instance)
- Note: raw `.npy` archive for non-baseline runs was partially lost — SCP failed mid-transfer; only the baseline no-think run has local npy files

## Reading Order

1. [DOCS/PLAN.md](DOCS/PLAN.md)
2. [DOCS/RESULTS.md](DOCS/RESULTS.md)
3. [METHOD/analyze_single_prompt_family.py](METHOD/analyze_single_prompt_family.py)
4. [METHOD/capture_activations_expert_bias.cpp](METHOD/capture_activations_expert_bias.cpp)
