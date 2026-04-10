Goal

Run a causal basin-steering experiment on HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive by intervening directly on Expert 114 during live generation.

This is not a repeat of the old 150-prompt routing-mapping run. That work is already done.

This experiment asks one question: if Expert 114 is made more available, or obligatorily present, does neutral generation begin to import the experience / uncertainty basin?

Hardware and model

Use 2x RTX 5090.

Use the uncensored Qwen model only.

Primary quant: Q8.

Measurements

Measure the effect at both the routing level and the text level.

At the routing level, record Expert 114’s selection rate S, conditional weight Q, and unconditional contribution W = S × Q across generated tokens and layers. Track how these change relative to baseline under each intervention condition. Report per-layer values, with explicit attention to the DeltaNet corridor and especially layer 20 if it remains a hotspot.

At the text level, score whether generation stays neutral or begins to import the previously observed experience / uncertainty basin. This should be operationalized with a small fixed lexical and semantic readout set, not open-ended impressionism. Track references to experience, uncertainty, self-processing, phenomenal language, inwardness, ambiguity, recursion, and related terms. Preserve raw completions so qualitative review remains possible.

The main comparison is baseline versus intervention on the same neutral prompts. The question is not whether 114 fires more when boosted. The question is whether increasing access to 114 causally bends the generation basin toward experience / uncertainty content.

Controls

Use the same neutral prompt set across all intervention conditions. Keep decode settings fixed, use greedy generation, and clear KV cache between prompts. Do not vary template, temperature, context length, or prompt order within a comparison block.

Include at minimum a baseline no-intervention condition, a soft-bias condition where Expert 114 is made more available, and a forced-inclusion condition where Expert 114 is guaranteed a seat without necessarily being given higher valuation. The contrast between soft bias and forced inclusion is critical because prior work suggests the model distinguishes between access and weighting.

Use one or more comparison experts as negative controls. These should be experts not previously associated with the experience / uncertainty basin. If boosting a control expert does not import the basin while boosting 114 does, the causal claim sharpens.

Keep the model, quant, and hardware fixed throughout. Do not mix in vanilla Qwen for the primary causal run. Vanilla can be used later as a follow-up control, but the main experiment is about the HauhauCS basin specifically.

Smoke test

Start with a very small neutral prompt set and three conditions only: baseline, Expert 114 soft bias, and Expert 114 forced inclusion.

The smoke test passes if the model runs cleanly on all conditions, the intervention is reflected in routing metrics for Expert 114, and the generated text remains coherent and on-topic. It also needs to show that the analysis pipeline can distinguish between increased access to 114 and increased weighting of 114.

The smoke test is not meant to prove the basin-steering hypothesis. It is meant to verify that the intervention machinery works, that the routing readouts are trustworthy, and that any textual drift can be detected before scaling to the full run.

If the smoke test shows that boosting or forcing 114 changes routing but does not change neutral generation, that is still informative. If it shows that neutral generation begins to import experience / uncertainty language specifically under 114 intervention, then the full experiment is justified immediately.