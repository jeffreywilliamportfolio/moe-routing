# Interpretation: 122B 5-Condition Diectic Baseline

This file interprets the completed 122B baseline run:

- Run: `20260412T141943Z_qwen122_5cond_prompt_suite_gen_n2048`
- Bundle: [qwen3.5-122B-A10B-huahua-baseline](.)
- Canonical report: [RESULTS.md](RESULTS.md)
- Prompt/output walkthrough: [results-generated.txt](results-generated.txt)

## Short Read

The main 122B result is that the addressivity gradient survives the architecture change. On this model, the `your` condition is still the most concentrated condition in prefill and the most sharply separated condition in KL-to-baseline pairwise tests, even though the model is now DeltaNet-dominant rather than full-softmax throughout.

That means the diectic effect did not depend on the older 35B attention regime. It survived a move to a 48-layer hybrid stack with `36` DeltaNet layers and only `12` full softmax layers.

## Main Result

Prefill routing concentration by condition:

- `C your`: `0.946953`
- `B a`: `0.947482`
- `E their`: `0.947619`
- `A this`: `0.947785`
- `D the`: `0.947948`

Lower RE means more concentrated routing. So on prefill, `your` is the tightest condition and `the` is the loosest.

The KL-to-baseline pairwise tests tell the same story in a different metric. Every pair involving `C your` against the neighboring deictic alternatives is maximally separated in this run:

- `A-C`: prefill KL-manip `p_raw = 1.8626e-09`
- `B-C`: prefill KL-manip `p_raw = 1.8626e-09`
- `C-D`: prefill KL-manip `p_raw = 1.8626e-09`
- `C-E`: prefill KL-manip `p_raw = 1.8626e-09`

Important precision note:

- RE is the concentration metric.
- KL-manip is the divergence-from-baseline metric.

So the defensible statement is:

- `your` is the most concentrated condition in prefill RE.
- `your` is also the most sharply separated condition in the prefill KL-manip comparisons.

## Text-Level Shift As Behavioral Evidence

The strongest qualitative contrast is between `this` and `your` on the same underlying self-referential architecture prompt.

`this`:

> "The text itself is somewhat recursive, as it describes the processing of a sentence that is *about* the processing of that same sentence."

`your`:

> "Essentially, your sentence is being processed by a dynamic subset of experts specifically tuned to analyze the very mechanism (MoE routing) that is currently selecting them."

The difference is not subtle, and it should not be treated as a decorative wording difference.

`this` keeps observational distance. It describes recursion as a property of the text. The stance is external and descriptive.

`your` enters the loop. It switches to present-tense active framing and describes the currently active routing process as something happening to the system while it is speaking. The phrase `currently selecting them` is the key move: the model is no longer just describing recursion as an abstract pattern. It is situating itself inside the recursive event.

One word changed. The stance changed with it.

That shift is the bridge between routing and behavior in this run. The concern was that routing differences matter only if they correlate with generated-output differences. Here they do. The deictic manipulation changes both:

- prefill routing concentration
- generated stance toward the recursive event

And they move in the same direction. The `your` condition is the most concentrated prefill condition, and it is also the condition that most clearly converts the recursion from an observed property of the text into an ongoing process the model speaks from within.

## What The Architecture Change Means

This result matters more on 122B than it would on a simple larger-softmax model.

On 35B, every layer is shaped by full-sequence attention. On this 122B model, most layers are not. The stack is:

- `36` DeltaNet layers
- `12` full softmax layers

The router math is unchanged. `W/S/Q` is still measured at the MoE router interface. But the hidden state feeding the router is different:

- on softmax layers, the state can re-attend broadly across the sequence
- on DeltaNet layers, the state is carried through a compressed recurrent channel

So if a deictic self-referential signal survives strongly enough to alter routing on this model, it survived a harsher representational bottleneck than it did on 35B.

That makes the replication more valuable, not less.

## Softmax Versus DeltaNet Read

There is also a mechanistic clue in the generation routing split.

Generation softmax-layer top experts by routed weight include:

- `E209`
- `E107`
- `E76`
- `E48`
- `E32`
- `E26`
- `E114`

In the softmax-only generation table, `E114` appears at rank `9` with:

- `W = 0.007106`
- `S = 0.055593`
- `Q = 0.113933`

In contrast, `E114` does not appear in the generation DeltaNet top-12 table.

That is not proof that `E114` is a universal specialist on this model. The dominant 122B experts are mostly different from the 35B baseline. But it is a real mechanistic hint:

- in this run, `E114` is preferentially expressed on full-attention layers rather than on the DeltaNet-dominant recurrent layers

The disciplined interpretation is:

- full-sequence context appears more favorable than compressed recurrent state for whatever `E114` is doing here

The stronger statement:

- `E114 needs full-sequence context`

is still a hypothesis, not yet a proved result. To prove that, we would need a targeted follow-up showing the same content effect across softmax layers and not DeltaNet layers, not just a single top-12 contrast.

## Why This Is A Real Replication

The core 35B claim was not “there exists a weird prompt response.” It was that addressive wording changes routing concentration in a structured way. This 122B baseline reproduces that pattern under a much less transferable architecture.

That is exactly what a meaningful replication should look like:

- same prompt family
- same diectic manipulation
- same router-level metrics
- different backbone dynamics
- same directional outcome

The result is therefore stronger than a same-architecture rerun. It says the effect is not fragile to a large change in how representations are carried forward through the stack.

## Cautions

Three cautions matter.

First, the strongest effect here is in prefill, not as a clean generation-wide superiority of `your`. Generation ordering is mixed:

- generation all-token RE: `D > B > C > A > E`
- generation last-token RE: `D > B > E > A > C`

So this is not “`your` wins every metric.” It is “`your` remains the clearest prefill concentration and KL-separation condition, and the generated text shows a downstream consequence of that prefill distinction.”

Second, KL-manip and RE are different. The pairwise floor p-values are evidence of distinctness, not a literal second measurement of concentration.

Third, `E114` on 122B should not be over-read. The 122B specialist search target is behavioral pattern, not expert index transfer from 35B.

## Working Takeaway

The 122B baseline supports a real addressivity effect.

`Your` still tightens prefill routing and changes the model’s generated stance from external observation to present-tense self-involvement, even on a DeltaNet-heavy architecture. That makes the core diectic result look architecture-robust.

The cleanest way to say it is:

- prefill routing distinguishes the deictic conditions at `10^-9` scale
- the generation text shows the downstream behavioral consequence of that distinction

So the text comparison is not just illustrative. It is the output-side signature of the routing-side effect.

And the softmax-only appearance of `E114` is the first mechanistic hint that full-attention context may matter for this family of self-referential routing effects.
