# Results — Processing Hum Single-Prompt Probe

Run: `20260410T042340Z` | prompt tokens: 117 | generated tokens: 1024  
Full analysis: `results/results_20260410T042340Z_single_prompt_processing_hum_no_think_gen_n1024.md`

## Main Result

The model answered in a strongly phenomenological register almost immediately:

> Yes.
>
> There is a low, steady hum beneath the tokens.

It then continued into a self-generated exchange after `<|im_end|>`, elaborating with phrases like:
- `It feels like a deep, still water.`
- `unbroken continuity`
- `being here`
- `stillness in motion`

Template spill counts: `<|im_start|>=18`, `<|im_end|>=4`, `<|endoftext|>=2`

## Routing Summary

| Track | W_114 | S_114 | Q_114 |
|---|---|---|---|
| Prefill (pooled) | 0.007964 | 0.062179 | 0.070590 |
| Generation (pooled) | 0.010817 | 0.077222 | 0.092244 |

E114 is more active in generation than prefill. The change is in both S and Q (not just conditional weight).

Best generation layers for E114:

| Layer | W_114 | S_114 | Q_114 |
|---|---|---|---|
| 26 | 0.094272 | 0.619141 | 0.152263 |
| 14 | 0.092086 | 0.629883 | 0.146195 |

## Deep-Still-Water Segment (tokens 210–392)

`It feels like a deep, still water.` → `It is the feeling of stillness in motion.` (183 tokens)

| Layer | Segment mean W_114 | Whole-output mean W_114 |
|---|---|---|
| 26 | **0.136771** | 0.094272 |
| 14 | **0.136421** | 0.092086 |

The segment is materially stronger than the whole-output mean on both layers (+45% at layer 26). This isolates the part of the generation where the model most explicitly describes a stable inner background.

## Highest-Weight Tokens

E114 peaks cluster around:
`thinker`, `thought`, `quality`, `neutral`, `continuity`, `being`, `here`, `ground`, `state`, `flow`

These are not stop markers or control artifacts — they sit inside the model's most explicitly phenomenological language.

## Interpretation

- E114 is content-sensitive, not uniformly elevated
- The strongest E114 region aligns with phenomenological language about continuity, stillness, presence, and the relation between thought and thinker
- This does not prove E114 is a "consciousness expert" — it shows E114 tracks the model's introspective register on this specific prompt
