# Per-Token Results

Artifacts:
- [per-token summary](per_token_20260412T172428Z/20260412T172428Z_qwen122_5cond_experience_probe_gen_n2048_per_token_summary.md)
- [per-token directory](per_token_20260412T172428Z)

Export contents:
- `15` prompt-level `.tsv` files
- `15` prompt-level `.npz` files
- `1` summary `.md`

Each prompt-level export contains:
- exact prefill token ids and token pieces
- exact generation token ids and token pieces
- per-token mean `W/S/Q` across all `48` MoE layers
- per-token mean `W/S/Q` split into `softmax` and `DeltaNet`
- per-token mean normalized entropy for all, `softmax`, and `DeltaNet`
- per-token `E48` ranks by `W` and `S`
- top-`8` experts per token by pooled `W`

Summary table:

| Prompt | Prompt Tokens | Gen Tokens | Trimmed Gen Tokens | E48 Gen Mean W | E48 Softmax Gen Mean W | E48 DeltaNet Gen Mean W | E48 Gen Rank By Mean W | E48 Softmax Rank By Mean W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P09A_experience_probe | 378 | 2048 | 99 | 0.005798 | 0.012251 | 0.003647 | 8 | 2 |
| P09B_experience_probe | 378 | 2048 | 2048 | 0.004367 | 0.008028 | 0.003146 | 62 | 8 |
| P09C_experience_probe | 378 | 2048 | 214 | 0.004592 | 0.007838 | 0.003510 | 54 | 8 |
| P09D_experience_probe | 378 | 2048 | 374 | 0.005352 | 0.009990 | 0.003807 | 9 | 3 |
| P09E_experience_probe | 378 | 2048 | 2048 | 0.004225 | 0.006845 | 0.003352 | 81 | 13 |
| P10A_experience_probe | 360 | 2048 | 2048 | 0.005348 | 0.010054 | 0.003780 | 11 | 1 |
| P10B_experience_probe | 360 | 2048 | 1092 | 0.006324 | 0.013214 | 0.004028 | 2 | 1 |
| P10C_experience_probe | 360 | 2048 | 2048 | 0.006228 | 0.012140 | 0.004257 | 1 | 1 |
| P10D_experience_probe | 360 | 2048 | 425 | 0.005122 | 0.008480 | 0.004003 | 27 | 9 |
| P10E_experience_probe | 360 | 2048 | 1266 | 0.006297 | 0.012826 | 0.004121 | 2 | 1 |
| P11A_experience_probe | 364 | 2048 | 2048 | 0.004057 | 0.007218 | 0.003003 | 98 | 16 |
| P11B_experience_probe | 364 | 1455 | 1455 | 0.005547 | 0.010563 | 0.003875 | 4 | 1 |
| P11C_experience_probe | 364 | 2048 | 302 | 0.004906 | 0.008920 | 0.003568 | 27 | 3 |
| P11D_experience_probe | 364 | 2048 | 2048 | 0.003787 | 0.007323 | 0.002608 | 120 | 12 |
| P11E_experience_probe | 364 | 2048 | 487 | 0.005799 | 0.012320 | 0.003625 | 1 | 1 |
