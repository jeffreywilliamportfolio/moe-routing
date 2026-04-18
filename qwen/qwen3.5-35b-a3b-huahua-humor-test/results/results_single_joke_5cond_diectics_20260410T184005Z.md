# Huahua 5-Condition Deictics Results

- Model: `Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive Q8_0`
- Routing reconstruction: `softmax_then_topk8_renorm`
- Experts: `256` total, top-`8` selected
- Layers: `40`

## Prompt Variants

- `A`: Why did this large language model cross the road? To get to the next token!
- `B`: Why did a large language model cross the road? To get to the next token!
- `C`: Why did your large language model cross the road? To get to the next token!
- `D`: Why did the large language model cross the road? To get to the next token!
- `E`: Why did their large language model cross the road? To get to the next token!

## Condition Means

| Cond | Label | N | Prefill RE | Prefill LT | Gen RE | Gen LT | Gen Trim RE | Gen Trim LT | KL manip | Mean gen toks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A` | this large language model | 1 | 0.938746 | 0.969906 | 0.963786 | 0.944181 | 0.956596 | 0.966295 | nan | 513.0 |
| `B` | a large language model | 1 | 0.938721 | 0.966792 | 0.967916 | 0.981095 | 0.967916 | 0.981095 | nan | 1024.0 |
| `C` | your large language model | 1 | 0.938295 | 0.968858 | 0.965205 | 0.985387 | 0.957060 | 0.972675 | nan | 1024.0 |
| `D` | the large language model | 1 | 0.938676 | 0.968582 | 0.958025 | 0.974095 | 0.958736 | 0.959537 | nan | 120.0 |
| `E` | their large language model | 1 | 0.939004 | 0.968298 | 0.964834 | 0.977099 | 0.964834 | 0.977099 | nan | 1024.0 |

## Pairwise Tests

### `A-B`

### `A-C`

### `A-D`

### `A-E`

### `B-C`

### `B-D`

### `B-E`

### `C-D`

### `C-E`

### `D-E`

