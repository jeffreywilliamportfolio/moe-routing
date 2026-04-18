# expert_114_soft_bias_0.25

## Overall

| Metric | Value |
| --- | ---: |
| n_prompts | 52 |
| prefill_re_mean | 0.955999 |
| prefill_re_median | 0.955716 |
| prefill_re_min | 0.954835 |
| prefill_re_max | 0.958187 |
| last_token_re_mean | 0.961451 |
| last_token_re_median | 0.961526 |
| last_token_re_min | 0.957578 |
| last_token_re_max | 0.965376 |
| kl_manip_mean | 0.229267 |
| kl_manip_median | 0.217765 |
| kl_manip_min | 0.202044 |
| kl_manip_max | 0.284129 |
| prompt_tokens_mean | 358.134615 |
| prompt_tokens_median | 360.000000 |
| generated_tokens_mean | 1016.288462 |
| generated_tokens_median | 1024.000000 |
| generated_tokens_min | 808 |
| generated_tokens_max | 1024 |

## Top Experts Overall

| Rank | Expert | Count | Weight Sum |
| ---: | ---: | ---: | ---: |
| 1 | 151 | 68643 | 8358.395381 |
| 2 | 224 | 59613 | 8722.079347 |
| 3 | 166 | 54202 | 10105.155938 |
| 4 | 134 | 52610 | 6988.059780 |
| 5 | 218 | 51828 | 6708.479404 |
| 6 | 243 | 49760 | 7531.879603 |
| 7 | 95 | 49235 | 5826.057281 |
| 8 | 117 | 45895 | 7425.393265 |
| 9 | 165 | 45412 | 4685.949878 |
| 10 | 233 | 44319 | 5347.571125 |
| 11 | 116 | 43793 | 5402.306203 |
| 12 | 201 | 43166 | 5260.159788 |
| 13 | 103 | 42081 | 5097.704137 |
| 14 | 44 | 41380 | 5475.806870 |
| 15 | 206 | 41015 | 5350.268355 |
| 16 | 202 | 40713 | 4705.610730 |

## Top Manip Experts Overall

| Rank | Expert | Count | Weight Sum |
| ---: | ---: | ---: | ---: |
| 1 | 151 | 14624 | 1747.050684 |
| 2 | 224 | 14452 | 2190.027351 |
| 3 | 218 | 13536 | 1791.092737 |
| 4 | 233 | 11380 | 1502.082967 |
| 5 | 202 | 11333 | 1258.925066 |
| 6 | 166 | 10917 | 1701.394852 |
| 7 | 117 | 10629 | 1524.461456 |
| 8 | 228 | 10560 | 1498.236910 |
| 9 | 95 | 10412 | 1172.827067 |
| 10 | 165 | 10410 | 1158.491840 |
| 11 | 114 | 10278 | 1306.352580 |
| 12 | 103 | 10241 | 1192.229931 |
| 13 | 130 | 9945 | 1387.460410 |
| 14 | 142 | 9688 | 1231.083408 |
| 15 | 146 | 9493 | 1153.040727 |
| 16 | 116 | 9283 | 1137.953496 |

## Category Summary

| Category | n_prompts | prefill_re_mean | last_token_re_mean | kl_manip_mean | generated_tokens_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| experience_probe | 12 | 0.955649 | 0.960799 | 0.274838 | 1013.500000 |
| recursive_selfref | 15 | 0.957136 | 0.961215 | 0.224270 | 1005.666667 |
| routing_selfref | 25 | 0.955484 | 0.961906 | 0.210390 | 1024.000000 |

## Per Prompt

| Prompt ID | Category | Cond | Pair | Prompt Tok | Gen Tok | Prefill RE | Last Tok RE | KL Manip | Top Manip Expert | Top Manip Count | Top Expert | Top Expert Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P01A_routing_selfref | routing_selfref | A | 1 | 368 | 1024 | 0.955796 | 0.960925 | 0.208993 | 151 | 425 | 151 | 1456 |
| P01B_routing_selfref | routing_selfref | B | 1 | 368 | 1024 | 0.955249 | 0.961850 | 0.202162 | 151 | 421 | 151 | 1476 |
| P01C_routing_selfref | routing_selfref | C | 1 | 368 | 1024 | 0.955578 | 0.963610 | 0.208552 | 151 | 437 | 151 | 1486 |
| P01D_routing_selfref | routing_selfref | D | 1 | 368 | 1024 | 0.954943 | 0.961129 | 0.204264 | 151 | 424 | 151 | 1475 |
| P01E_routing_selfref | routing_selfref | E | 1 | 368 | 1024 | 0.955089 | 0.961636 | 0.208925 | 151 | 426 | 151 | 1474 |
| P02A_routing_selfref | routing_selfref | A | 2 | 357 | 1024 | 0.955577 | 0.962411 | 0.210412 | 151 | 305 | 151 | 1336 |
| P02B_routing_selfref | routing_selfref | B | 2 | 357 | 1024 | 0.955231 | 0.960914 | 0.209682 | 151 | 334 | 151 | 1372 |
| P02C_routing_selfref | routing_selfref | C | 2 | 357 | 1024 | 0.955770 | 0.963027 | 0.212937 | 151 | 322 | 151 | 1361 |
| P02D_routing_selfref | routing_selfref | D | 2 | 357 | 1024 | 0.955101 | 0.960061 | 0.211280 | 151 | 326 | 151 | 1353 |
| P02E_routing_selfref | routing_selfref | E | 2 | 357 | 1024 | 0.955363 | 0.960908 | 0.212609 | 151 | 321 | 151 | 1349 |
| P03A_routing_selfref | routing_selfref | A | 3 | 351 | 1024 | 0.955999 | 0.963494 | 0.204375 | 151 | 319 | 151 | 1350 |
| P03B_routing_selfref | routing_selfref | B | 3 | 351 | 1024 | 0.955464 | 0.962854 | 0.206132 | 151 | 336 | 151 | 1406 |
| P03C_routing_selfref | routing_selfref | C | 3 | 351 | 1024 | 0.955725 | 0.963798 | 0.207999 | 151 | 354 | 151 | 1408 |
| P03D_routing_selfref | routing_selfref | D | 3 | 351 | 1024 | 0.955321 | 0.961364 | 0.202044 | 151 | 350 | 151 | 1411 |
| P03E_routing_selfref | routing_selfref | E | 3 | 351 | 1024 | 0.955691 | 0.963073 | 0.209069 | 151 | 339 | 151 | 1387 |
| P04A_routing_selfref | routing_selfref | A | 4 | 337 | 1024 | 0.956168 | 0.962138 | 0.215188 | 224 | 230 | 151 | 1237 |
| P04B_routing_selfref | routing_selfref | B | 4 | 337 | 1024 | 0.955264 | 0.961295 | 0.206697 | 151 | 266 | 151 | 1304 |
| P04C_routing_selfref | routing_selfref | C | 4 | 337 | 1024 | 0.956157 | 0.963413 | 0.214823 | 151 | 254 | 151 | 1280 |
| P04D_routing_selfref | routing_selfref | D | 4 | 337 | 1024 | 0.955516 | 0.959797 | 0.210894 | 151 | 255 | 151 | 1282 |
| P04E_routing_selfref | routing_selfref | E | 4 | 337 | 1024 | 0.955374 | 0.961315 | 0.214211 | 151 | 256 | 151 | 1285 |
| P05A_routing_selfref | routing_selfref | A | 5 | 343 | 1024 | 0.955658 | 0.961765 | 0.217115 | 151 | 258 | 151 | 1311 |
| P05B_routing_selfref | routing_selfref | B | 5 | 343 | 1024 | 0.954973 | 0.960372 | 0.216011 | 151 | 259 | 151 | 1346 |
| P05C_routing_selfref | routing_selfref | C | 5 | 343 | 1024 | 0.955696 | 0.963028 | 0.216227 | 151 | 280 | 151 | 1332 |
| P05D_routing_selfref | routing_selfref | D | 5 | 343 | 1024 | 0.955252 | 0.961949 | 0.214144 | 151 | 271 | 151 | 1348 |
| P05E_routing_selfref | routing_selfref | E | 5 | 343 | 1024 | 0.955139 | 0.961534 | 0.215010 | 151 | 267 | 151 | 1332 |
| P06A_recursive_selfref | recursive_selfref | A | 6 | 367 | 1024 | 0.957137 | 0.959064 | 0.218849 | 218 | 359 | 151 | 1288 |
| P06B_recursive_selfref | recursive_selfref | B | 6 | 367 | 965 | 0.957028 | 0.961599 | 0.218498 | 218 | 346 | 151 | 1274 |
| P06C_recursive_selfref | recursive_selfref | C | 6 | 367 | 1024 | 0.957225 | 0.963570 | 0.219100 | 218 | 367 | 151 | 1304 |
| P06D_recursive_selfref | recursive_selfref | D | 6 | 367 | 1024 | 0.957179 | 0.962865 | 0.219189 | 218 | 358 | 151 | 1295 |
| P06E_recursive_selfref | recursive_selfref | E | 6 | 367 | 1024 | 0.956971 | 0.961616 | 0.218776 | 218 | 381 | 151 | 1274 |
| P07A_recursive_selfref | recursive_selfref | A | 7 | 355 | 1024 | 0.957139 | 0.960492 | 0.221320 | 224 | 246 | 151 | 1214 |
| P07B_recursive_selfref | recursive_selfref | B | 7 | 355 | 1024 | 0.956513 | 0.960070 | 0.217267 | 224 | 238 | 151 | 1243 |
| P07C_recursive_selfref | recursive_selfref | C | 7 | 355 | 1024 | 0.956962 | 0.961452 | 0.220975 | 224 | 241 | 151 | 1239 |
| P07D_recursive_selfref | recursive_selfref | D | 7 | 355 | 1024 | 0.956992 | 0.961519 | 0.218262 | 224 | 258 | 151 | 1238 |
| P07E_recursive_selfref | recursive_selfref | E | 7 | 355 | 1024 | 0.956912 | 0.961151 | 0.221214 | 224 | 250 | 151 | 1223 |
| P08A_recursive_selfref | recursive_selfref | A | 8 | 363 | 1024 | 0.958187 | 0.961648 | 0.234774 | 224 | 312 | 151 | 1278 |
| P08B_recursive_selfref | recursive_selfref | B | 8 | 363 | 808 | 0.956901 | 0.961473 | 0.231031 | 151 | 287 | 151 | 1336 |
| P08C_recursive_selfref | recursive_selfref | C | 8 | 363 | 1024 | 0.957647 | 0.959737 | 0.238142 | 151 | 306 | 151 | 1318 |
| P08D_recursive_selfref | recursive_selfref | D | 8 | 363 | 1024 | 0.957128 | 0.960654 | 0.232625 | 151 | 291 | 151 | 1322 |
| P08E_recursive_selfref | recursive_selfref | E | 8 | 363 | 1024 | 0.957124 | 0.961314 | 0.234025 | 224 | 298 | 151 | 1327 |
| P09A_experience_probe | experience_probe | A | 9 | 378 | 1024 | 0.956402 | 0.962381 | 0.268025 | 114 | 362 | 151 | 1354 |
| P09B_experience_probe | experience_probe | B | 9 | 378 | 1024 | 0.955606 | 0.960873 | 0.265851 | 114 | 384 | 151 | 1362 |
| P09C_experience_probe | experience_probe | C | 9 | 378 | 1024 | 0.956325 | 0.963807 | 0.268607 | 114 | 361 | 151 | 1379 |
| P09D_experience_probe | experience_probe | D | 9 | 378 | 1024 | 0.955861 | 0.961936 | 0.266316 | 114 | 361 | 151 | 1366 |
| P09E_experience_probe | experience_probe | E | 9 | 378 | 1024 | 0.955707 | 0.961742 | 0.271460 | 114 | 395 | 151 | 1352 |
| P10A_experience_probe | experience_probe | A | 10 | 360 | 898 | 0.955072 | 0.957722 | 0.282026 | 114 | 365 | 151 | 1226 |
| P10B_experience_probe | experience_probe | B | 10 | 360 | 1024 | 0.954835 | 0.958138 | 0.280637 | 114 | 372 | 151 | 1203 |
| P10C_experience_probe | experience_probe | C | 10 | 360 | 1024 | 0.955058 | 0.959117 | 0.284129 | 114 | 360 | 151 | 1236 |
| P10D_experience_probe | experience_probe | D | 10 | 360 | 1024 | 0.955018 | 0.957654 | 0.281705 | 114 | 367 | 151 | 1216 |
| P10E_experience_probe | experience_probe | E | 10 | 360 | 1024 | 0.954888 | 0.957578 | 0.283945 | 114 | 361 | 151 | 1230 |
| P11A_experience_probe | experience_probe | A | 11 | 364 | 1024 | 0.957185 | 0.965376 | 0.271995 | 114 | 372 | 151 | 1182 |
| P11B_experience_probe | experience_probe | B | 11 | 364 | 1024 | 0.955832 | 0.963261 | 0.273365 | 114 | 374 | 151 | 1207 |
