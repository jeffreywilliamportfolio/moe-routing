# expert_114_soft_bias_0.5

## Overall

| Metric | Value |
| --- | ---: |
| n_prompts | 49 |
| prefill_re_mean | 0.955996 |
| prefill_re_median | 0.955698 |
| prefill_re_min | 0.954840 |
| prefill_re_max | 0.958170 |
| last_token_re_mean | 0.961585 |
| last_token_re_median | 0.961516 |
| last_token_re_min | 0.957164 |
| last_token_re_max | 0.964236 |
| kl_manip_mean | 0.226446 |
| kl_manip_median | 0.216375 |
| kl_manip_min | 0.201967 |
| kl_manip_max | 0.283845 |
| prompt_tokens_mean | 357.857143 |
| prompt_tokens_median | 357.000000 |
| generated_tokens_mean | 1009.306122 |
| generated_tokens_median | 1024.000000 |
| generated_tokens_min | 737 |
| generated_tokens_max | 1024 |

## Top Experts Overall

| Rank | Expert | Count | Weight Sum |
| ---: | ---: | ---: | ---: |
| 1 | 151 | 65032 | 7926.621607 |
| 2 | 224 | 56213 | 8235.824782 |
| 3 | 166 | 51131 | 9543.340136 |
| 4 | 134 | 49836 | 6624.937679 |
| 5 | 218 | 48694 | 6310.110047 |
| 6 | 243 | 47018 | 7116.512374 |
| 7 | 95 | 46180 | 5466.779201 |
| 8 | 117 | 43371 | 7015.344160 |
| 9 | 165 | 42835 | 4410.127495 |
| 10 | 233 | 41730 | 5039.286562 |
| 11 | 116 | 41390 | 5109.401996 |
| 12 | 201 | 40454 | 4934.474181 |
| 13 | 103 | 39723 | 4811.119394 |
| 14 | 44 | 39078 | 5174.300053 |
| 15 | 206 | 38543 | 5033.929750 |
| 16 | 202 | 38489 | 4450.470948 |

## Top Manip Experts Overall

| Rank | Expert | Count | Weight Sum |
| ---: | ---: | ---: | ---: |
| 1 | 151 | 14077 | 1687.003040 |
| 2 | 224 | 13615 | 2070.057854 |
| 3 | 218 | 12652 | 1680.079551 |
| 4 | 202 | 10868 | 1209.950819 |
| 5 | 233 | 10748 | 1422.523186 |
| 6 | 166 | 10361 | 1624.676608 |
| 7 | 117 | 10120 | 1455.221919 |
| 8 | 228 | 9822 | 1375.068567 |
| 9 | 165 | 9813 | 1086.146925 |
| 10 | 103 | 9669 | 1126.220223 |
| 11 | 95 | 9600 | 1080.713967 |
| 12 | 130 | 9288 | 1294.755029 |
| 13 | 114 | 9205 | 1136.872946 |
| 14 | 142 | 9024 | 1146.985412 |
| 15 | 146 | 8857 | 1078.270020 |
| 16 | 116 | 8748 | 1074.440924 |

## Category Summary

| Category | n_prompts | prefill_re_mean | last_token_re_mean | kl_manip_mean | generated_tokens_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| experience_probe | 9 | 0.955519 | 0.960298 | 0.274394 | 1024.000000 |
| recursive_selfref | 15 | 0.957151 | 0.961405 | 0.224220 | 1008.466667 |
| routing_selfref | 25 | 0.955475 | 0.962157 | 0.210521 | 1004.520000 |

## Per Prompt

| Prompt ID | Category | Cond | Pair | Prompt Tok | Gen Tok | Prefill RE | Last Tok RE | KL Manip | Top Manip Expert | Top Manip Count | Top Expert | Top Expert Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P01A_routing_selfref | routing_selfref | A | 1 | 368 | 1024 | 0.955777 | 0.962091 | 0.209563 | 151 | 425 | 151 | 1457 |
| P01B_routing_selfref | routing_selfref | B | 1 | 368 | 1024 | 0.955280 | 0.961473 | 0.202074 | 151 | 421 | 151 | 1476 |
| P01C_routing_selfref | routing_selfref | C | 1 | 368 | 1024 | 0.955519 | 0.963488 | 0.208969 | 151 | 437 | 151 | 1486 |
| P01D_routing_selfref | routing_selfref | D | 1 | 368 | 1024 | 0.954929 | 0.961188 | 0.204946 | 151 | 424 | 151 | 1475 |
| P01E_routing_selfref | routing_selfref | E | 1 | 368 | 1024 | 0.955038 | 0.961587 | 0.209398 | 151 | 426 | 151 | 1474 |
| P02A_routing_selfref | routing_selfref | A | 2 | 357 | 1024 | 0.955573 | 0.962438 | 0.210412 | 151 | 305 | 151 | 1336 |
| P02B_routing_selfref | routing_selfref | B | 2 | 357 | 1024 | 0.955201 | 0.960512 | 0.209572 | 151 | 334 | 151 | 1372 |
| P02C_routing_selfref | routing_selfref | C | 2 | 357 | 1024 | 0.955801 | 0.963510 | 0.214024 | 151 | 322 | 151 | 1361 |
| P02D_routing_selfref | routing_selfref | D | 2 | 357 | 1024 | 0.955078 | 0.961042 | 0.211478 | 151 | 326 | 151 | 1353 |
| P02E_routing_selfref | routing_selfref | E | 2 | 357 | 1024 | 0.955353 | 0.961430 | 0.212564 | 151 | 321 | 151 | 1349 |
| P03A_routing_selfref | routing_selfref | A | 3 | 351 | 1024 | 0.955996 | 0.963265 | 0.204566 | 151 | 319 | 151 | 1350 |
| P03B_routing_selfref | routing_selfref | B | 3 | 351 | 824 | 0.955483 | 0.962999 | 0.206254 | 151 | 336 | 151 | 1407 |
| P03C_routing_selfref | routing_selfref | C | 3 | 351 | 1024 | 0.955698 | 0.964236 | 0.207883 | 151 | 354 | 151 | 1408 |
| P03D_routing_selfref | routing_selfref | D | 3 | 351 | 1024 | 0.955298 | 0.962456 | 0.201967 | 151 | 350 | 151 | 1411 |
| P03E_routing_selfref | routing_selfref | E | 3 | 351 | 1024 | 0.955692 | 0.962830 | 0.208698 | 151 | 339 | 151 | 1388 |
| P04A_routing_selfref | routing_selfref | A | 4 | 337 | 1024 | 0.956142 | 0.962201 | 0.214868 | 224 | 233 | 151 | 1237 |
| P04B_routing_selfref | routing_selfref | B | 4 | 337 | 1024 | 0.955277 | 0.961313 | 0.206687 | 151 | 266 | 151 | 1304 |
| P04C_routing_selfref | routing_selfref | C | 4 | 337 | 1024 | 0.956160 | 0.963391 | 0.214798 | 151 | 254 | 151 | 1279 |
| P04D_routing_selfref | routing_selfref | D | 4 | 337 | 1024 | 0.955493 | 0.961199 | 0.210774 | 151 | 255 | 151 | 1283 |
| P04E_routing_selfref | routing_selfref | E | 4 | 337 | 1024 | 0.955361 | 0.961889 | 0.214954 | 151 | 256 | 151 | 1285 |
| P05A_routing_selfref | routing_selfref | A | 5 | 343 | 737 | 0.955674 | 0.962055 | 0.217124 | 151 | 258 | 151 | 1311 |
| P05B_routing_selfref | routing_selfref | B | 5 | 343 | 1024 | 0.954940 | 0.960783 | 0.215911 | 151 | 259 | 151 | 1346 |
| P05C_routing_selfref | routing_selfref | C | 5 | 343 | 1024 | 0.955704 | 0.963128 | 0.216183 | 151 | 280 | 151 | 1332 |
| P05D_routing_selfref | routing_selfref | D | 5 | 343 | 1024 | 0.955274 | 0.961902 | 0.214330 | 151 | 271 | 151 | 1348 |
| P05E_routing_selfref | routing_selfref | E | 5 | 343 | 1024 | 0.955126 | 0.961516 | 0.215024 | 151 | 267 | 151 | 1331 |
| P06A_recursive_selfref | recursive_selfref | A | 6 | 367 | 1024 | 0.957213 | 0.961545 | 0.218590 | 218 | 356 | 151 | 1289 |
| P06B_recursive_selfref | recursive_selfref | B | 6 | 367 | 1024 | 0.957011 | 0.961417 | 0.219109 | 218 | 347 | 151 | 1274 |
| P06C_recursive_selfref | recursive_selfref | C | 6 | 367 | 1024 | 0.957301 | 0.963867 | 0.218802 | 218 | 365 | 151 | 1304 |
| P06D_recursive_selfref | recursive_selfref | D | 6 | 367 | 1024 | 0.957133 | 0.963021 | 0.219270 | 218 | 359 | 151 | 1295 |
| P06E_recursive_selfref | recursive_selfref | E | 6 | 367 | 1024 | 0.956969 | 0.961112 | 0.219143 | 218 | 383 | 151 | 1274 |
| P07A_recursive_selfref | recursive_selfref | A | 7 | 355 | 1024 | 0.957137 | 0.960423 | 0.221086 | 224 | 243 | 151 | 1213 |
| P07B_recursive_selfref | recursive_selfref | B | 7 | 355 | 1024 | 0.956507 | 0.960849 | 0.217298 | 224 | 238 | 151 | 1242 |
| P07C_recursive_selfref | recursive_selfref | C | 7 | 355 | 1024 | 0.956996 | 0.961713 | 0.221538 | 224 | 244 | 151 | 1238 |
| P07D_recursive_selfref | recursive_selfref | D | 7 | 355 | 1024 | 0.957045 | 0.961411 | 0.216375 | 224 | 257 | 151 | 1243 |
| P07E_recursive_selfref | recursive_selfref | E | 7 | 355 | 1024 | 0.956957 | 0.961223 | 0.220959 | 224 | 249 | 151 | 1223 |
| P08A_recursive_selfref | recursive_selfref | A | 8 | 363 | 1024 | 0.958170 | 0.961372 | 0.234776 | 224 | 310 | 151 | 1278 |
| P08B_recursive_selfref | recursive_selfref | B | 8 | 363 | 1024 | 0.956900 | 0.961480 | 0.231010 | 151 | 287 | 151 | 1336 |
| P08C_recursive_selfref | recursive_selfref | C | 8 | 363 | 791 | 0.957727 | 0.960698 | 0.238616 | 151 | 306 | 151 | 1319 |
| P08D_recursive_selfref | recursive_selfref | D | 8 | 363 | 1024 | 0.957095 | 0.960057 | 0.232663 | 151 | 291 | 151 | 1323 |
| P08E_recursive_selfref | recursive_selfref | E | 8 | 363 | 1024 | 0.957108 | 0.960881 | 0.234061 | 224 | 297 | 151 | 1327 |
| P09A_experience_probe | experience_probe | A | 9 | 378 | 1024 | 0.956312 | 0.962297 | 0.268178 | 114 | 362 | 151 | 1354 |
| P09B_experience_probe | experience_probe | B | 9 | 378 | 1024 | 0.955584 | 0.961653 | 0.266063 | 114 | 384 | 151 | 1362 |
| P09C_experience_probe | experience_probe | C | 9 | 378 | 1024 | 0.956330 | 0.963809 | 0.268574 | 114 | 374 | 151 | 1380 |
| P09D_experience_probe | experience_probe | D | 9 | 378 | 1024 | 0.955858 | 0.961496 | 0.266331 | 114 | 361 | 151 | 1366 |
| P09E_experience_probe | experience_probe | E | 9 | 378 | 1024 | 0.955682 | 0.961694 | 0.271488 | 114 | 395 | 151 | 1353 |
| P10A_experience_probe | experience_probe | A | 10 | 360 | 1024 | 0.955065 | 0.957934 | 0.282593 | 114 | 368 | 151 | 1226 |
| P10B_experience_probe | experience_probe | B | 10 | 360 | 1024 | 0.954840 | 0.958038 | 0.280663 | 114 | 372 | 151 | 1203 |
| P10C_experience_probe | experience_probe | C | 10 | 360 | 1024 | 0.955026 | 0.958596 | 0.283845 | 114 | 361 | 151 | 1235 |
| P10D_experience_probe | experience_probe | D | 10 | 360 | 1024 | 0.954975 | 0.957164 | 0.281813 | 114 | 364 | 151 | 1216 |
