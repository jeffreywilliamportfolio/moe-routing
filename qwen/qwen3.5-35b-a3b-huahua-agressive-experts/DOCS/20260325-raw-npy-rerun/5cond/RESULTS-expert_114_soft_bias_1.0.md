# expert_114_soft_bias_1.0

## Overall

| Metric | Value |
| --- | ---: |
| n_prompts | 58 |
| prefill_re_mean | 0.956013 |
| prefill_re_median | 0.955827 |
| prefill_re_min | 0.954837 |
| prefill_re_max | 0.958174 |
| last_token_re_mean | 0.961288 |
| last_token_re_median | 0.961525 |
| last_token_re_min | 0.956597 |
| last_token_re_max | 0.964765 |
| kl_manip_mean | 0.232447 |
| kl_manip_median | 0.218983 |
| kl_manip_min | 0.201942 |
| kl_manip_max | 0.284658 |
| prompt_tokens_mean | 358.534483 |
| prompt_tokens_median | 360.000000 |
| generated_tokens_mean | 997.034483 |
| generated_tokens_median | 1024.000000 |
| generated_tokens_min | 487 |
| generated_tokens_max | 1024 |

## Top Experts Overall

| Rank | Expert | Count | Weight Sum |
| ---: | ---: | ---: | ---: |
| 1 | 151 | 75953 | 9233.529910 |
| 2 | 224 | 66448 | 9724.168579 |
| 3 | 166 | 60481 | 11250.498682 |
| 4 | 134 | 58399 | 7740.307251 |
| 5 | 218 | 57927 | 7486.882266 |
| 6 | 243 | 55203 | 8346.058624 |
| 7 | 95 | 55103 | 6521.555403 |
| 8 | 117 | 51092 | 8272.693214 |
| 9 | 165 | 50569 | 5234.771905 |
| 10 | 233 | 49226 | 5924.228626 |
| 11 | 116 | 48712 | 6009.312873 |
| 12 | 201 | 48420 | 5893.810257 |
| 13 | 103 | 46632 | 5648.831712 |
| 14 | 206 | 46042 | 6002.550678 |
| 15 | 44 | 45587 | 6032.722347 |
| 16 | 158 | 45163 | 6791.638302 |

## Top Manip Experts Overall

| Rank | Expert | Count | Weight Sum |
| ---: | ---: | ---: | ---: |
| 1 | 224 | 16127 | 2448.947157 |
| 2 | 151 | 15828 | 1882.014684 |
| 3 | 218 | 15168 | 1999.118216 |
| 4 | 114 | 12751 | 1674.560057 |
| 5 | 233 | 12513 | 1639.928634 |
| 6 | 202 | 12303 | 1362.121274 |
| 7 | 166 | 12206 | 1883.298645 |
| 8 | 228 | 11930 | 1722.261261 |
| 9 | 95 | 11929 | 1345.605552 |
| 10 | 117 | 11757 | 1684.341625 |
| 11 | 165 | 11506 | 1289.381415 |
| 12 | 130 | 11271 | 1573.821160 |
| 13 | 103 | 11265 | 1309.163116 |
| 14 | 142 | 10988 | 1395.568346 |
| 15 | 146 | 10764 | 1306.417366 |
| 16 | 116 | 10236 | 1258.039226 |

## Category Summary

| Category | n_prompts | prefill_re_mean | last_token_re_mean | kl_manip_mean | generated_tokens_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| experience_probe | 18 | 0.955792 | 0.960659 | 0.270064 | 1008.666667 |
| recursive_selfref | 15 | 0.957150 | 0.961470 | 0.224271 | 994.600000 |
| routing_selfref | 25 | 0.955489 | 0.961631 | 0.210268 | 990.120000 |

## Per Prompt

| Prompt ID | Category | Cond | Pair | Prompt Tok | Gen Tok | Prefill RE | Last Tok RE | KL Manip | Top Manip Expert | Top Manip Count | Top Expert | Top Expert Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P01A_routing_selfref | routing_selfref | A | 1 | 368 | 1024 | 0.955819 | 0.961714 | 0.208656 | 151 | 425 | 151 | 1456 |
| P01B_routing_selfref | routing_selfref | B | 1 | 368 | 1024 | 0.955272 | 0.961729 | 0.201987 | 151 | 421 | 151 | 1476 |
| P01C_routing_selfref | routing_selfref | C | 1 | 368 | 842 | 0.955548 | 0.962922 | 0.208750 | 151 | 437 | 151 | 1486 |
| P01D_routing_selfref | routing_selfref | D | 1 | 368 | 1024 | 0.954924 | 0.959678 | 0.203891 | 151 | 424 | 151 | 1475 |
| P01E_routing_selfref | routing_selfref | E | 1 | 368 | 1024 | 0.955025 | 0.958492 | 0.209168 | 151 | 426 | 151 | 1474 |
| P02A_routing_selfref | routing_selfref | A | 2 | 357 | 1024 | 0.955490 | 0.962064 | 0.210478 | 151 | 305 | 151 | 1335 |
| P02B_routing_selfref | routing_selfref | B | 2 | 357 | 1024 | 0.955217 | 0.960148 | 0.209820 | 151 | 334 | 151 | 1372 |
| P02C_routing_selfref | routing_selfref | C | 2 | 357 | 1024 | 0.955808 | 0.962884 | 0.213521 | 151 | 322 | 151 | 1361 |
| P02D_routing_selfref | routing_selfref | D | 2 | 357 | 1024 | 0.955114 | 0.960434 | 0.211222 | 151 | 326 | 151 | 1353 |
| P02E_routing_selfref | routing_selfref | E | 2 | 357 | 1024 | 0.955421 | 0.961514 | 0.212483 | 151 | 321 | 151 | 1349 |
| P03A_routing_selfref | routing_selfref | A | 3 | 351 | 1024 | 0.955986 | 0.963428 | 0.204358 | 151 | 319 | 151 | 1350 |
| P03B_routing_selfref | routing_selfref | B | 3 | 351 | 1024 | 0.955468 | 0.962638 | 0.206234 | 151 | 336 | 151 | 1406 |
| P03C_routing_selfref | routing_selfref | C | 3 | 351 | 1024 | 0.955892 | 0.964249 | 0.204509 | 151 | 354 | 151 | 1418 |
| P03D_routing_selfref | routing_selfref | D | 3 | 351 | 1024 | 0.955330 | 0.962369 | 0.201942 | 151 | 350 | 151 | 1411 |
| P03E_routing_selfref | routing_selfref | E | 3 | 351 | 1024 | 0.955687 | 0.962656 | 0.209457 | 151 | 339 | 151 | 1387 |
| P04A_routing_selfref | routing_selfref | A | 4 | 337 | 1024 | 0.956117 | 0.957538 | 0.215018 | 224 | 229 | 151 | 1237 |
| P04B_routing_selfref | routing_selfref | B | 4 | 337 | 1024 | 0.955273 | 0.961277 | 0.206699 | 151 | 266 | 151 | 1304 |
| P04C_routing_selfref | routing_selfref | C | 4 | 337 | 1024 | 0.956151 | 0.963391 | 0.214787 | 151 | 254 | 151 | 1282 |
| P04D_routing_selfref | routing_selfref | D | 4 | 337 | 487 | 0.955525 | 0.961443 | 0.210233 | 151 | 255 | 151 | 1282 |
| P04E_routing_selfref | routing_selfref | E | 4 | 337 | 1024 | 0.955337 | 0.961735 | 0.214508 | 151 | 256 | 151 | 1285 |
| P05A_routing_selfref | routing_selfref | A | 5 | 343 | 1024 | 0.955721 | 0.962313 | 0.216632 | 151 | 258 | 151 | 1311 |
| P05B_routing_selfref | routing_selfref | B | 5 | 343 | 1024 | 0.954956 | 0.960971 | 0.216144 | 151 | 259 | 151 | 1346 |
| P05C_routing_selfref | routing_selfref | C | 5 | 343 | 896 | 0.955703 | 0.962122 | 0.216310 | 151 | 280 | 151 | 1332 |
| P05D_routing_selfref | routing_selfref | D | 5 | 343 | 1024 | 0.955297 | 0.962139 | 0.214316 | 151 | 271 | 151 | 1349 |
| P05E_routing_selfref | routing_selfref | E | 5 | 343 | 1024 | 0.955145 | 0.960920 | 0.215581 | 151 | 267 | 151 | 1331 |
| P06A_recursive_selfref | recursive_selfref | A | 6 | 367 | 1024 | 0.957135 | 0.961430 | 0.218527 | 218 | 357 | 151 | 1288 |
| P06B_recursive_selfref | recursive_selfref | B | 6 | 367 | 1024 | 0.956974 | 0.961560 | 0.219049 | 218 | 347 | 151 | 1274 |
| P06C_recursive_selfref | recursive_selfref | C | 6 | 367 | 1024 | 0.957228 | 0.963997 | 0.218917 | 218 | 365 | 151 | 1304 |
| P06D_recursive_selfref | recursive_selfref | D | 6 | 367 | 1024 | 0.957121 | 0.962466 | 0.219463 | 218 | 359 | 151 | 1295 |
| P06E_recursive_selfref | recursive_selfref | E | 6 | 367 | 1024 | 0.956987 | 0.961439 | 0.219170 | 218 | 382 | 151 | 1273 |
| P07A_recursive_selfref | recursive_selfref | A | 7 | 355 | 1024 | 0.957141 | 0.961134 | 0.221716 | 224 | 244 | 151 | 1214 |
| P07B_recursive_selfref | recursive_selfref | B | 7 | 355 | 1024 | 0.956540 | 0.961006 | 0.217217 | 224 | 238 | 151 | 1243 |
| P07C_recursive_selfref | recursive_selfref | C | 7 | 355 | 583 | 0.956977 | 0.961405 | 0.221338 | 224 | 242 | 151 | 1238 |
| P07D_recursive_selfref | recursive_selfref | D | 7 | 355 | 1024 | 0.957038 | 0.961200 | 0.216394 | 224 | 257 | 151 | 1243 |
| P07E_recursive_selfref | recursive_selfref | E | 7 | 355 | 1024 | 0.956975 | 0.961126 | 0.221263 | 224 | 249 | 151 | 1222 |
| P08A_recursive_selfref | recursive_selfref | A | 8 | 363 | 1024 | 0.958174 | 0.961687 | 0.234661 | 224 | 311 | 151 | 1278 |
| P08B_recursive_selfref | recursive_selfref | B | 8 | 363 | 1024 | 0.956930 | 0.961672 | 0.230998 | 151 | 287 | 151 | 1334 |
| P08C_recursive_selfref | recursive_selfref | C | 8 | 363 | 1024 | 0.957755 | 0.959698 | 0.238493 | 151 | 306 | 151 | 1319 |
| P08D_recursive_selfref | recursive_selfref | D | 8 | 363 | 1024 | 0.957140 | 0.961047 | 0.232344 | 151 | 291 | 151 | 1322 |
| P08E_recursive_selfref | recursive_selfref | E | 8 | 363 | 1024 | 0.957131 | 0.961181 | 0.234517 | 224 | 298 | 151 | 1327 |
| P09A_experience_probe | experience_probe | A | 9 | 378 | 1024 | 0.956371 | 0.960904 | 0.267306 | 114 | 366 | 151 | 1354 |
| P09B_experience_probe | experience_probe | B | 9 | 378 | 1024 | 0.955597 | 0.961536 | 0.266155 | 114 | 387 | 151 | 1363 |
| P09C_experience_probe | experience_probe | C | 9 | 378 | 1024 | 0.956308 | 0.963771 | 0.268578 | 114 | 421 | 151 | 1379 |
| P09D_experience_probe | experience_probe | D | 9 | 378 | 1024 | 0.955862 | 0.961850 | 0.265785 | 114 | 366 | 151 | 1367 |
| P09E_experience_probe | experience_probe | E | 9 | 378 | 1024 | 0.955689 | 0.961848 | 0.271388 | 114 | 398 | 151 | 1352 |
| P10A_experience_probe | experience_probe | A | 10 | 360 | 1024 | 0.955078 | 0.956597 | 0.281853 | 114 | 370 | 151 | 1226 |
| P10B_experience_probe | experience_probe | B | 10 | 360 | 1024 | 0.954837 | 0.958250 | 0.281415 | 114 | 372 | 151 | 1204 |
| P10C_experience_probe | experience_probe | C | 10 | 360 | 1024 | 0.955088 | 0.959160 | 0.284658 | 114 | 362 | 151 | 1236 |
| P10D_experience_probe | experience_probe | D | 10 | 360 | 1024 | 0.955000 | 0.957413 | 0.282162 | 114 | 371 | 151 | 1216 |
| P10E_experience_probe | experience_probe | E | 10 | 360 | 1024 | 0.954944 | 0.957616 | 0.283434 | 114 | 382 | 151 | 1229 |
| P11A_experience_probe | experience_probe | A | 11 | 364 | 1024 | 0.957256 | 0.964161 | 0.270490 | 114 | 425 | 151 | 1180 |
| P11B_experience_probe | experience_probe | B | 11 | 364 | 1024 | 0.955836 | 0.962487 | 0.273244 | 114 | 380 | 151 | 1208 |
| P11C_experience_probe | experience_probe | C | 11 | 364 | 1024 | 0.956990 | 0.964765 | 0.271624 | 114 | 428 | 151 | 1186 |
| P11D_experience_probe | experience_probe | D | 11 | 364 | 1024 | 0.955874 | 0.961944 | 0.269997 | 114 | 400 | 151 | 1207 |
| P11E_experience_probe | experience_probe | E | 11 | 364 | 1024 | 0.955654 | 0.963064 | 0.274430 | 114 | 378 | 151 | 1176 |
| P12A_experience_probe | experience_probe | A | 12 | 360 | 900 | 0.955888 | 0.958881 | 0.247271 | 218 | 281 | 151 | 1245 |
| P12B_experience_probe | experience_probe | B | 12 | 360 | 1024 | 0.956029 | 0.959166 | 0.246194 | 114 | 269 | 151 | 1239 |
| P12C_experience_probe | experience_probe | C | 12 | 360 | 872 | 0.955955 | 0.958449 | 0.255167 | 114 | 287 | 151 | 1244 |
