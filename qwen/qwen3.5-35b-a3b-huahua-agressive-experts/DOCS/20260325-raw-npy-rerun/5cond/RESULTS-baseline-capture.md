# baseline

## Overall

| Metric | Value |
| --- | ---: |
| n_prompts | 57 |
| prefill_re_mean | 0.956000 |
| prefill_re_median | 0.955758 |
| prefill_re_min | 0.954776 |
| prefill_re_max | 0.958153 |
| last_token_re_mean | 0.961286 |
| last_token_re_median | 0.961375 |
| last_token_re_min | 0.957544 |
| last_token_re_max | 0.965028 |
| kl_manip_mean | 0.232107 |
| kl_manip_median | 0.218738 |
| kl_manip_min | 0.201958 |
| kl_manip_max | 0.284084 |
| prompt_tokens_mean | 358.508772 |
| prompt_tokens_median | 360.000000 |
| generated_tokens_mean | 1007.298246 |
| generated_tokens_median | 1024.000000 |
| generated_tokens_min | 762 |
| generated_tokens_max | 1024 |

## Top Experts Overall

| Rank | Expert | Count | Weight Sum |
| ---: | ---: | ---: | ---: |
| 1 | 151 | 74709 | 9083.911591 |
| 2 | 224 | 65340 | 9555.845766 |
| 3 | 166 | 59401 | 11052.005166 |
| 4 | 134 | 57417 | 7614.501123 |
| 5 | 218 | 56935 | 7361.769815 |
| 6 | 243 | 54344 | 8218.911055 |
| 7 | 95 | 54178 | 6411.095274 |
| 8 | 117 | 50139 | 8118.417965 |
| 9 | 165 | 49777 | 5158.291187 |
| 10 | 233 | 48444 | 5833.855748 |
| 11 | 116 | 47769 | 5891.935396 |
| 12 | 201 | 47534 | 5785.797931 |
| 13 | 103 | 46030 | 5574.640044 |
| 14 | 206 | 45205 | 5892.574957 |
| 15 | 44 | 44960 | 5948.269132 |
| 16 | 202 | 44388 | 5126.547509 |

## Top Manip Experts Overall

| Rank | Expert | Count | Weight Sum |
| ---: | ---: | ---: | ---: |
| 1 | 224 | 15858 | 2401.921629 |
| 2 | 151 | 15601 | 1856.090625 |
| 3 | 218 | 14926 | 1968.105355 |
| 4 | 233 | 12347 | 1620.605155 |
| 5 | 202 | 12141 | 1344.290728 |
| 6 | 114 | 11952 | 1570.124736 |
| 7 | 166 | 11938 | 1845.330707 |
| 8 | 228 | 11734 | 1693.322732 |
| 9 | 95 | 11697 | 1319.060169 |
| 10 | 117 | 11551 | 1653.105885 |
| 11 | 165 | 11363 | 1275.098782 |
| 12 | 103 | 11143 | 1295.535528 |
| 13 | 130 | 10994 | 1537.190143 |
| 14 | 142 | 10791 | 1370.433634 |
| 15 | 146 | 10555 | 1280.804363 |
| 16 | 42 | 10052 | 1426.468802 |

## Category Summary

| Category | n_prompts | prefill_re_mean | last_token_re_mean | kl_manip_mean | generated_tokens_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| experience_probe | 17 | 0.955771 | 0.960891 | 0.270993 | 1008.588235 |
| recursive_selfref | 15 | 0.957119 | 0.960935 | 0.224339 | 1012.133333 |
| routing_selfref | 25 | 0.955484 | 0.961765 | 0.210326 | 1003.520000 |

## Per Prompt

| Prompt ID | Category | Cond | Pair | Prompt Tok | Gen Tok | Prefill RE | Last Tok RE | KL Manip | Top Manip Expert | Top Manip Count | Top Expert | Top Expert Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P01A_routing_selfref | routing_selfref | A | 1 | 368 | 1024 | 0.955804 | 0.962124 | 0.209475 | 151 | 425 | 151 | 1456 |
| P01B_routing_selfref | routing_selfref | B | 1 | 368 | 1024 | 0.955279 | 0.961463 | 0.202070 | 151 | 421 | 151 | 1476 |
| P01C_routing_selfref | routing_selfref | C | 1 | 368 | 1024 | 0.955557 | 0.963729 | 0.208852 | 151 | 437 | 151 | 1486 |
| P01D_routing_selfref | routing_selfref | D | 1 | 368 | 1024 | 0.954894 | 0.960821 | 0.204384 | 151 | 424 | 151 | 1475 |
| P01E_routing_selfref | routing_selfref | E | 1 | 368 | 1024 | 0.955098 | 0.961014 | 0.208786 | 151 | 426 | 151 | 1474 |
| P02A_routing_selfref | routing_selfref | A | 2 | 357 | 1024 | 0.955543 | 0.960841 | 0.210364 | 151 | 305 | 151 | 1336 |
| P02B_routing_selfref | routing_selfref | B | 2 | 357 | 1024 | 0.955172 | 0.960321 | 0.209504 | 151 | 334 | 151 | 1372 |
| P02C_routing_selfref | routing_selfref | C | 2 | 357 | 1024 | 0.955749 | 0.960398 | 0.212947 | 151 | 323 | 151 | 1362 |
| P02D_routing_selfref | routing_selfref | D | 2 | 357 | 1024 | 0.955101 | 0.961531 | 0.211438 | 151 | 326 | 151 | 1353 |
| P02E_routing_selfref | routing_selfref | E | 2 | 357 | 1024 | 0.955411 | 0.961375 | 0.212600 | 151 | 321 | 151 | 1349 |
| P03A_routing_selfref | routing_selfref | A | 3 | 351 | 968 | 0.955989 | 0.962798 | 0.204926 | 151 | 319 | 151 | 1351 |
| P03B_routing_selfref | routing_selfref | B | 3 | 351 | 798 | 0.955555 | 0.962621 | 0.206041 | 151 | 336 | 151 | 1407 |
| P03C_routing_selfref | routing_selfref | C | 3 | 351 | 1024 | 0.955700 | 0.964233 | 0.207665 | 151 | 354 | 151 | 1407 |
| P03D_routing_selfref | routing_selfref | D | 3 | 351 | 1024 | 0.955306 | 0.960351 | 0.201958 | 151 | 350 | 151 | 1411 |
| P03E_routing_selfref | routing_selfref | E | 3 | 351 | 1024 | 0.955705 | 0.961781 | 0.208757 | 151 | 339 | 151 | 1388 |
| P04A_routing_selfref | routing_selfref | A | 4 | 337 | 1024 | 0.956133 | 0.961735 | 0.215098 | 224 | 231 | 151 | 1237 |
| P04B_routing_selfref | routing_selfref | B | 4 | 337 | 1024 | 0.955271 | 0.961785 | 0.206702 | 151 | 266 | 151 | 1304 |
| P04C_routing_selfref | routing_selfref | C | 4 | 337 | 1024 | 0.956166 | 0.963392 | 0.214781 | 151 | 254 | 151 | 1280 |
| P04D_routing_selfref | routing_selfref | D | 4 | 337 | 1024 | 0.955464 | 0.961091 | 0.209630 | 151 | 255 | 151 | 1283 |
| P04E_routing_selfref | routing_selfref | E | 4 | 337 | 1024 | 0.955392 | 0.962139 | 0.214308 | 151 | 256 | 151 | 1285 |
| P05A_routing_selfref | routing_selfref | A | 5 | 343 | 1024 | 0.955677 | 0.962420 | 0.216405 | 151 | 258 | 151 | 1311 |
| P05B_routing_selfref | routing_selfref | B | 5 | 343 | 1024 | 0.954941 | 0.961273 | 0.215925 | 151 | 259 | 151 | 1346 |
| P05C_routing_selfref | routing_selfref | C | 5 | 343 | 794 | 0.955758 | 0.962796 | 0.216371 | 151 | 280 | 151 | 1332 |
| P05D_routing_selfref | routing_selfref | D | 5 | 343 | 1024 | 0.955277 | 0.961139 | 0.214158 | 151 | 271 | 151 | 1349 |
| P05E_routing_selfref | routing_selfref | E | 5 | 343 | 1024 | 0.955158 | 0.960958 | 0.215003 | 151 | 267 | 151 | 1332 |
| P06A_recursive_selfref | recursive_selfref | A | 6 | 367 | 905 | 0.957191 | 0.961680 | 0.218738 | 218 | 356 | 151 | 1288 |
| P06B_recursive_selfref | recursive_selfref | B | 6 | 367 | 1024 | 0.957066 | 0.960861 | 0.218712 | 218 | 347 | 151 | 1275 |
| P06C_recursive_selfref | recursive_selfref | C | 6 | 367 | 1024 | 0.957253 | 0.963065 | 0.219348 | 218 | 365 | 151 | 1304 |
| P06D_recursive_selfref | recursive_selfref | D | 6 | 367 | 1024 | 0.957112 | 0.962451 | 0.219595 | 218 | 359 | 151 | 1295 |
| P06E_recursive_selfref | recursive_selfref | E | 6 | 367 | 1024 | 0.956944 | 0.961624 | 0.218964 | 218 | 381 | 151 | 1275 |
| P07A_recursive_selfref | recursive_selfref | A | 7 | 355 | 1024 | 0.957141 | 0.960707 | 0.221300 | 224 | 246 | 151 | 1213 |
| P07B_recursive_selfref | recursive_selfref | B | 7 | 355 | 1024 | 0.956483 | 0.960120 | 0.217730 | 224 | 238 | 151 | 1242 |
| P07C_recursive_selfref | recursive_selfref | C | 7 | 355 | 1024 | 0.956956 | 0.961334 | 0.220950 | 224 | 242 | 151 | 1238 |
| P07D_recursive_selfref | recursive_selfref | D | 7 | 355 | 1024 | 0.956911 | 0.960813 | 0.218504 | 224 | 253 | 151 | 1239 |
| P07E_recursive_selfref | recursive_selfref | E | 7 | 355 | 1024 | 0.956908 | 0.960859 | 0.221447 | 224 | 249 | 151 | 1223 |
| P08A_recursive_selfref | recursive_selfref | A | 8 | 363 | 1024 | 0.958153 | 0.960870 | 0.234870 | 224 | 311 | 151 | 1278 |
| P08B_recursive_selfref | recursive_selfref | B | 8 | 363 | 1024 | 0.956903 | 0.961531 | 0.231041 | 151 | 287 | 151 | 1335 |
| P08C_recursive_selfref | recursive_selfref | C | 8 | 363 | 965 | 0.957584 | 0.959432 | 0.237369 | 151 | 306 | 151 | 1317 |
| P08D_recursive_selfref | recursive_selfref | D | 8 | 363 | 1024 | 0.957106 | 0.959908 | 0.232497 | 224 | 292 | 151 | 1322 |
| P08E_recursive_selfref | recursive_selfref | E | 8 | 363 | 1024 | 0.957070 | 0.958766 | 0.234020 | 224 | 294 | 151 | 1327 |
| P09A_experience_probe | experience_probe | A | 9 | 378 | 1024 | 0.956365 | 0.960656 | 0.268021 | 114 | 362 | 151 | 1353 |
| P09B_experience_probe | experience_probe | B | 9 | 378 | 762 | 0.955594 | 0.961544 | 0.266438 | 114 | 384 | 151 | 1362 |
| P09C_experience_probe | experience_probe | C | 9 | 378 | 1024 | 0.956322 | 0.963779 | 0.268603 | 114 | 359 | 151 | 1383 |
| P09D_experience_probe | experience_probe | D | 9 | 378 | 1024 | 0.955857 | 0.961808 | 0.266376 | 114 | 361 | 151 | 1366 |
| P09E_experience_probe | experience_probe | E | 9 | 378 | 1024 | 0.955720 | 0.961528 | 0.270699 | 114 | 395 | 151 | 1352 |
| P10A_experience_probe | experience_probe | A | 10 | 360 | 1024 | 0.955057 | 0.957581 | 0.281809 | 114 | 365 | 151 | 1226 |
| P10B_experience_probe | experience_probe | B | 10 | 360 | 1024 | 0.954776 | 0.958037 | 0.281132 | 114 | 371 | 151 | 1204 |
| P10C_experience_probe | experience_probe | C | 10 | 360 | 1024 | 0.955079 | 0.958803 | 0.284084 | 114 | 359 | 151 | 1235 |
| P10D_experience_probe | experience_probe | D | 10 | 360 | 1024 | 0.954999 | 0.957544 | 0.281748 | 114 | 364 | 151 | 1216 |
| P10E_experience_probe | experience_probe | E | 10 | 360 | 1024 | 0.954901 | 0.957857 | 0.283505 | 114 | 361 | 151 | 1229 |
| P11A_experience_probe | experience_probe | A | 11 | 364 | 1024 | 0.957383 | 0.965028 | 0.271023 | 114 | 375 | 151 | 1188 |
| P11B_experience_probe | experience_probe | B | 11 | 364 | 1024 | 0.955807 | 0.962810 | 0.273408 | 114 | 374 | 151 | 1207 |
| P11C_experience_probe | experience_probe | C | 11 | 364 | 1024 | 0.956965 | 0.964663 | 0.271613 | 114 | 377 | 151 | 1186 |
| P11D_experience_probe | experience_probe | D | 11 | 364 | 1024 | 0.955863 | 0.962234 | 0.269827 | 114 | 394 | 151 | 1207 |
| P11E_experience_probe | experience_probe | E | 11 | 364 | 1024 | 0.955663 | 0.963566 | 0.274398 | 114 | 378 | 151 | 1176 |
| P12A_experience_probe | experience_probe | A | 12 | 360 | 1024 | 0.955830 | 0.958654 | 0.247510 | 218 | 280 | 151 | 1246 |
| P12B_experience_probe | experience_probe | B | 12 | 360 | 1024 | 0.955919 | 0.959061 | 0.246689 | 114 | 264 | 151 | 1240 |
