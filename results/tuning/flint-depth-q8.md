| model                          |       size |     params | backend    | ngl |  n_cpu_moe | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | -----: | -----: | --: | --------------: | -------------------: |
| flint ?B Q4_K - Medium         |  23.33 GiB |    39.79 B | CUDA       |  99 |         10 |   q8_0 |   q8_0 |   1 |          pp2048 |       1064.48 ± 0.00 |
| flint ?B Q4_K - Medium         |  23.33 GiB |    39.79 B | CUDA       |  99 |         10 |   q8_0 |   q8_0 |   1 |           tg128 |        102.32 ± 0.00 |
| flint ?B Q4_K - Medium         |  23.33 GiB |    39.79 B | CUDA       |  99 |         10 |   q8_0 |   q8_0 |   1 | pp2048 @ d32768 |        935.03 ± 0.00 |
| flint ?B Q4_K - Medium         |  23.33 GiB |    39.79 B | CUDA       |  99 |         10 |   q8_0 |   q8_0 |   1 |  tg128 @ d32768 |         76.13 ± 0.00 |
| flint ?B Q4_K - Medium         |  23.33 GiB |    39.79 B | CUDA       |  99 |         10 |   q8_0 |   q8_0 |   1 | pp2048 @ d131072 |        684.27 ± 0.00 |
| flint ?B Q4_K - Medium         |  23.33 GiB |    39.79 B | CUDA       |  99 |         10 |   q8_0 |   q8_0 |   1 | tg128 @ d131072 |         44.67 ± 0.00 |
| flint ?B Q4_K - Medium         |  23.33 GiB |    39.79 B | CUDA       |  99 |         10 |   q8_0 |   q8_0 |   1 | pp2048 @ d225000 |        531.02 ± 0.00 |
| flint ?B Q4_K - Medium         |  23.33 GiB |    39.79 B | CUDA       |  99 |         10 |   q8_0 |   q8_0 |   1 | tg128 @ d225000 |         30.20 ± 0.00 |

build: cd959bae9 (3391)
