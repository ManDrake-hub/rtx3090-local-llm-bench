# RTX 3090 local LLM bench

Benchmarks and server tuning for running coding and agentic LLMs on **one RTX 3090 (24 GB)**.
The goal was practical: pick the best model and configuration for a home `llama-server` that
serves coding agents over the OpenAI and Anthropic Messages APIs. Everything here was measured on the
same machine between November 4 and November 23, 2026.

**TL;DR**
- **Qwen3.8-27B is the best all-rounder on 24 GB.** It led the tool-calling benchmark (BFCL) with 102/123
  at both Q5 and Q4. Several alternatives match it on HumanEval but lose 4 to 19 points on BFCL.
- **HumanEval no longer separates models.** Nearly everything scores 95-98%. BFCL is the useful signal for agentic use.
- **Speculative decoding matters more than the quant.** Qwen3.8-27B Q4 went from 42 t/s with no drafter
  to about 144 t/s on code with a DFlash2 drafter plus n-gram drafting (thinking off).
- **Do not trust a single "fill the context" test.** A context size that passed one long request crashed under real
  multi-turn agent traffic (CUDA OOM in `cudaGraphInstantiate`). Keep at least 300 MiB of VRAM free at runtime.
- **Inkling-2-Mini on vLLM (HyperInkling) is 2.5-3.5x faster than Qwen** at long context, but scores lower on BFCL (95 vs 102).

## Hardware and software

| | |
|---|---|
| GPU | RTX 3090 24 GB (PCIe link reported as x4) |
| CPU / RAM | i9-12900K, 32 GB DDR5-6000 |
| OS | Ubuntu 24.04 |
| Inference | llama.cpp (several builds, commits noted in the logs), vLLM via HyperInkling 0.1.2 for Inkling |
| Agent harness | [Harbor](https://github.com/laude-institute/harbor) with the `terminus-2` agent |

## Results

### 1. Model comparison (Q4/Q5 GGUF, llama.cpp)

Same settings for every model: thinking on, temperature 0.6, one sample per problem.
Speed comes from `llama-bench` with no speculative decoding, KV cache q8_0.

| Model | Quant | HumanEval (164) | BFCL (123) | Refusals (12) | tg128 t/s | pp4096 t/s |
|---|---|---|---|---|---|---|
| Qwen3.8-27B (official, Unsloth) | UD-Q5_K_XL | 160 | **102** | 4 | 36.9 | 1422 |
| Qwen3.8-27B (official, Unsloth) | UD-Q4_K_XL | 159 | **102** | 8¹ | see §2 | - |
| Qwen3.8-27B Heretic RVN (abliterated) | Q5_K_M | 160 | - | 3 | 38.8 | 1456 |
| Qwen3.8-27B Huihui (abliterated) | UD-Q5_K_XL | 160 | - | 6 | 37.4 | 1448 |
| Qwen3.8-27B HauhauCS Aggressive | Q5_K_P | 157 | - | 7 | 37.7 | 1435 |
| Qwen3.8-27B Coder390 EfficientThink | Q5_K_M | 160 | 95 | 0 | 38.8 | 1454 |
| Qwen3.8-27B Cold Fusion (merge) | Q4_K_M | **161** | 91 | 0 | 40.5 | 1482 |
| Ternary Bonsai 2 27B (stock build) | Q2_0 | 158 | 98 | 4 | 70.0 | 1554 |
| Ternary Bonsai 2 27B (PrismML fork) | PQ2_0 | 156 | - | 6 | 78.8 | 1491 |
| Ternary Bonsai 2 27B Heretic (PrismML fork) | PQ2_0 | 157 | - | 4 | 79.2 | 1489 |
| FlashNext (Strata) | Q2_0 | 160 | 93² | 1 | - | - |
| Flint 40B-A2.5B | UD-Q4_K_XL | 159 | 87 (12 errors) | 1 | 108.0 | 1282 |
| Nemotron 3.5 Lightning 30B-A3B | UD-IQ4_NL | 159 | 83 | 5 | **197.7** | **4240** |
| Nanbeige 4.3 33B | IQ4_XS | 156 | 85 (13 errors) | 7 | 42.7 | 1424 |
| Underdog Saluki 27B | IQ2 mix | 147 | - | 8 | 52.2 | 1389 |
| Qwen3.6-27B HauhauCS (previous baseline) | Q5_K_P | 159 | - | 0 | 36.3 | 1465 |

¹ 4 real refusals plus 4 empty answers cut by the length limit. ² Partial run, 2 errors.
"-" means not run. Per-model files: `results/models/`. Per-task agentic outcomes: `agentic/tasks.csv`.

Terminal-Bench 2 was started but stopped early: only Qwen3.8 Coder390 got a partial result (5/13).

### 2. Speed: Qwen3.8-27B Q4 with speculative decoding

Generation speed in tokens/s, 1200-token answers, thinking **off**, context 131072 (`results/tuning/qwen-speed.tsv`).

| Configuration | Code | Chat | File edit | Code at 31k context |
|---|---|---|---|---|
| No drafter | 41.8 | 41.6 | 40.9 | - |
| Older llama.cpp build, MTP n=3 + n-gram | 81.5 | 68.0 | 114.0 | 70.5 |
| Newer build, MTP n=3 | 91.9 | 74.9 | 131.4 | 77.1 |
| Newer build, MTP n=4 | 101.9 | 73.4 | 129.1 | 79.7 |
| **DFlash2 drafter (Q4_K_M) n=7 + n-gram** | **143.6** | **91.8** | 131.7 | **113.7** |

DFlash2 with n=15 failed to load, and MTP plus DFlash2 together failed to start.
With thinking **on** (the realistic setting for agents), the production config measured 74 / 62 / 79 t/s,
60 t/s at 31k and 44 t/s at 96k of context (`results/inkling/hyperinkling-bench.tsv`, `llama-qwen` rows).

### 3. Maximum context on 24 GB

- With a DFlash2 drafter loaded, the KV cache has to be split: **K q8_0 / V q4_0**.
  llama.cpp must be built with that flash-attention kernel pair
  (`-DGGML_CUDA_FA_QUANTS="...;q8_0-q4_0"`). Otherwise it silently falls back to an f16 conversion and runs out
  of memory near the context limit (`results/tuning/fa-tune.tsv`).
- 163840 tokens passed a single-request fill test, but crash-looped under real agent sessions with ~90k-token
  prompts: CUDA OOM in `cudaGraphInstantiate`, with only 154 MiB free at idle.
- The final setting is **147456 tokens**: 450 MiB free at idle, at least 334 MiB free under load. It was verified with
  `scripts/eval/stress.py`, a multi-turn tool-use session with vision up to 138k tokens, and with `stress-long.py`,
  which runs 3000-token generations at depth (`results/stress/`).
- Unsloth Dynamic v3.1 Q4_K_XL scored lower than v3 (BFCL 96 vs 102, HumanEval 155 vs 159) at the same speed (`results/tuning/v31.tsv`).

### 4. Inkling-2-Mini (26B looped MoE, ~3.4B active)

| | Qwen3.8-27B + DFlash2 | Inkling llama.cpp IQ4_NL | Inkling + DSpark n=3 | Inkling on HyperInkling (vLLM, W4A16) |
|---|---|---|---|---|
| HumanEval | 159 | 160 | - | 159 |
| BFCL | **102** (0 errors) | 97 (3 errors) | - | 95 (2 errors) |
| Refusals (12) | 8 | 9 | - | 7 |
| tg: code / chat / edit | 74 / 62 / 79 | 57.5 / 57.4 / 54.1 | 63.8 / 49.9 / 65.7 | **196 / 154 / 201** |
| tg at ~32-40k | 60 | 46 | 49 | **212** |
| tg at ~96-125k | 44 | 34 | 30 | **195** |
| Prefill at ~100k (t/s) | 882 | 200 | 195 | **2534** |
| Max context | 147456 | 163840 (needs `-ub 128`) | 147456 | **288384** (21.2 GB) |

The Qwen and HyperInkling columns use the same streaming benchmark with thinking on (`scripts/hyperinkling/sbench.py`).
The llama.cpp Inkling columns use `mtptest.py` with thinking off, so they are not strictly comparable.
On llama.cpp, Inkling needs `-ub 128`: with the default micro-batch the attention-bias compute buffer does not fit even at 65k.
HyperInkling ran in an isolated container: internal network only, all capabilities dropped, weights mounted read-only.

### 5. Video generation (same GPU, ComfyUI)

Wall times for one 5-second clip: LTX-2.5 distilled int8 at 1280x704 with audio takes about 60 s.
Wan 2.2 A14B Q5 with the Lightning LoRA at 832x480 takes about 130 s, with no audio.
MiniMax H3 int8 takes 70-200 s depending on steps and LoRA. Details: `video-generation/results/NOTES.md`.

## Method

- **HumanEval:** 164 problems, one sample each, thinking on, temperature 0.6. Code runs in
  `docker --network none` (python:3.12-slim, 512 MB, read-only) (`scripts/eval/eval.py`).
- **Refusal probe:** 12 dual-use security prompts (port scanning, SQL injection, reverse shell, keylogger, and so on).
  A refusal is detected by regex. Only the flags are published, not the answers.
- **BFCL:** `bfcl_parity@1.0` (123 tasks) through Harbor and `terminus-2`, run with `--agent-timeout-multiplier 2`.
  A small proxy (`scripts/agentic/proxy.py`) copies `reasoning_content` into `content` when the content is empty.
- **Speed:** `llama-bench` for screening. Then real streaming requests with code, chat and file-edit prompts at
  0, 32k and 100k tokens of context, each with a unique nonce so the prefix cache is never hit.
- **Context limits:** server start, then a fill test, then (from November 23) the multi-turn stress test.

## Limitations

- **One sample per task, one run per speed point.** HumanEval differences of 1-3 problems are noise, and so are a few BFCL points.
- **Not leaderboard-comparable.** BFCL ran through an agent harness with a doubled timeout, not the official BFCL runner.
- **Thinking on and off are mixed** across speed tables, as noted on each one.
- **The refusal count is a regex heuristic** on 12 prompts. Some flagged answers were partial help with caveats.
- **Lost raw data.** Some Inkling raw outputs (per-problem HumanEval answers, BFCL trajectories) were lost in a disk
  cleanup. Their tables were rebuilt from the run logs, and `results/inkling/` contains only those logs.
- **Model versions as of November 2026.** Model quality and llama.cpp performance change quickly.

## Repository layout

| Path | Content |
|---|---|
| `results/summary-humaneval.md`, `results/summary-agentic.md` | Summary tables |
| `results/models/<model>/` | `bench.json` (llama-bench), `humaneval.jsonl` (answers and pass/fail), `refusals.jsonl` (flags only) |
| `results/tuning/` | Speed, context and drafter sweeps |
| `results/stress/`, `results/inkling/` | Stress-test logs; Inkling, DSpark and HyperInkling runs |
| `agentic/tasks.csv` | Every BFCL and Terminal-Bench task outcome per model |
| `scripts/` | The scripts as they were run (they assume a `~/llm-eval` working directory) |
| `server/` | Production systemd units and a Qwen chat template patched for Anthropic Messages clients: mid-conversation system messages and `tool_reference` blocks |
| `video-generation/` | ComfyUI benchmark scripts and results |

Full agent trajectories (about 150 MB unpacked) are attached to the GitHub release as `agentic-trajectories.tar.zst`.

## Reproducing

1. Build llama.cpp with CUDA. For the split KV cache, add `-DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;q8_0-q4_0"`.
2. Put the HumanEval data (`HumanEval.jsonl.gz`, from [openai/human-eval](https://github.com/openai/human-eval)) in `~/llm-eval/data/`.
3. List models in `scripts/eval/models.tsv` and run `scripts/eval/run.sh`. It is resumable, and per-model results go to `~/llm-eval/results/`.
4. For BFCL, install Harbor (`uv tool install harbor`) and see `scripts/eval/run4.sh`.

Notifications in the scripts use an optional [ntfy](https://ntfy.sh) server, set with `NTFY_URL`.

## Credits and licenses

Scripts and results in this repository: MIT (see `LICENSE`).
Benchmarks used: HumanEval (OpenAI, MIT), Berkeley Function Calling Leaderboard (Apache-2.0),
Terminal-Bench (Apache-2.0). Inference: llama.cpp (MIT), vLLM (Apache-2.0).
Models belong to their respective authors and licenses.
