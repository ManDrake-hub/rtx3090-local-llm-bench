# Video-gen benchmark notes (RTX 3090, 2026-11-13)

Recovered from terminal logs. Every number below appears in the logs; anything inferred is marked *(inferred)*.

- Times in `bench.tsv` are local time (CET, UTC+1), HH:MM. Log timestamps in brackets are UTC on 2026-11-13.
- Each wall time covers the whole ComfyUI prompt: model load plus sampling plus decode, measured by `bench.py` from submit to completion.
- `peakVRAM` is the highest `nvidia-smi` memory.used sampled once a second. `minAvail` is the lowest MemAvailable, and `swap` is the most swap in use during the run.
- Clip properties (resolution, frames, fps, length, audio level) come from `sheet.py`.

## Machine

- RTX 3090: ComfyUI reported `Total VRAM 24124 MB, total RAM 31760 MB`, with DynamicVRAM enabled [15:31:09Z].
- PCIe link: x16 (gen1 shows only at idle).
- Swap is about 8 GB (`free -g`: Swap total 7 GiB).
- ComfyUI v0.45.0-16-gdc9ffe6a, torch 2.14.1+cu130, uv-managed CPython 3.12.13.
- Gotcha 1: the system Python 3.12 has no `Python.h`, so Triton fails to compile with a gcc error in `cuda_utils`.
- Gotcha 2: torch cu128 gives the warning "You need pytorch with cu130".
- The GPU is shared with the `llama-qwen` service, which must be stopped first (see `videomode.sh`). ComfyUI started while it was running gave a CUDA OOM at startup.

## Models used

| Model | Files |
|---|---|
| LTX-2.5 | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot`, Gemma4-12B text encoder int8, video/audio VAE bf16, x2 latent upscaler |
| Wan 2.2 A14B | QuantStack GGUF Q5_K_M high/low noise (T2V and I2V), umt5_xxl fp8 text encoder, Wan2.1 VAE, lightx2v Lightning 4-step LoRAs (T2V Seko-V2.0, I2V Seko-V1) |
| MiniMax H3 | `minimax_h3_fl2va_pruned_int8_convrot` (t2v/i2v) and `minimax_h3_ref2va_pruned_int8_convrot` (ref2v), qwen3vl_32b int8 text encoder, video VAE fp16, audio VAE fp32. LoRAs: `minimax_h3_fl2v_turbo_4step_v1.3_768p` (t2v/i2v) and `minimax_h3_ref2v_turbo_4step_v0.1` (ref2v) |
| MiniMax H3-Image | `minimax_h3_image_pruned_int8_convrot` and image VAE fp16, used for keyframes |

Total model size on disk was 183 GB (`du -sh ~/ComfyUI/models`, 16:14:37Z).

## 1. Base benchmark: 5 s clips, seed 42

Prompts:
- **t2v:** a golden retriever on a beach at sunset.
- **i2v:** `Felis_catus-cat_on_snow.jpg`, a Wikimedia Commons 1280x853 thumbnail.

| Test | Resolution | Frames / length | Steps | Time | Peak VRAM | Min RAM avail | Swap | Audio (rms / peak dBFS) | Verdict | Log |
|---|---|---|---|---|---|---|---|---|---|---|
| ltx_t2v (0.9 MP) | 1280x704 | 121 f, 24 fps, 5.04 s | distilled, 8+3 sigmas, x2 latent upscale *(from template)* | **62 s** | 23993 MiB | 22462 MiB | 0 | -22.6 / -0.4 | Very good: the camera tracks the dog and audio is present | 15:35:37Z |
| ltx_i2v (0.9 MP, 3:2) | 1152x768 | 121 f, 24 fps, 5.04 s | same as above | **56 s** | 23761 MiB | 21857 MiB | 0 | -38.3 / -17.5 | No explicit verdict logged | 15:41:11Z |
| wan_t2v | 832x480 | 81 f, 16 fps, 5.06 s | Lightning 4 steps (2 high + 2 low) | **133 s** | 17201 MiB | 17840 MiB | 8191 MiB | no audio | OK, silent | 15:41:11Z |
| wan_i2v | 768x512 | 81 f, 16 fps, 5.06 s | Lightning 4 steps | **130 s** | 15187 MiB | 17165 MiB | 7580 MiB | no audio | OK, silent | 15:41:11Z |
| h3_t2v (0.4 MP, turbo) | 864x480 | 124 f, 24 fps, 5.17 s | 4 steps, fl2v turbo LoRA v1.3 | **70 s** | 23849 MiB | 17120 MiB | 4446 MiB | -38.4 / -14.0 | **Bad: heavy texture artifacts** | 15:42:58Z |
| h3_i2v (0.4 MP, turbo, 3:2) | 800x544 | 124 f, 24 fps, 5.17 s | 4 steps, fl2v turbo LoRA v1.3 | **72 s** | 23851 MiB | 18237 MiB | 6357 MiB | -43.6 / -25.2 | Checked for the same artifacts; result not stated explicitly *(uncertain)* | 15:44:14Z |
| h3_t2v_20st (0.4 MP, no LoRA) | 864x480 | 124 f, 24 fps, 5.17 s | 20 steps | **202 s** | 23955 MiB | 19552 MiB | 4203 MiB | -39.7 / -17.8 | Clean ("at 20 steps it is clean, ~200 s") | 15:51:18Z |
| h3_t2v_768 (1.0 MP, turbo) | 1376x768 | 124 f, 24 fps, 5.17 s | 4 steps, turbo LoRA | **208 s** | 24083 MiB | 18726 MiB | 4157 MiB | -30.8 / -10.5 | No explicit verdict logged *(uncertain)* | 15:51:18Z |

Failed attempts:
- **16:33, ltx_t2v error (2 s).** Triton could not compile because the venv used the system Python without headers. Fixed by recreating the venv with uv-managed Python.
- **Earlier ltx_t2v run, rejected at validation.** The disabled prompt-enhancer branch still required `gemma4_e2b_it_int8_convrot.safetensors`. Fixed with `drop_enhancer()` in `bench.py`. This run never reached `bench.tsv`.

## Summary and verdicts

- **LTX-2.5 distilled int8:** fastest and best overall. A 5-6 s clip at 1280x704 with audio takes about 56-74 s.
  - It pins the GPU at about 24 GB.
  - Base runs used no swap.
- **Wan 2.2 A14B Q5 GGUF + Lightning:** about 124-134 s per 5 s clip at 832x480.
  - Peak VRAM 15-21 GB, but it is the heaviest on system RAM: swap up to 8 GB.
  - Produces no audio.
- **MiniMax H3 int8:**
  - The fl2v turbo LoRA v1.3 at 4 steps on t2v gives heavy artifacts.
  - Without the LoRA at 20 steps the output is clean, but takes about 200 s.
  - ref2v with the ref2v turbo LoRA v0.1 is clean at about 85 s per 5.9 s clip at 864x480, with audio.
- **H3-Image keyframes:** about 20-35 s each at 2048x1152.
- **RAM:** 32 GB forces 4-8 GB of swap. Upgrading to 64 GB was the recommended fix.

