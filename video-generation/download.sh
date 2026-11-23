#!/bin/bash
# Download video-gen models into ComfyUI model dirs. Resumable (hf download).
set -u
M=$HOME/ComfyUI/models
mkdir -p $M/{diffusion_models,text_encoders,vae,loras,latent_upscale_models,unet}
export HF_HUB_ENABLE_HF_TRANSFER=0
dl(){ # repo file destdir
  local repo=$1 f=$2 dst=$3
  if [ -e "$dst/$(basename $f)" ]; then echo "skip $f"; return; fi
  echo "$(date +%T) START $repo/$f"
  hf download "$repo" "$f" --local-dir $HOME/videogen/hfcache/$repo >/dev/null && \
    mv "$HOME/videogen/hfcache/$repo/$f" "$dst/" && echo "$(date +%T) OK $f" || echo "$(date +%T) FAIL $f"
}
# --- LTX-2.5 (first: smallest complete set)
R=comfyicu/LTX-2.5
dl $R diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors $M/diffusion_models
dl $R text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors $M/text_encoders
dl $R vae/ltx-2.5-video-vae-bf16.safetensors $M/vae
dl $R vae/ltx-2.5-audio-vae-bf16.safetensors $M/vae
dl $R latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors $M/latent_upscale_models
# --- Wan 2.2 A14B
dl Comfy-Org/Wan-2.2 split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors $M/text_encoders
dl QuantStack/Wan2.2-I2V-A14B-GGUF VAE/Wan2.1_VAE.safetensors $M/vae
for n in HighNoise LowNoise; do
  dl QuantStack/Wan2.2-I2V-A14B-GGUF $n/Wan2.2-I2V-A14B-$n-Q5_K_M.gguf $M/unet
  dl QuantStack/Wan2.2-T2V-A14B-GGUF $n/Wan2.2-T2V-A14B-$n-Q5_K_M.gguf $M/unet
done
# Lightning LoRAs (same filenames -> rename)
for v in "I2V-A14B-4steps-lora-rank64-Seko-V1:i2v" "T2V-A14B-4steps-lora-rank64-Seko-V2.0:t2v"; do
  d=${v%%:*}; t=${v##*:}
  for h in high low; do
    out=$M/loras/wan22_lightning_${t}_${h}.safetensors
    [ -e $out ] && continue
    hf download lightx2v/Wan2.2-Lightning Wan2.2-$d/${h}_noise_model.safetensors --local-dir $HOME/videogen/hfcache/lx >/dev/null && \
      mv $HOME/videogen/hfcache/lx/Wan2.2-$d/${h}_noise_model.safetensors $out && echo "OK $out"
  done
done
# --- MiniMax H3
R=Comfy-Org/MiniMax-H3
dl $R diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors $M/diffusion_models
dl $R text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors $M/text_encoders
dl $R vae/minimax_h3_video_vae_fp16.safetensors $M/vae
dl $R vae/minimax_h3_audio_vae_fp32.safetensors $M/vae
dl lightx2v/Minimax-h3-Turbo minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16.safetensors $M/loras
echo "$(date +%T) ALL DONE"
