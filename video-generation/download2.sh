#!/bin/bash
M=$HOME/ComfyUI/models
dl(){ hf download "$1" "$2" --local-dir $HOME/videogen/hfcache/$1 >/dev/null && mv "$HOME/videogen/hfcache/$1/$2" "$3/" && echo "$(date +%T) OK $2" || echo "FAIL $2"; }
dl Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors $M/diffusion_models
dl Comfy-Org/MiniMax-H3 loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors $M/loras
dl Comfy-Org/MiniMax-H3-Image vae/minimax_h3_image_vae_fp16.safetensors $M/vae
dl Comfy-Org/MiniMax-H3-Image diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors $M/diffusion_models
echo ALL DONE
