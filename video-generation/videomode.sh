#!/bin/bash
# videomode.sh on  -> stop llama-qwen, start ComfyUI (http://127.0.0.1:8188)
# videomode.sh off -> stop ComfyUI, start llama-qwen
case "$1" in
  on)  systemctl --user stop llama-qwen
       cd ~/ComfyUI && nohup .venv/bin/python main.py --listen 127.0.0.1 --port 8188 --output-directory ~/videogen/out > ~/videogen/comfy.log 2>&1 &
       echo "ComfyUI starting on :8188" ;;
  off) pkill -f "[m]ain.py --listen 127.0.0.1 --port 8188"; sleep 3
       systemctl --user start llama-qwen; systemctl --user is-active llama-qwen ;;
  *) echo "usage: $0 on|off" ;;
esac
