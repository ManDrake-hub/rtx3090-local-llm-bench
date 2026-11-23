#!/usr/bin/env bash
# New build with q8_0-q4_0 FA kernel. For base + heretic: max ctx (startup) then REAL fill test near max, speed, vision.
set -u
OUT=~/llm-eval/results/fa-tune.tsv
L=~/llm-eval/results/fa-tune-server.log
BIN=~/llama.cpp-prod/build-fa/bin/llama-server
DR=~/models/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
MM=~/models/Qwen3.8-27B-mmproj-F16.gguf
export PATH=$HOME/miniconda3/bin:$PATH
log() { echo "# $(date '+%F %T') $*" >> "$OUT"; }
SRV=
trap 'kill ${SRV:-} 2>/dev/null; wait ${SRV:-} 2>/dev/null; sleep 5; systemctl --user restart llama-qwen; log "llama-qwen restarted ($(systemctl --user is-active llama-qwen))"' EXIT
up() { local ctx=$1
  $BIN -m $GG --host 127.0.0.1 --port 8093 -ngl 99 -fa on -c $ctx -np 1 -ctk q8_0 -ctv q4_0 \
    -md $DR -ngld 99 --spec-type draft-dflash2,ngram-mod --spec-draft-n-max 7 \
    --mmproj $MM --no-mmproj-offload --jinja --no-webui > $L 2>&1 < /dev/null & SRV=$!
  for _ in $(seq 300); do kill -0 $SRV 2>/dev/null || { wait $SRV 2>/dev/null; SRV=; return 1; }
    curl -sf localhost:8093/health | grep -q ok && return 0; sleep 1; done
  down; return 1; }
down() { [ -n "$SRV" ] && { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }; SRV=; sleep 3; }
alive() { kill -0 $SRV 2>/dev/null; }
: > $OUT
systemctl --user stop llama-qwen; sleep 5
for spec in "base:$HOME/models/Qwen3.8-27B-UD-Q4_K_XL.gguf" "heretic:$HOME/models/Qwen3.8-27B-Heretic-RVN-Q4_K_M-multilingual-mtp.gguf"; do
  name=${spec%%:*}; GG=${spec#*:}
  for ctx in 196608 180224 172032 163840 159744 147456 131072; do
    up $ctx || { log "$name ctx=$ctx no-start"; continue; }
    log "$name ctx=$ctx started vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
    python3 -I ~/llm-eval/mtptest.py 8093 $name-$ctx-fill $((ctx - 6000)) >> $OUT 2>&1
    if alive; then
      log "$name ctx=$ctx FILL OK peak=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
      python3 -I ~/llm-eval/mtptest.py 8093 $name-$ctx >> $OUT 2>&1
      python3 -I ~/llm-eval/mtptest.py 8093 $name-$ctx-d32k 32000 >> $OUT 2>&1
      python3 -I ~/llm-eval/mtptest.py 8093 $name-$ctx-d100k 100000 >> $OUT 2>&1
      alive && log "$name BEST ctx=$ctx" || log "$name ctx=$ctx crashed in speed tests"
      down; alive || true
      grep -q "$name BEST" $OUT && break
    else
      log "$name ctx=$ctx CRASH on fill: $(grep -E ' E ' $L | tail -1)"; wait $SRV 2>/dev/null; SRV=; sleep 3
    fi
  done
done
log DONE
