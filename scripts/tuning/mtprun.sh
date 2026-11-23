#!/usr/bin/env bash
# Qwen3.8-27B UD-Q4_K_XL: speed with/without MTP, then max context with MTP.
set -u
M=~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
S=~/llama.cpp-new/build/bin/llama-server
P=8093
OUT=~/llm-eval/results/mtp-q38q4.tsv
LOG=~/llm-eval/results/mtp-server.log
: > "$OUT"
systemctl --user stop llama-bonsai

start() { # ctx kv extra...
  local ctx=$1 kv=$2; shift 2
  "$S" -m "$M" --host 127.0.0.1 --port $P -ngl 99 -fa on -c "$ctx" -np 1 -ctk "$kv" -ctv "$kv" \
    --jinja --no-webui "$@" > "$LOG" 2>&1 < /dev/null &
  SRV=$!
  for _ in $(seq 300); do
    kill -0 $SRV 2>/dev/null || return 1
    curl -sf localhost:$P/health | grep -q ok && return 0
    sleep 1
  done
  return 1
}
stop() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }

# 1) speed at 65k ctx q8_0
for cfg in "none" "draft-mtp 2" "draft-mtp 3" "draft-mtp,ngram-mod 3"; do
  set -- $cfg
  extra=(); [ "$1" != none ] && extra=(--spec-type "$1" --spec-draft-n-max "$2")
  if start 65536 q8_0 "${extra[@]}"; then
    echo -e "# $cfg vram=$(vram)MiB" >> "$OUT"
    python3 -I ~/llm-eval/mtptest.py $P "$cfg" >> "$OUT" 2>&1
  else
    echo "# $cfg FAILED: $(grep -iE 'error|fail' "$LOG" | tail -2)" >> "$OUT"
  fi
  stop
done

# 2) max context with MTP (n-max 2)
for kv in q8_0 q4_0; do
  for ctx in 262144 229376 196608 163840 150000 131072; do
    if start $ctx $kv --spec-type draft-mtp --spec-draft-n-max 2; then
      echo -e "# maxctx kv=$kv ctx=$ctx OK vram=$(vram)MiB" >> "$OUT"
      stop; break
    fi
    echo "# maxctx kv=$kv ctx=$ctx fail" >> "$OUT"; stop
  done
done

# 3) speed at depth ~120k with MTP, q8_0 at 150000 (or best q8 ctx)
c8=$(grep -oP 'kv=q8_0 ctx=\K[0-9]+(?= OK)' "$OUT" | head -1)
if [ -n "$c8" ] && start "$c8" q8_0 --spec-type draft-mtp --spec-draft-n-max 2; then
  for d in 32000 100000; do
    [ $d -lt $((c8 - 4000)) ] && python3 -I ~/llm-eval/mtptest.py $P "depth-mtp-q8" $d >> "$OUT" 2>&1
  done
  stop
fi
echo "# DONE" >> "$OUT"
