#!/usr/bin/env bash
# After v31test: if llama-qwen switched to v3.1, find max ctx (q8_0 KV, MTP) and set it in the unit.
set -u
OUT=~/llm-eval/results/v31.tsv
UNIT=~/.config/systemd/user/llama-qwen.service
NEW=~/models/Qwen3.8-27B-UD-Q4_K_XL-v31.gguf
LS=~/llama.cpp-new/build/bin/llama-server
NTFY=${NTFY_URL:-}
log() { echo "# $(date '+%F %T') $*" >> "$OUT"; }
ntfy() { curl -s -H "Title: $1" -H "Tags: bar_chart" -d "$2" "$NTFY" >/dev/null 2>&1; }
while systemctl --user is-active -q v31test; do sleep 60; done
grep -q 'UD-Q4_K_XL-v31.gguf' "$UNIT" || { log "ctx probe skipped (v3.1 not adopted)"; exit 0; }

trap 'kill ${SRV:-} 2>/dev/null; wait ${SRV:-} 2>/dev/null; sleep 5; systemctl --user daemon-reload; systemctl --user restart llama-qwen; log "llama-qwen restarted after ctx probe"' EXIT
systemctl --user stop llama-qwen; sleep 5
best=0
for ctx in 180224 172032 165888 159744 155648 152576 150000 147456 143360 139264 131072; do
  $LS -m "$NEW" --host 127.0.0.1 --port 8093 -ngl 99 -fa on -c $ctx -np 1 -ctk q8_0 -ctv q8_0 \
    --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --jinja --no-webui > ~/llm-eval/results/v31-ctx.log 2>&1 < /dev/null & SRV=$!
  ok=0; for _ in $(seq 300); do kill -0 $SRV 2>/dev/null || break; curl -sf localhost:8093/health | grep -q ok && { ok=1; break; }; sleep 1; done
  v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=; sleep 3
  log "maxctx v31 ctx=$ctx $([ $ok = 1 ] && echo OK || echo fail) vram=${v}MiB"
  [ $ok = 1 ] && { best=$ctx; break; }
done
if [ $best -gt 0 ]; then
  sed -i -E "s/-c [0-9]+ -np 1/-c $best -np 1/" "$UNIT"
  log "llama-qwen ctx set to $best"
  ntfy "v3.1 max context" "Max ctx v3.1 (KV q8_0 + MTP): $best tokens (old 150000). llama-qwen set to $best.
dsh client: update contextWindow to $best."
else
  ntfy "v3.1 context" "Context probe failed, unit unchanged (150000)."
fi
