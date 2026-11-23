#!/usr/bin/env bash
# Qwen3.8-27B Q4 speed sweep: prod llama.cpp vs latest, MTP draft length, DFlash2 drafter (z-lab), combos.
# Speculative decoding is lossless -> speed only. Report only, production unchanged.
set -u
OUT=~/llm-eval/results/qwen-speed.tsv
LOGS=~/llm-eval/results/qwen-speed-server.log
OLD=~/llama.cpp-new/build/bin/llama-server
LC=~/llama.cpp-qwen4
NEW=$LC/build/bin/llama-server
GG=~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
DR=~/models/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
NTFY=${NTFY_URL:-}
export PATH=$HOME/miniconda3/bin:$PATH
log() { echo "# $(date '+%F %T') $*" >> "$OUT"; }
ntfy() { curl -s -H "Title: $1" -H "Tags: bar_chart" -d "$2" "$NTFY" >/dev/null 2>&1; }
SRV=; STOPPED=0
restore() {
  kill ${SRV:-} 2>/dev/null; wait ${SRV:-} 2>/dev/null
  if [ $STOPPED = 1 ]; then sleep 5; systemctl --user restart llama-qwen
    for _ in $(seq 120); do curl -sf localhost:8080/health | grep -q ok && break; sleep 2; done
    log "llama-qwen restored ($(systemctl --user is-active llama-qwen))"; fi
}
trap restore EXIT
up() { local bin=$1 ctx=$2; shift 2
  "$bin" -m "$GG" --host 127.0.0.1 --port 8093 -ngl 99 -fa on -c $ctx -np 1 -ctk q8_0 -ctv q8_0 \
    --jinja --no-webui "$@" > $LOGS 2>&1 < /dev/null & SRV=$!
  for _ in $(seq 300); do kill -0 $SRV 2>/dev/null || { wait $SRV 2>/dev/null; SRV=; return 1; }
    curl -sf localhost:8093/health | grep -q ok && return 0; sleep 1; done
  down; return 1; }
down() { [ -n "$SRV" ] && { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }; SRV=; sleep 3; }

: > $OUT
mkdir -p ~/models/dflash
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf --local-dir ~/models/dflash > /dev/null 2>&1 || log "drafter download FAIL"
git -C $LC pull -q && cmake --build $LC/build --target llama-server -j 20 > ~/llm-eval/results/qwen-speed-build.log 2>&1 || log "build FAIL"
log "old=$(git -C ~/llama.cpp-new log -1 --format=%h) new=$(git -C $LC log -1 --format=%h)"

systemctl --user stop llama-qwen; STOPPED=1; sleep 5
MTPN="--spec-type draft-mtp,ngram-mod --spec-draft-n-max"
DF="-md $DR -ngld 99"
while IFS='|' read -r name bin args; do
  [ -z "$name" ] && continue
  if up ${!bin} 131072 $args; then
    log "$name up vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)MiB"
    python3 -I ~/llm-eval/mtptest.py 8093 $name >> $OUT 2>&1
    python3 -I ~/llm-eval/mtptest.py 8093 $name-d32k 32000 >> $OUT 2>&1
    down
  else log "$name FAIL: $(grep -iE ' E |error|invalid|cannot' $LOGS | tail -2 | tr '\n' ' ')"; fi
done <<EOF
prod-old|OLD|$MTPN 3
mtp3-new|NEW|$MTPN 3
mtp2-new|NEW|$MTPN 2
mtp4-new|NEW|$MTPN 4
mtp5-new|NEW|$MTPN 5
dflash7|NEW|$DF --spec-type draft-dflash2 --spec-draft-n-max 7
dflash7-ngram|NEW|$DF --spec-type draft-dflash2,ngram-mod --spec-draft-n-max 7
dflash15|NEW|$DF --spec-type draft-dflash2 --spec-draft-n-max 15
mtp-dflash|NEW|$DF --spec-type draft-mtp,draft-dflash2,ngram-mod --spec-draft-n-max 7
EOF

# max ctx with drafter loaded (costs VRAM)
for ctx in 159744 147456 139264 131072; do
  if up $NEW $ctx $DF --spec-type draft-dflash2 --spec-draft-n-max 7; then down; log "maxctx dflash $ctx OK"; break; fi
  log "maxctx dflash $ctx fail"
done
ntfy "Qwen velocita: sweep finito" "$(grep -vP '^#' $OUT | awk -F'\t' '{print $1, $2, $5}')
$(grep -E 'maxctx.*OK|FAIL' $OUT | cut -c23-)"
log DONE
