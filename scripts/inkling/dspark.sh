#!/usr/bin/env bash
# Inkling-2-Mini + DSpark drafter (llama.cpp PR #32218) speed test. Report only.
set -u
D=~/llm-eval/inkling
OUT=$D/dspark.tsv
LC=~/llama.cpp-dspark
BIN=$LC/build/bin/llama-server
GG=~/models/inkling/Inkling-2-Mini-IQ4_NL.gguf
DR=~/models/inkling/dspark-Inkling-2-Mini-Q8_0.gguf
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
fail() { log "FAIL: $1"; ntfy "Inkling DSpark: problem" "$1"; exit 1; }
up() { local ctx=$1; shift
  "$BIN" -m "$GG" --host 127.0.0.1 --port 8093 -ngl 99 -fa on -c $ctx -np 1 -ctk q8_0 -ctv q8_0 -ub 128 \
    --jinja --no-webui "$@" > $D/dspark-server.log 2>&1 < /dev/null & SRV=$!
  for _ in $(seq 300); do kill -0 $SRV 2>/dev/null || { wait $SRV 2>/dev/null; SRV=; return 1; }
    curl -sf localhost:8093/health | grep -q ok && return 0; sleep 1; done
  down; return 1; }
down() { [ -n "$SRV" ] && { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }; SRV=; sleep 3; }

: > $OUT
log "build PR #32218 $(git -C $LC log -1 --format=%h) + download drafter"
hf download unsloth/Inkling-2-Mini-GGUF dspark-Inkling-2-Mini-Q8_0.gguf --local-dir ~/models/inkling > $D/dl-dspark.log 2>&1 || fail "drafter download failed"
cmake -S $LC -B $LC/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON > $D/build-dspark.log 2>&1 \
  && cmake --build $LC/build --target llama-server -j 20 >> $D/build-dspark.log 2>&1 || fail "build failed"

systemctl --user stop llama-qwen; STOPPED=1; sleep 5
# baseline with same build (no drafter, ngram) then DSpark n=3,5,7
for cfg in "ngram:--spec-type ngram-mod --spec-draft-n-max 3" \
           "dspark3:-md $DR -ngld 99 --spec-type draft-dspark --spec-draft-n-max 3" \
           "dspark5:-md $DR -ngld 99 --spec-type draft-dspark --spec-draft-n-max 5" \
           "dspark7:-md $DR -ngld 99 --spec-type draft-dspark --spec-draft-n-max 7"; do
  name=${cfg%%:*}; args=${cfg#*:}
  if up 131072 $args; then
    log "$name up vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)MiB"
    python3 -I ~/llm-eval/mtptest.py 8093 $name >> $OUT 2>&1
    python3 -I ~/llm-eval/mtptest.py 8093 $name-d32k 32000 >> $OUT 2>&1
    python3 -I ~/llm-eval/mtptest.py 8093 $name-d100k 100000 >> $OUT 2>&1
    down
  else log "$name FAIL: $(grep -iE ' E |error' $D/dspark-server.log | tail -2 | tr '\n' ' ')"; fi
done
# max ctx with best dspark (n=3 default)
for ctx in 196608 163840 147456 131072; do
  if up $ctx -md $DR -ngld 99 --spec-type draft-dspark --spec-draft-n-max 3; then down; log "maxctx dspark $ctx OK"; break; fi
  log "maxctx dspark $ctx fail"
done
ntfy "Inkling + DSpark speed" "$(grep -vP '^#' $OUT | awk -F'\t' '{print $1, $2, $5, $7}' | sed 's/draft_acc=//')
(Qwen3.8 MTP: codegen 88.8, chat 69.1, edit 165.3, 31k 68, 96k 51)
$(grep maxctx $OUT | grep OK)"
log DONE
