#!/usr/bin/env bash
# Probe ctx for prod config (new build + DFlash2 + mmproj CPU), K q8_0 / V q4_0 vs q8/q8; speed + vision check.
set -u
OUT=~/llm-eval/results/heretic-tune.tsv
L=~/llm-eval/results/heretic-tune-server.log
BIN=~/llama.cpp-prod/build/bin/llama-server
GG=~/models/Qwen3.8-27B-Heretic-RVN-Q4_K_M-multilingual-mtp.gguf
DR=~/models/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
MM=~/models/Qwen3.8-27B-mmproj-F16.gguf
export PATH=$HOME/miniconda3/bin:$PATH
log() { echo "# $(date '+%F %T') $*" >> "$OUT"; }
SRV=
trap 'kill ${SRV:-} 2>/dev/null; wait ${SRV:-} 2>/dev/null; sleep 5; systemctl --user restart llama-qwen; log "llama-qwen restarted ($(systemctl --user is-active llama-qwen))"' EXIT
up() { local ctx=$1 kv=$2
  $BIN -m $GG --host 127.0.0.1 --port 8093 -ngl 99 -fa on -c $ctx -np 1 -ctk q8_0 -ctv $kv \
    -md $DR -ngld 99 --spec-type draft-dflash2,ngram-mod --spec-draft-n-max 7 \
    --mmproj $MM --no-mmproj-offload --jinja --no-webui > $L 2>&1 < /dev/null & SRV=$!
  for _ in $(seq 300); do kill -0 $SRV 2>/dev/null || { wait $SRV 2>/dev/null; SRV=; return 1; }
    curl -sf localhost:8093/health | grep -q ok && return 0; sleep 1; done
  down; return 1; }
down() { [ -n "$SRV" ] && { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }; SRV=; sleep 3; }
: > $OUT
systemctl --user stop llama-qwen; sleep 5
~/llama.cpp-new/build/bin/llama-server -m $GG --host 127.0.0.1 --port 8093 -ngl 99 -fa on -c 159744 -np 1 -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --jinja --no-webui > $L 2>&1 < /dev/null & SRV=$!
for _ in $(seq 300); do curl -sf localhost:8093/health | grep -q ok && break; sleep 1; done
python3 -I ~/llm-eval/mtptest.py 8093 heretic-old >> $OUT 2>&1
python3 -I ~/llm-eval/mtptest.py 8093 heretic-old-d32k 32000 >> $OUT 2>&1
python3 -I ~/llm-eval/mtptest.py 8093 heretic-old-d100k 100000 >> $OUT 2>&1
down
for kv in q4_0 q8_0; do
  for ctx in 180224 172032 167936 163840 159744 147456 131072; do # 172032 163840 159744 147456 131072; do
    if up $ctx $kv; then log "maxctx kv=$kv ctx=$ctx OK vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; down; echo "$kv $ctx" >> $OUT.best; break; fi
    log "maxctx kv=$kv ctx=$ctx fail"
  done
done
read KV CTX < <(head -1 $OUT.best); rm -f $OUT.best
up $CTX $KV || { log "best fail"; exit 1; }
python3 -I ~/llm-eval/mtptest.py 8093 heretic-dflash-$CTX >> $OUT 2>&1
python3 -I ~/llm-eval/mtptest.py 8093 heretic-dflash-$CTX-d32k 32000 >> $OUT 2>&1
python3 -I ~/llm-eval/mtptest.py 8093 heretic-dflash-$CTX-d100k 100000 >> $OUT 2>&1
# vision check: tiny red PNG
python3 -I - >> $OUT 2>&1 <<'EOF'
import base64, json, struct, zlib, urllib.request
def png(w, h, rgb):
    raw = b''.join(b'\x00' + bytes(rgb) * w for _ in range(h))
    c = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d))
    return b'\x89PNG\r\n\x1a\n' + c(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + c(b'IDAT', zlib.compress(raw)) + c(b'IEND', b'')
img = base64.b64encode(png(64, 64, (255, 0, 0))).decode()
body = {'max_tokens': 50, 'chat_template_kwargs': {'enable_thinking': False}, 'messages': [{'role': 'user', 'content': [
    {'type': 'text', 'text': 'What color is this image? One word.'},
    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + img}}]}]}
r = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8093/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=300))
print('# vision:', r['choices'][0]['message']['content'].strip()[:80])
EOF
log "used kv=$KV ctx=$CTX"
down
log DONE
