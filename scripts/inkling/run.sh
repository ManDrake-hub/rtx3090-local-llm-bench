#!/usr/bin/env bash
# Inkling-2-Mini (ggml-org IQ4_NL) vs Qwen3.8-27B Q4+MTP. Same tests: max ctx, speed, HumanEval+refusals, BFCL.
# Uses separate fresh llama.cpp build (~/llama.cpp-qwen4); production untouched. NO auto-swap: report only.
set -u
D=~/llm-eval/inkling
A=~/llm-eval/agentic
R=~/llm-eval/results
OUT=$D/bench.tsv
NAME=inkling2-mini-iq4nl
M=~/models/inkling
LC=~/llama.cpp-qwen4
BIN=$LC/build/bin/llama-server
SPEC="--spec-type ngram-mod --spec-draft-n-max 3 -ub 128"
NTFY=${NTFY_URL:-}
export OPENAI_API_KEY=sk-local
export PATH=$HOME/.local/bin:$HOME/miniconda3/bin:$PATH
log() { echo "# $(date '+%F %T') $*" >> "$OUT"; }
ntfy() { curl -s -H "Title: $1" -H "Tags: bar_chart" -d "$2" "$NTFY" >/dev/null 2>&1; }
SRV=; PRX=; STAT=; STOPPED=0
restore() {
  kill ${PRX:-} ${SRV:-} ${STAT:-} 2>/dev/null; wait ${SRV:-} 2>/dev/null
  if [ $STOPPED = 1 ]; then
    sleep 5; systemctl --user restart llama-qwen
    for _ in $(seq 120); do curl -sf localhost:8080/health | grep -q ok && break; sleep 2; done
    log "llama-qwen restored ($(systemctl --user is-active llama-qwen))"
  fi
}
trap restore EXIT
fail() { log "FAIL: $1"; ntfy "Inkling test: problema" "$1
Server Qwen3.8 invariato."; exit 1; }
up() { # port ctx np extra...
  local port=$1 ctx=$2 np=$3; shift 3
  "$BIN" -m "$GG" --host 127.0.0.1 --port $port -ngl 99 -fa on -c $ctx -np $np -ctk q8_0 -ctv q8_0 \
    --jinja --no-webui "$@" > $D/server.log 2>&1 < /dev/null & SRV=$!
  for _ in $(seq 300); do kill -0 $SRV 2>/dev/null || { wait $SRV 2>/dev/null; SRV=; return 1; }
    curl -sf localhost:$port/health | grep -q ok && return 0; sleep 1; done
  down; return 1
}
down() { [ -n "$SRV" ] && { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }; SRV=; sleep 3; }

: > $OUT
ntfy "Test Inkling-2-Mini avviato" "Build llama.cpp nuovo + download IQ4_NL, poi ctx, velocita, HumanEval, BFCL (~3h). Server :8080 giu solo durante test GPU."

# 1) download + build (server still up)
log "download + build"
mkdir -p $M
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir $M > $D/dl.log 2>&1 || fail "download fallito"
GG=$M/Inkling-2-Mini-IQ4_NL.gguf
{ [ -d $LC ] || git clone -q https://github.com/ggml-org/llama.cpp $LC; } && git -C $LC pull -q \
  && cmake -S $LC -B $LC/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON > $D/build.log 2>&1 \
  && cmake --build $LC/build --target llama-server -j 20 >> $D/build.log 2>&1 || fail "build llama.cpp fallita"
log "gguf size=$(stat -c%s $GG) llama.cpp=$(git -C $LC log -1 --format=%h)"

systemctl --user stop llama-qwen; STOPPED=1; sleep 5
up 8093 16384 1 $SPEC || fail "llama.cpp non carica Inkling: $(grep -iE 'error|unknown|fail' $D/server.log | tail -1)"
down

( while sleep 900; do
    he=$(wc -l < $R/$NAME/humaneval.jsonl 2>/dev/null || echo 0)
    hp=$(grep -c '"pass": true' $R/$NAME/humaneval.jsonl 2>/dev/null || echo 0)
    bn=$(ls $A/jobs/bfcl/$NAME/*/result.json 2>/dev/null | wc -l)
    ntfy "Test Inkling - $(date +%H:%M)" "Fase: $(tail -1 $OUT | cut -c23-)
HumanEval Inkling: $hp/$he (Qwen3.8: 159/164)
BFCL Inkling: $bn/123 task (Qwen3.8: 82.9%)"
  done ) & STAT=$!

# 2) max ctx (sliding window -> may go very high)
BEST=0
for ctx in 163840 147456 131072 114688 98304 65536; do
  if up 8093 $ctx 1 $SPEC; then V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); down
    log "maxctx $ctx OK vram=${V}MiB"; BEST=$ctx; break; fi
  log "maxctx $ctx fail"
done
[ $BEST -gt 0 ] || fail "non parte nemmeno a 65k"

# 3) speed (at min(BEST,159744) for fair compare + at BEST)
C=$BEST; [ $C -gt 159744 ] && C=159744
up 8093 $C 1 $SPEC || fail "server non parte a $C"
python3 -I ~/llm-eval/mtptest.py 8093 inkling >> $OUT 2>&1
python3 -I ~/llm-eval/mtptest.py 8093 inkling-depth 32000 >> $OUT 2>&1
python3 -I ~/llm-eval/mtptest.py 8093 inkling-depth 100000 >> $OUT 2>&1
down
ntfy "Inkling velocita + contesto" "Contesto max: $BEST (Qwen3.8: 159744)
$(grep -P '^inkling' $OUT | cut -f2,3,5 | tr '\t' ' ')
(Qwen3.8: codegen 88.8, chat 69.1, edit 165.3, 31k 68, 96k 51)"

# 4) HumanEval + refusals
log "humaneval"
LLAMA_SERVER=$BIN EVAL_PARALLEL=2 EVAL_CTX=131072,98304,65536 EVAL_EXTRA_ARGS="$SPEC" \
  python3 -I ~/llm-eval/eval.py $NAME "$GG" > $R/$NAME.log 2>&1
HE=$(grep -c '"pass": true' $R/$NAME/humaneval.jsonl 2>/dev/null || echo 0)
RF=$(python3 -I -c "import json,sys;print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get('refused')))" $R/$NAME/refusals.jsonl 2>/dev/null || echo ?)
LOOP=$(python3 -I -c "import json,sys;print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get('finish')=='length'))" $R/$NAME/humaneval.jsonl 2>/dev/null || echo ?)
log "humaneval pass=$HE/164 refusals=$RF length-cut=$LOOP"
ntfy "Inkling HumanEval" "Inkling: $HE/164 (Qwen3.8: 159/164), troncati per lunghezza: $LOOP, rifiuti $RF/12"

# 5) BFCL
log "bfcl"
ok=0; for c in 131072 98304 65536; do up 8091 $c 2 $SPEC && { ok=1; SLOT=$((c/2)); break; }; done
[ $ok = 1 ] || fail "server BFCL non parte"
python3 -I $A/proxy.py 8092 8091 < /dev/null & PRX=$!; sleep 1
mkdir -p $A/jobs/bfcl
( while sleep 60; do find $A/jobs -type f -size +1G \( -name '*.pane' -o -name '*.cast' -o -name '*.txt' \) -exec truncate -s 0 {} \; ; done ) & WD=$!
OT=$((SLOT/4)); [ $OT -gt 16384 ] && OT=16384
mi="{\"max_input_tokens\":$SLOT,\"max_output_tokens\":$OT,\"input_cost_per_token\":0,\"output_cost_per_token\":0}"
rm -rf $A/jobs/bfcl/$NAME
harbor run -d bfcl_parity@1.0 -a terminus-2 -m openai/local --ak api_base=http://127.0.0.1:8092/v1 --ak temperature=0.6 \
  --ak record_terminal_session=false --ak "model_info=$mi" --agent-timeout-multiplier 2 -n 2 \
  -o $A/jobs/bfcl --job-name $NAME -y -q > $A/jobs/bfcl/$NAME.log 2>&1 < /dev/null
log "bfcl rc=$? $(grep -oE 'Mean: [0-9.]+' $A/jobs/bfcl/$NAME.log | tail -1)"
kill $WD $PRX 2>/dev/null; PRX=; down

cnt() { python3 -I - "$1" <<'EOF'
import glob, json, sys
p = 0
for t in glob.glob(f'{sys.argv[1]}/*/result.json'):
    try:
        r = (json.load(open(t)).get('verifier_result') or {}).get('rewards', {}).get('reward')
        p += r is not None and float(r) >= 1
    except Exception: pass
print(p)
EOF
}
BN=$(cnt $A/jobs/bfcl/$NAME); BO=$(cnt $A/jobs/bfcl/qwen38-q4-mtp)
log "bfcl pass new=$BN old=$BO"
ntfy "Inkling RISULTATO FINALE" "BFCL: Inkling $BN/123 vs Qwen3.8 $BO/123
HumanEval: Inkling $HE vs 159 | rifiuti $RF/12
Codegen t/s: Inkling $(grep -P '^inkling\tcodegen' $OUT | grep -oP 'tg=\K[0-9.]+') vs 88.8 | ctx max $BEST vs 159744
Server: resta Qwen3.8 (nessuno swap automatico). File: ~/models/inkling"
touch $D/DONE
log "DONE"
