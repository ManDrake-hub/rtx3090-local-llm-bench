#!/usr/bin/env bash
# Qwen4-27B vs current Qwen3.8-27B Q4+MTP. Args: <hf_repo> <quant_pattern>
# Download, llama.cpp support check (rebuild separate llama.cpp if needed), MTP check, max ctx, speed,
# HumanEval+refusals, BFCL. Swap llama-qwen only if BFCL >= old+2 and HumanEval >= old-1. Restore server always.
# On transient failure (no llama.cpp support yet) exits without DONE -> watcher retries next check.
set -u
REPO=$1 PAT=$2
D=~/llm-eval/qwen4
A=~/llm-eval/agentic
R=~/llm-eval/results
OUT=$D/bench.tsv
NAME=qwen4-27b-$(echo $PAT | tr 'A-Z_' 'a-z-')
M=~/models/qwen4
UNIT=~/.config/systemd/user/llama-qwen.service
LS=~/llama.cpp-new/build/bin/llama-server
LC4=~/llama.cpp-qwen4
NTFY=${NTFY_URL:-}
export OPENAI_API_KEY=sk-local
export PATH=$HOME/.local/bin:$HOME/miniconda3/bin:/usr/local/cuda/bin:$PATH
log() { echo "# $(date '+%F %T') $*" >> "$OUT"; }
ntfy() { curl -s -H "Title: $1" -H "Tags: bar_chart" -d "$2" "$NTFY" >/dev/null 2>&1; }
SRV=; PRX=; STAT=; STOPPED=0
restore() {
  kill ${PRX:-} ${SRV:-} ${STAT:-} 2>/dev/null; wait ${SRV:-} 2>/dev/null
  if [ $STOPPED = 1 ]; then
    sleep 5; systemctl --user daemon-reload; systemctl --user restart llama-qwen
    for _ in $(seq 120); do curl -sf localhost:8080/health | grep -q ok && break; sleep 2; done
    log "llama-qwen restored ($(systemctl --user is-active llama-qwen))"
  fi
}
trap restore EXIT
fail() { log "FAIL: $1"; ntfy "Qwen4 test: problem" "$1
Qwen3.8 server unchanged. Will retry at the next check."; exit 1; }
up() { # bin port ctx np extra... -> 0 if healthy (SRV set)
  local bin=$1 port=$2 ctx=$3 np=$4; shift 4
  "$bin" -m "$GG" --host 127.0.0.1 --port $port -ngl 99 -fa on -c $ctx -np $np -ctk q8_0 -ctv q8_0 \
    --jinja --no-webui "$@" > $D/server.log 2>&1 < /dev/null & SRV=$!
  for _ in $(seq 300); do kill -0 $SRV 2>/dev/null || { wait $SRV 2>/dev/null; SRV=; return 1; }
    curl -sf localhost:$port/health | grep -q ok && return 0; sleep 1; done
  down; return 1
}
down() { [ -n "$SRV" ] && { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }; SRV=; sleep 3; }

log "start $REPO $PAT"

# 1) download (resumable)
mkdir -p $M
hf download "$REPO" --include "*${PAT}*" --local-dir $M >> $D/dl.log 2>&1 || fail "download failed ($REPO)"
hf download "$REPO" --include "*mmproj*F16*" --local-dir $M >> $D/dl.log 2>&1
GG=$(find $M -name "*${PAT}*.gguf" ! -iname '*mmproj*' | sort | head -1)
MMPROJ=$(find $M -iname '*mmproj*F16*.gguf' | head -1)
[ -s "$GG" ] || fail "GGUF file not found after download"
log "gguf=$GG size=$(du -cb $(find $M -name "*${PAT}*.gguf" ! -iname '*mmproj*') | tail -1 | cut -f1) mmproj=${MMPROJ:-none}"

# 2) MTP heads?
NEXTN=$(python3 -I -c "
import sys; sys.path.insert(0, '/home/user/llama.cpp-new/gguf-py'); from gguf import GGUFReader
print(sum('nextn' in t.name or '.mtp' in t.name for t in GGUFReader(sys.argv[1]).tensors))" "$GG" 2>/dev/null | tail -1)
if [ "${NEXTN:-0}" -gt 0 ] 2>/dev/null; then SPEC="--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3"; else SPEC="--spec-type ngram-mod --spec-draft-n-max 3"; fi
log "mtp tensors=${NEXTN:-?} spec=$SPEC"

# 3) llama.cpp support: current build, else fresh build in separate dir (production build untouched)
systemctl --user stop llama-qwen; STOPPED=1; sleep 5
BIN=$LS
if up $BIN 8093 16384 1 $SPEC; then down; else
  log "current llama.cpp can't load: $(grep -iE 'error|unknown|fail' $D/server.log | tail -2 | tr '\n' ' ')"
  { [ -d $LC4 ] || git clone -q https://github.com/ggml-org/llama.cpp $LC4; } && git -C $LC4 pull -q \
    && cmake -S $LC4 -B $LC4/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON > $D/build.log 2>&1 \
    && cmake --build $LC4/build --target llama-server -j 20 >> $D/build.log 2>&1 || fail "new llama.cpp build failed"
  BIN=$LC4/build/bin/llama-server
  log "llama.cpp new build $(git -C $LC4 log -1 --format=%h)"
  if ! up $BIN 8093 16384 1 $SPEC; then
    SPEC="--spec-type ngram-mod --spec-draft-n-max 3"   # maybe MTP for new arch unsupported
    up $BIN 8093 16384 1 $SPEC || fail "llama.cpp does not support Qwen4 yet: $(grep -iE 'error|unknown' $D/server.log | tail -1)"
    log "MTP not usable, spec=$SPEC"
  fi
  down
fi

# periodic status every 15 min
( while sleep 900; do
    he=$(wc -l < $R/$NAME/humaneval.jsonl 2>/dev/null || echo 0)
    hp=$(grep -c '"pass": true' $R/$NAME/humaneval.jsonl 2>/dev/null || echo 0)
    bn=$(ls $A/jobs/bfcl/$NAME/*/result.json 2>/dev/null | wc -l)
    ntfy "Test Qwen4 - $(date +%H:%M)" "Phase: $(tail -1 $OUT | cut -c23-)
HumanEval Qwen4: $hp/$he (Qwen3.8: 159/164)
BFCL Qwen4: $bn/123 tasks (Qwen3.8: 82.9%)"
  done ) & STAT=$!

# 4) max ctx (startup = KV allocated)
BEST=0
for ctx in 262144 229376 196608 180224 172032 163840 159744 155648 150000 143360 131072 114688 98304 65536; do
  if up $BIN 8093 $ctx 1 $SPEC; then V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); down
    log "maxctx $ctx OK vram=${V}MiB"; BEST=$ctx; break; fi
done
[ $BEST -gt 0 ] || fail "does not start even at 65k ctx"

# 5) speed
up $BIN 8093 $BEST 1 $SPEC || fail "server does not start at ctx $BEST"
python3 -I ~/llm-eval/mtptest.py 8093 qwen4 >> $OUT 2>&1
python3 -I ~/llm-eval/mtptest.py 8093 qwen4-depth 32000 >> $OUT 2>&1
down
TG=$(grep -P '^qwen4\tcodegen' $OUT | tail -1 | grep -oP 'tg=\K[0-9.]+')
ntfy "Qwen4 speed + context" "Max context: $BEST (Qwen3.8: 159744)
$(grep -P '^qwen4' $OUT | cut -f2,5 | tr '\t' ' ')
(Qwen3.8: codegen 88.8, chat 69.1, edit 165.3, 31k 68)"

# 6) HumanEval + refusals
log "humaneval"
LLAMA_SERVER=$BIN EVAL_PARALLEL=2 EVAL_CTX=131072,98304,65536 EVAL_EXTRA_ARGS="$SPEC" \
  python3 -I ~/llm-eval/eval.py $NAME "$GG" > $R/$NAME.log 2>&1
HE=$(grep -c '"pass": true' $R/$NAME/humaneval.jsonl 2>/dev/null || echo 0)
RF=$(python3 -I -c "import json,sys;print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get('refused')))" $R/$NAME/refusals.jsonl 2>/dev/null || echo ?)
log "humaneval pass=$HE/164 refusals=$RF"
ntfy "Qwen4 HumanEval" "Qwen4: $HE/164 (Qwen3.8: 159/164)"

# 7) BFCL
log "bfcl"
ok=0; for c in 131072 98304 65536; do up $BIN 8091 $c 2 $SPEC && { ok=1; SLOT=$((c/2)); break; }; done
[ $ok = 1 ] || fail "BFCL server does not start"
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
HO=$(grep -c '"pass": true' $R/qwen38-q4-mtp/humaneval.jsonl)
log "bfcl pass new=$BN old=$BO"
touch $D/DONE

# 8) decide
SUMMARY="BFCL: Qwen4 $BN/123 vs Qwen3.8 $BO/123
HumanEval: Qwen4 $HE vs $HO | Qwen4 refusals $RF/12
Codegen speed: Qwen4 ${TG:-?} vs 88.8 | ctx max $BEST vs 159744"
if [ "$BN" -ge $((BO + 2)) ] && [ "$HE" -ge $((HO - 1)) ]; then
  cp "$UNIT" "$UNIT.bak-preqwen4"
  PCTX=$BEST; [ $PCTX -gt 262144 ] && PCTX=262144
  sed -i -E \
    -e "s#-m %h/models/[^ ]+#-m $GG#" \
    -e "s#-c [0-9]+ -np 1#-c $PCTX -np 1#" \
    -e "s#%h/llama.cpp-(new|prod)/build[^/]*/bin/llama-server#$BIN#" \
    -e "\#-md %h/models/dflash/#d" \
    -e "s#--spec-type [^\\\\]+#$SPEC #" \
    -e "s#^Description=.*#Description=llama-server Qwen4-27B $PAT (LAN, for DeepSeek Harness)#" "$UNIT"
  if [ -n "$MMPROJ" ]; then sed -i "s#--mmproj [^ ]*#--mmproj $MMPROJ#" "$UNIT"
  else sed -i 's#--mmproj [^ ]* --no-mmproj-offload##' "$UNIT"; fi
  log "WINNER Qwen4 -> llama-qwen switched (backup $UNIT.bak-preqwen4)"
  ntfy "Qwen4 WINS - server updated" "$SUMMARY
llama-qwen now uses Qwen4 (ctx $PCTX). Model alias unchanged 'qwen3.8-27b'. dsh client: contextWindow $PCTX. Rollback: cp $UNIT.bak-preqwen4 $UNIT."
else
  log "Qwen4 not better -> kept Qwen3.8 (Qwen4 files kept in $M)"
  ntfy "Qwen4 no better - keeping Qwen3.8" "$SUMMARY
Server unchanged. Qwen4 files kept in ~/models/qwen4 (safe to delete)."
fi
log "DONE"
