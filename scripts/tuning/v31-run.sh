#!/usr/bin/env bash
# Unsloth Dynamic v3.1 Qwen3.8-27B UD-Q4_K_XL vs current (v3) Q4+MTP.
# Download as separate file, check MTP heads, speed, HumanEval+refusals, BFCL.
# Swap llama-qwen to v3.1 only if BFCL >= old+2 tasks, HumanEval >= old-1, codegen tg >= 90% old. Else delete v3.1.
set -u
A=~/llm-eval/agentic
R=~/llm-eval/results
OUT=$R/v31.tsv
NEW=~/models/Qwen3.8-27B-UD-Q4_K_XL-v31.gguf
OLD=~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
NAME=qwen38-q4-v31
LS=~/llama.cpp-new/build/bin/llama-server
UNIT=~/.config/systemd/user/llama-qwen.service
NTFY=${NTFY_URL:-}
export OPENAI_API_KEY=sk-local
export PATH=$HOME/.local/bin:$HOME/miniconda3/bin:$PATH
log() { echo "# $(date '+%F %T') $*" >> "$OUT"; }
ntfy() { curl -s -H "Title: $1" -H "Tags: bar_chart" -d "$2" "$NTFY" >/dev/null 2>&1; }
SRV=; PRX=; STAT=
restore() {
  kill ${PRX:-} ${SRV:-} ${STAT:-} 2>/dev/null; wait ${SRV:-} 2>/dev/null
  sleep 5; systemctl --user daemon-reload; systemctl --user start llama-qwen
  for _ in $(seq 120); do curl -sf localhost:8080/health | grep -q ok && break; sleep 2; done
  log "llama-qwen restored ($(systemctl --user is-active llama-qwen))"
}
trap restore EXIT

# periodic status every 15 min
( while sleep 900; do
    he=$(wc -l < $R/$NAME/humaneval.jsonl 2>/dev/null || echo 0)
    hp=$(grep -c '"pass": true' $R/$NAME/humaneval.jsonl 2>/dev/null || echo 0)
    bn=$(ls $A/jobs/bfcl/$NAME/*/result.json 2>/dev/null | wc -l)
    ntfy "Test v3.1 - $(date +%H:%M)" "Phase: $(tail -1 $OUT | cut -c23-)
HumanEval v3.1: $hp/$he (old 159/164)
BFCL v3.1: $bn/123 tasks done (old 82.9%)"
  done ) & STAT=$!

: > "$OUT"
ntfy "v3.1 test started" "Downloading Qwen3.8 Q4 Dynamic v3.1, then speed, HumanEval, BFCL (~4h). Server :8080 off during tests."

# 1) download
log "download"
D=~/models/v31dl; mkdir -p $D
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf --local-dir $D > ~/models/dl-v31.log 2>&1 \
  && mv $D/Qwen3.8-27B-UD-Q4_K_XL.gguf "$NEW"
rm -rf $D
if [ ! -s "$NEW" ]; then log "download FAILED"; ntfy "v3.1 test FAILED" "Download failed, server unchanged."; exit 1; fi
if cmp -s <(head -c 50000000 "$NEW") <(head -c 50000000 "$OLD") && [ $(stat -c%s "$NEW") = $(stat -c%s "$OLD") ]; then
  log "same file as old, abort"; rm -f "$NEW"; ntfy "Test v3.1" "File identical to the old one, nothing to do."; exit 0; fi
log "size new=$(stat -c%s "$NEW") old=$(stat -c%s "$OLD")"

# 2) MTP heads present?
NEXTN=$(PYTHONPATH=~/llama.cpp-new/gguf-py python3 -c "
import sys; from gguf import GGUFReader
r=GGUFReader(sys.argv[1]); print(sum('nextn' in t.name for t in r.tensors))" "$NEW" 2>&1 | tail -1)
log "nextn tensors=$NEXTN"
if ! [ "$NEXTN" -gt 0 ] 2>/dev/null; then
  ntfy "Test v3.1 STOP" "v3.1 file has no MTP heads (nextn=$NEXTN): would lose half the speed. Keeping the old one, v3.1 deleted."
  rm -f "$NEW"; exit 0; fi

systemctl --user stop llama-qwen; sleep 5

# 3) speed (same config as production)
log "speed"
$LS -m "$NEW" --host 127.0.0.1 --port 8093 -ngl 99 -fa on -c 150000 -np 1 -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --jinja --no-webui > $R/v31-server.log 2>&1 < /dev/null & SRV=$!
ok=0; for _ in $(seq 300); do kill -0 $SRV 2>/dev/null || break; curl -sf localhost:8093/health | grep -q ok && { ok=1; break; }; sleep 1; done
if [ $ok = 1 ]; then
  echo "# vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)MiB" >> $OUT
  python3 -I ~/llm-eval/mtptest.py 8093 v31 >> $OUT 2>&1
  python3 -I ~/llm-eval/mtptest.py 8093 v31-depth 32000 >> $OUT 2>&1
else
  log "server FAILED at 150k: $(grep -iE 'error|fail' $R/v31-server.log | tail -2)"
  ntfy "v3.1 test FAILED" "llama-server does not start with the v3.1 file at 150k ctx. Keeping the old one."; rm -f "$NEW"; exit 1
fi
kill $SRV; wait $SRV 2>/dev/null; SRV=; sleep 3
TG=$(grep -P '^v31\tcodegen' $OUT | grep -oP 'tg=\K[0-9.]+')
ntfy "v3.1 test speed" "$(grep -P '^v31' $OUT | cut -f2,5 | tr '\t' ' ')
(old: codegen 88.8, chat 69.1, edit 165.3, 31k 68)"

# 4) HumanEval + refusals (same as old run)
log "humaneval"
EVAL_PARALLEL=2 EVAL_CTX=131072,98304 EVAL_EXTRA_ARGS="--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3" \
  python3 -I ~/llm-eval/eval.py $NAME "$NEW" > $R/$NAME.log 2>&1
HE=$(grep -c '"pass": true' $R/$NAME/humaneval.jsonl 2>/dev/null || echo 0)
log "humaneval pass=$HE/164"
ntfy "Test v3.1 HumanEval" "v3.1: $HE/164 (old 159/164)"

# 5) BFCL (same as old run)
log "bfcl"
$LS -m "$NEW" --host 127.0.0.1 --port 8091 -ngl 99 -fa on -c 131072 -np 2 -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --jinja --no-webui > $A/server.log 2>&1 < /dev/null & SRV=$!
for _ in $(seq 300); do curl -sf localhost:8091/health | grep -q ok && break; sleep 1; done
python3 -I $A/proxy.py 8092 8091 < /dev/null & PRX=$!; sleep 1
mkdir -p $A/jobs/bfcl
( while sleep 60; do find $A/jobs -type f -size +1G \( -name '*.pane' -o -name '*.cast' -o -name '*.txt' \) -exec truncate -s 0 {} \; ; done ) & WD=$!
mi='{"max_input_tokens":65536,"max_output_tokens":16384,"input_cost_per_token":0,"output_cost_per_token":0}'
harbor run -d bfcl_parity@1.0 -a terminus-2 -m openai/local --ak api_base=http://127.0.0.1:8092/v1 --ak temperature=0.6 \
  --ak record_terminal_session=false --ak "model_info=$mi" --agent-timeout-multiplier 2 -n 2 \
  -o $A/jobs/bfcl --job-name $NAME -y -q > $A/jobs/bfcl/$NAME.log 2>&1 < /dev/null
log "bfcl rc=$? $(grep -oE 'Mean: [0-9.]+' $A/jobs/bfcl/$NAME.log | tail -1)"
kill $WD $PRX $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=; PRX=

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

# 6) decide
HO=$(grep -c '"pass": true' $R/qwen38-q4-mtp/humaneval.jsonl)
win=0
[ "$BN" -ge $((BO + 2)) ] && [ "$HE" -ge $((HO - 1)) ] && awk "BEGIN{exit !(${TG:-0} >= 0.9*88.8)}" && win=1
SUMMARY="BFCL: v3.1 $BN/123 vs old $BO/123
HumanEval: v3.1 $HE vs $HO
Codegen speed: v3.1 ${TG:-?} vs 88.8"
if [ $win = 1 ]; then
  sed -i "s#Qwen3.8-27B-UD-Q4_K_XL.gguf#Qwen3.8-27B-UD-Q4_K_XL-v31.gguf#" "$UNIT"
  log "WINNER v3.1 -> llama-qwen switched (old file kept)"
  ntfy "v3.1 WINS - server updated" "$SUMMARY
llama-qwen now uses v3.1. Old file kept as backup."
else
  rm -f "$NEW"
  log "v3.1 not better -> kept old, v3.1 deleted"
  ntfy "v3.1 no better - keeping the old one" "$SUMMARY
v3.1 file deleted, server unchanged."
fi
log "DONE"
