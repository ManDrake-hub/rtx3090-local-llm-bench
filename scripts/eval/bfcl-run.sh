#!/usr/bin/env bash
# After stratatest: BFCL on Flash-Next (Strata), then HumanEval+refusals+BFCL on Qwen3.8 Q4+MTP. Restore llama-qwen at end.
set -u
S=~/.local/bin/strata
A=~/llm-eval/agentic
OUT=~/llm-eval/results/decide-bfcl.tsv
export OPENAI_API_KEY=sk-local
export PATH=$HOME/.local/bin:$PATH
log() { echo "# $(date '+%F %T') $*" >> "$OUT"; }
restore() { $S launch --stop >/dev/null 2>&1; kill ${SRV:-0} ${PRX:-0} 2>/dev/null; systemctl --user start llama-qwen; log "llama-qwen restored"; }
trap restore EXIT
while systemctl --user is-active -q decide; do sleep 30; done
systemctl --user stop llama-qwen; sleep 3

bfcl() { # name port par ctx_per_slot
  local name=$1 port=$2 par=$3 slot=$4; local out=$((slot/4)); [ $out -gt 16384 ] && out=16384
  local mi="{\"max_input_tokens\":$slot,\"max_output_tokens\":$out,\"input_cost_per_token\":0,\"output_cost_per_token\":0}"
  mkdir -p $A/jobs/bfcl
  ( while sleep 60; do find $A/jobs -type f -size +1G \( -name '*.pane' -o -name '*.cast' -o -name '*.txt' \) -exec truncate -s 0 {} \; ; done ) & local WD=$!
  harbor run -d bfcl_parity@1.0 -a terminus-2 -m openai/local --ak api_base=http://127.0.0.1:$port/v1 --ak temperature=0.6 \
    --ak record_terminal_session=false --ak "model_info=$mi" --agent-timeout-multiplier 2 -n $par \
    -o $A/jobs/bfcl --job-name $name -y -q > $A/jobs/bfcl/$name.log 2>&1 < /dev/null
  log "bfcl $name rc=$? $(grep -oE 'Mean: [0-9.]+' $A/jobs/bfcl/$name.log | tail -1)"
  kill $WD 2>/dev/null
}

# 1) Flash-Next BFCL
log "strata launch"
$S launch --family qwen --model q2_0 --context 128k --parallel 1 --vision off --access local --port 8094 \
  --no-api-key --tune skip --yes --no-tui > ~/strata-dl/launch2.log 2>&1
for _ in $(seq 900); do curl -sf localhost:8094/health >/dev/null && break; sleep 2; done
python3 -I $A/proxy.py 8095 8094 < /dev/null & PRX=$!; sleep 1
bfcl flashnext-q2_0-strata 8095 1 131072
kill $PRX; PRX=; $S launch --stop >/dev/null 2>&1; sleep 5

# 3) Qwen BFCL
~/llama.cpp-new/build/bin/llama-server -m ~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf --host 127.0.0.1 --port 8091 -ngl 99 -fa on \
  -c 131072 -np 2 -ctk q8_0 -ctv q8_0 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --jinja --no-webui \
  > $A/server.log 2>&1 < /dev/null & SRV=$!
for _ in $(seq 300); do curl -sf localhost:8091/health | grep -q ok && break; sleep 1; done
python3 -I $A/proxy.py 8092 8091 < /dev/null & PRX=$!; sleep 1
bfcl qwen38-q4-mtp 8092 2 65536
kill $PRX $SRV; wait $SRV 2>/dev/null; SRV=; PRX=
python3 -I $A/summarize4.py > /dev/null 2>&1
log "DONE"
