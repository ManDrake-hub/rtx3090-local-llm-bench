#!/usr/bin/env bash
# After stratatest: BFCL on Flash-Next (Strata), then HumanEval+refusals+BFCL on Qwen3.8 Q4+MTP. Restore llama-qwen at end.
set -u
S=~/.local/bin/strata
A=~/llm-eval/agentic
OUT=~/llm-eval/results/iq3.tsv
export OPENAI_API_KEY=sk-local
export PATH=$HOME/.local/bin:$PATH
log() { echo "# $(date '+%F %T') $*" >> "$OUT"; }
restore() { $S launch --stop >/dev/null 2>&1; kill ${SRV:-0} ${PRX:-0} 2>/dev/null; systemctl --user start llama-qwen; log "llama-qwen restored"; }
trap restore EXIT
while systemctl --user is-active -q bfclrun || pgrep -f "model IQ3_XXS" >/dev/null; do sleep 30; done
[ -n "$(ls ~/.local/share/strata/Strata-data/models/IQ3_XXS/*.gguf 2>/dev/null)" ] || { log "IQ3_XXS download missing"; exit 1; }
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

# Flash-Next IQ3_XXS: speed + BFCL
log "strata launch iq3_xxs"
$S launch --set STRATA_ROUTE_TAIL_SKIP=0 --family qwen --model iq3_xxs --context 128k --parallel 1 --vision off --access local --port 8094 \
  --no-api-key --tune skip --yes --no-tui > ~/strata-dl/launch-iq3.log 2>&1
for _ in $(seq 900); do curl -sf localhost:8094/health >/dev/null && break; sleep 2; done
curl -sf localhost:8094/health >/dev/null || { log "iq3 server not healthy: $(tail -3 ~/strata-dl/launch-iq3.log | tr '\n' ' ')"; exit 1; }
log "up; vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) ram=$(free -g | awk '/Mem/{print $3"G"}')"
python3 -I ~/llm-eval/mtptest.py 8094 iq3 >> "$OUT" 2>&1
python3 -I ~/llm-eval/mtptest.py 8094 iq3-depth 32000 >> "$OUT" 2>&1
python3 -I $A/proxy.py 8095 8094 < /dev/null & PRX=$!; sleep 1
bfcl flashnext-iq3_xxs-noskip 8095 1 131072
kill $PRX; PRX=
python3 -I $A/summarize4.py > /dev/null 2>&1
log "DONE"
