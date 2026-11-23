#!/usr/bin/env bash
# HyperInkling (vLLM sm86, Inkling-2-Mini W4A16 + DSpark) vs Qwen3.8 prod. Report only, no swap.
# Container: no Internet (internal network), no ports published, model read-only, caps dropped, RAM capped.
set -u
H=~/llm-eval/hyperink; A=~/llm-eval/agentic; R=~/llm-eval/results; OUT=$H/bench.tsv; NAME=inkling2-mini-hyperink
IMG=ghcr.io/pento95/hyperinkling:0.1.2
REV=92de097d944d06b02a97f80efcdce9f2bc0f2305
M=~/models/inkling/hyperink-w4a16
NTFY=${NTFY_URL:-}
export OPENAI_API_KEY=sk-local PATH=$HOME/.local/bin:$HOME/miniconda3/bin:$PATH
log() { echo "# $(date '+%F %T') $*" >> "$OUT"; }
ntfy() { curl -s -H "Title: $1" -H "Tags: bar_chart" -d "$2" "$NTFY" >/dev/null 2>&1; }
fail() { log "FAIL: $1"; docker logs --tail 60 hyperink > $H/container-fail.log 2>&1; ntfy "HyperInkling: problem" "$1
Server restored."; exit 1; }
count() { grep -c "$1" "$2" 2>/dev/null || true; }
: > $OUT
ntfy "HyperInkling test started" "Downloading ~10 GB image + 17 GB weights (server still up), then ~3h of tests with :8080 down."
log "download"
hf download Pentium95/Inkling-2-Mini-W4A16-g128-gptq --revision $REV --local-dir $M > $H/dl.log 2>&1 || fail "weights download"
docker pull -q $IMG >> $H/dl.log 2>&1 || fail "image pull"
log "image $(docker image inspect -f '{{index .RepoDigests 0}} {{.Size}}' $IMG) weights $(du -sh $M | cut -f1)"

# reference: same streaming bench on the current production server
for u in llama-qwen llama-qwen-heretic; do systemctl --user is-active -q $u && echo $u > $H/ACTIVE_UNIT; done
[ -s $H/ACTIVE_UNIT ] || fail "no LLM server active (game mode?): not touching the GPU"
log "reference on $(cat $H/ACTIVE_UNIT)"
python3 -I $H/sbench.py 8080 $(cat $H/ACTIVE_UNIT) 0,32000,100000 ~/.config/llama/api-key >> $OUT 2>&1

systemctl --user stop llama-qwen llama-qwen-heretic; sleep 5
docker rm -f hyperink >/dev/null 2>&1; docker network create --internal hyperink-net >/dev/null 2>&1
docker run -d --name hyperink --gpus all --network hyperink-net --cap-drop ALL --security-opt no-new-privileges \
  --memory 24g --shm-size 12g --pids-limit 4096 -v $M:/app/models/ink:ro \
  -e MODEL=/app/models/ink -e HOST=0.0.0.0 -e PORT=8000 -e ENV_FILE= -e HF_HUB_OFFLINE=1 \
  -e KV_OFFLOAD_GB=4 -e MM_CACHE_GB=1 -e MAX_SEQS=2 $IMG > /dev/null 2>> $OUT || fail "docker run"
IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' hyperink)
ok=0; for _ in $(seq 360); do
  curl -sf -m 3 http://$IP:8000/health >/dev/null && { ok=1; break; }
  [ "$(docker inspect -f '{{.State.Running}}' hyperink 2>/dev/null)" = true ] || break; sleep 5; done
[ $ok = 1 ] || fail "vLLM does not start: $(docker logs --tail 400 hyperink 2>&1 | grep -E 'Error|error' | tail -1 | cut -c1-250)"
sleep 60   # warmup
python3 -I $H/adapter.py 8091 $IP:8000 < /dev/null & python3 -I $H/adapter.py 8092 $IP:8000 --fill < /dev/null &
sleep 2
ML=$(curl -s http://$IP:8000/v1/models | python3 -I -c 'import json,sys;print(json.load(sys.stdin)["data"][0].get("max_model_len"))')
log "up max_len=$ML vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) ram=$(free -g | awk 'NR==2{print $7}')G free; $(docker logs hyperink 2>&1 | grep -iE 'KV cache size|Maximum concurrency' | tail -1 | cut -c1-160)"

# speed + long-context stability
python3 -I $H/sbench.py 8091 hyperink 0,32000,100000,200000 >> $OUT 2>&1
[ "$(docker inspect -f '{{.State.Running}}' hyperink)" = true ] || fail "container died during speed tests"
ntfy "HyperInkling speed" "Max context: $ML
$(grep -vP '^#' $OUT | cut -f1,2,3,5,6 | tr '\t' ' ')"

( while sleep 900; do
    f=$R/$NAME/humaneval.jsonl
    ntfy "HyperInkling - $(date +%H:%M)" "Phase: $(grep '^#' $OUT | tail -1 | cut -c23-120)
HumanEval: $(count '"pass": true' $f)/$(cat $f 2>/dev/null | wc -l) (Inkling llama.cpp 160, Qwen 159)
BFCL: $(ls $A/jobs/bfcl/$NAME/*/result.json 2>/dev/null | wc -l)/123 (Inkling 97, Qwen 102)"
  done ) &

log "humaneval"
EVAL_EXTERNAL_PORT=8091 EVAL_PARALLEL=2 python3 -I ~/llm-eval/eval.py $NAME external > $R/$NAME.log 2>&1
HE=$(count '"pass": true' $R/$NAME/humaneval.jsonl); ERR=$(count '"finish": "error' $R/$NAME/humaneval.jsonl)
log "humaneval pass=$HE/164 errors=$ERR length=$(count '"finish": "length"' $R/$NAME/humaneval.jsonl)"
[ "$(docker inspect -f '{{.State.Running}}' hyperink)" = true ] || fail "container died during HumanEval"
ntfy "HyperInkling HumanEval" "$HE/164, request errors: $ERR"

log "bfcl"
mkdir -p $A/jobs/bfcl
( while sleep 60; do find $A/jobs -type f -size +1G \( -name '*.pane' -o -name '*.cast' -o -name '*.txt' \) -exec truncate -s 0 {} \; ; done ) &
mi='{"max_input_tokens":65536,"max_output_tokens":16384,"input_cost_per_token":0,"output_cost_per_token":0}'
harbor run -d bfcl_parity@1.0 -a terminus-2 -m openai/local --ak api_base=http://127.0.0.1:8092/v1 --ak temperature=0.6 \
  --ak record_terminal_session=false --ak "model_info=$mi" --agent-timeout-multiplier 2 -n 2 \
  -o $A/jobs/bfcl --job-name $NAME -y -q > $A/jobs/bfcl/$NAME.log 2>&1 < /dev/null
log "bfcl rc=$? $(grep -oE 'Mean: [0-9.]+' $A/jobs/bfcl/$NAME.log | tail -1) container=$(docker inspect -f '{{.State.Running}}' hyperink)"
python3 -I - $A/jobs/bfcl >> $OUT <<'PY'
import glob, json, sys
for n in ('qwen38-q4-mtp', 'inkling2-mini-iq4nl', 'inkling2-mini-hyperink'):
    p = t = e = 0
    for f in glob.glob(f'{sys.argv[1]}/{n}/*/result.json'):
        d = json.load(open(f)); t += 1
        r = ((d.get('verifier_result') or {}).get('rewards') or {}).get('reward')
        p += r is not None and float(r) >= 1; e += bool(d.get('exception_info'))
    print(f'# BFCL {n}: {p}/{t} exceptions={e}')
PY
docker logs hyperink > $H/container.log 2>&1
ntfy "HyperInkling RESULT" "HumanEval $HE/164 (Qwen 159, Inkling llama.cpp 160)
$(grep '^# BFCL' $OUT | cut -c3-)
Context $ML
No automatic swap, server restored."
log DONE
