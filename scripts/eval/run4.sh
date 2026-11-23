#!/usr/bin/env bash
# Queue 4: agentic / coding benchmarks via Harbor + terminus-2 agent, after queue3 (ALL_DONE3).
# Benchmark-major order: every model runs bench 1, then every model bench 2, ... so early results are comparable.
# Resumable: finished (model,bench) pairs have DONE marker; unfinished Harbor jobs are resumed.
set -u
ROOT=~/llm-eval
A=$ROOT/agentic
SERVER=~/llama.cpp-new/build/bin/llama-server
PORT=8091; PPORT=8092
export OPENAI_API_KEY=sk-local
cd "$ROOT"
status() { echo "$(date '+%F %T') $*" | tee -a "$ROOT/STATUS"; }

# name | dataset | extra harbor args
BENCHES=(
  "bfcl|bfcl_parity@1.0|"
  "tbench2|terminal-bench@2.0|"
  "swebench40|swebench-verified@1.0|-l 40"
  "livecodebench|livecodebench@6.0|"
)

SRV=; PRX=
stop_all() { [ -n "$PRX" ] && kill "$PRX" 2>/dev/null; [ -n "$SRV" ] && { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }; SRV=; PRX=; }
trap stop_all EXIT

start_server() { # gguf par ctx_list extra -> sets SRV, CTX
  local gguf=$1 par=$2 ctxs=$3 extra=$4 ctx
  for ctx in ${ctxs//,/ }; do
    # shellcheck disable=SC2086
    "$SERVER" -m "$gguf" --host 127.0.0.1 --port $PORT -ngl 99 -fa on -c "$ctx" -np "$par" -ctk q8_0 -ctv q8_0 \
      --jinja --no-webui $extra > "$A/server.log" 2>&1 < /dev/null &
    SRV=$!
    for _ in $(seq 600); do
      kill -0 $SRV 2>/dev/null || break
      curl -sf http://127.0.0.1:$PORT/health | grep -q ok && { CTX=$ctx; return 0; }
      sleep 1
    done
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=
    status "  server failed ctx=$ctx, trying smaller"
  done
  return 1
}

status "queue4 waiting for queue3"
while [ ! -e "$ROOT/ALL_DONE3" ]; do
  systemctl --user is-active -q llm-eval3 || break
  sleep 60
done
status "queue4 start"

for b in "${BENCHES[@]}"; do
  IFS='|' read -r bname ds bargs <<< "$b"
  grep -v '^#' models4.tsv | while IFS=$'\t' read -r name gguf par ctxs extra; do
    [ -z "$name" ] && continue
    job="$A/jobs/$bname/$name"
    [ -e "$job/DONE" ] && continue
    [ -s "$gguf" ] || { status "$name: MISSING $gguf"; continue; }
    status "$name: $bname start server"
    start_server "$gguf" "$par" "$ctxs" "$extra" || { status "$name: server could not start"; continue; }
    python3 -I "$A/proxy.py" $PPORT $PORT < /dev/null & PRX=$!
    sleep 1
    slot=$((CTX / par)); out=$((slot / 4)); [ $out -gt 16384 ] && out=16384
    mi="{\"max_input_tokens\":$slot,\"max_output_tokens\":$out,\"input_cost_per_token\":0,\"output_cost_per_token\":0}"
    status "$name: $bname running (ctx/slot=$slot, concurrency=$par)"
    # watchdog: runaway task output can make *.pane/*.cast files grow to tens of GB
    ( while sleep 60; do find "$A/jobs" -type f -size +1G \( -name '*.pane' -o -name '*.cast' -o -name '*.txt' \) -exec truncate -s 0 {} \; ; done ) & WD=$!
    if [ -e "$job/config.json" ]; then
      harbor job resume -p "$job" > "$job.log" 2>&1 < /dev/null
    else
      mkdir -p "$A/jobs/$bname"
      # shellcheck disable=SC2086
      harbor run -d "$ds" $bargs -a terminus-2 -m openai/local \
        --ak api_base=http://127.0.0.1:$PPORT/v1 --ak temperature=0.6 --ak record_terminal_session=false --ak "model_info=$mi" \
        --agent-timeout-multiplier 2 -n "$par" -o "$A/jobs/$bname" --job-name "$name" -y -q > "$job.log" 2>&1 < /dev/null
    fi
    rc=$?
    kill $WD 2>/dev/null
    stop_all
    if [ $rc -eq 0 ]; then
      touch "$job/DONE"
      status "$name: $bname done $(grep -oE 'Mean: [0-9.]+' "$job.log" | tail -1)"
    else
      status "$name: $bname harbor exit $rc (see $job.log)"
    fi
  done
done
python3 -I "$A/summarize4.py" > /dev/null 2>&1
status "queue4 finished"
touch "$ROOT/ALL_DONE4"
