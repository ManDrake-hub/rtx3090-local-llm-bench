#!/usr/bin/env bash
# Second queue (Bonsai 2 models). Waits for the main pipeline (ALL_DONE), then runs models2.tsv.
# Resumable like run.sh. Each row picks its own llama.cpp build.
set -u
ROOT=~/llm-eval
cd "$ROOT"
status() { echo "$(date '+%F %T') $*" | tee -a "$ROOT/STATUS"; }
status "queue2 waiting for main pipeline"
while [ ! -e "$ROOT/ALL_DONE" ]; do
  systemctl --user is-active -q llm-eval || break   # main pipeline died/stopped: go ahead anyway
  sleep 60
done
status "queue2 start"
grep -v '^#' models2.tsv | while IFS=$'\t' read -r name gguf bin; do
  [ -z "$name" ] && continue
  out="$ROOT/results/$name"; mkdir -p "$out"
  [ -s "$gguf" ] || { status "$name: MISSING $gguf"; continue; }
  [ -x "$bin/llama-server" ] || { status "$name: MISSING build $bin"; continue; }
  if [ ! -s "$out/bench.json" ]; then
    status "$name: llama-bench"
    "$bin/llama-bench" -m "$gguf" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 4096 -n 128 -d 0,16384 -r 2 -o json \
      > "$out/bench.json" 2> "$out/bench.log" || status "$name: bench failed"
  fi
  status "$name: eval"
  LLAMA_SERVER="$bin/llama-server" python3 -I "$ROOT/eval.py" "$name" "$gguf" >> "$out/eval.log" 2>&1 \
    || status "$name: eval failed (see eval.log)"
  python3 -I "$ROOT/summarize.py" > /dev/null 2>&1
  status "$name: done"
done
python3 -I "$ROOT/summarize.py" > /dev/null 2>&1
status "queue2 finished"
touch "$ROOT/ALL_DONE2"
