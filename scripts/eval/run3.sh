#!/usr/bin/env bash
# Third queue: waits for queue2 (ALL_DONE2), then models3.tsv with per-model parallel/ctx/extra args.
set -u
ROOT=~/llm-eval
BIN=~/llama.cpp-new/build/bin
HF=~/miniconda3/bin/hf
cd "$ROOT"
status() { echo "$(date '+%F %T') $*" | tee -a "$ROOT/STATUS"; }
status "queue3 waiting for queue2"
# pre-download while waiting (disk/network only, no GPU)
grep -v '^#' models3.tsv | while IFS=$'\t' read -r name repo file par ctx extra; do
  [ -z "$name" ] && continue
  [ -s "$HOME/models/eval/$file" ] || "$HF" download "$repo" "$file" --local-dir "$HOME/models/eval" \
    >> "$ROOT/dl-queue3.log" 2>&1 || status "$name: DOWNLOAD FAILED"
done
status "queue3 downloads finished"
while [ ! -e "$ROOT/ALL_DONE2" ]; do
  systemctl --user is-active -q llm-eval2 || break
  sleep 60
done
status "queue3 start"
grep -v '^#' models3.tsv | while IFS=$'\t' read -r name repo file par ctx extra; do
  [ -z "$name" ] && continue
  out="$ROOT/results/$name"; mkdir -p "$out"
  gguf="$HOME/models/eval/$file"
  [ -s "$gguf" ] || { status "$name: MISSING $gguf"; continue; }
  if [ ! -s "$out/bench.json" ]; then
    status "$name: llama-bench"
    # shellcheck disable=SC2086
    "$BIN/llama-bench" -m "$gguf" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 4096 -n 128 -d 0,16384 -r 2 -o json $extra \
      > "$out/bench.json" 2> "$out/bench.log" || status "$name: bench failed"
  fi
  status "$name: eval"
  EVAL_PARALLEL="$par" EVAL_CTX="$ctx" EVAL_EXTRA_ARGS="$extra" python3 -I "$ROOT/eval.py" "$name" "$gguf" \
    >> "$out/eval.log" 2>&1 || status "$name: eval failed (see eval.log)"
  python3 -I "$ROOT/summarize.py" > /dev/null 2>&1
  status "$name: done"
done
python3 -I "$ROOT/summarize.py" > /dev/null 2>&1
status "queue3 finished"
touch "$ROOT/ALL_DONE3"
