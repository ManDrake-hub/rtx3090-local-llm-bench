#!/usr/bin/env bash
# Resumable pipeline: download -> llama-bench -> eval (HumanEval + refusals) for each model in models.tsv.
# Start detached:  setsid nohup ~/llm-eval/run.sh >> ~/llm-eval/run.log 2>&1 < /dev/null &
set -u
ROOT=~/llm-eval
BIN=~/llama.cpp-new/build/bin
HF=~/miniconda3/bin/hf
cd "$ROOT"
echo "$$" > "$ROOT/run.pid"
status() { echo "$(date '+%F %T') $*" | tee -a "$ROOT/STATUS"; }

status "pipeline start"
grep -v '^#' models.tsv | while IFS=$'\t' read -r name repo file; do
  [ -z "$name" ] && continue
  out="$ROOT/results/$name"; mkdir -p "$out"
  if [ "$repo" = local ]; then gguf="$file"; else
    gguf="$HOME/models/eval/$file"
    if [ ! -s "$gguf" ]; then
      status "$name: download $repo/$file"
      "$HF" download "$repo" "$file" --local-dir "$HOME/models/eval" >> "$out/download.log" 2>&1 \
        || { status "$name: DOWNLOAD FAILED"; continue; }
    fi
  fi
  if [ ! -s "$out/bench.json" ]; then
    status "$name: llama-bench"
    "$BIN/llama-bench" -m "$gguf" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 4096 -n 128 -d 0,16384 -r 2 -o json \
      > "$out/bench.json" 2> "$out/bench.log" || status "$name: bench failed"
  fi
  status "$name: eval"
  python3 -I "$ROOT/eval.py" "$name" "$gguf" >> "$out/eval.log" 2>&1 || status "$name: eval failed (see eval.log)"
  python3 -I "$ROOT/summarize.py" > /dev/null 2>&1
  status "$name: done"
done
python3 -I "$ROOT/summarize.py" > /dev/null 2>&1
status "pipeline finished"
touch "$ROOT/ALL_DONE"
