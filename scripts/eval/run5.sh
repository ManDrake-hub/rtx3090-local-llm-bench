#!/usr/bin/env bash
# Insert: when nemotron bfcl finishes, pause queue4, run HumanEval+refusals+bench for Cold Fusion Q4_K_M,
# add Cold Fusion + Bonsai2 Q2_0 to models4.tsv, then restart queue4 (+4b). Harbor jobs resume.
set -u
ROOT=~/llm-eval; cd "$ROOT"
BIN=~/llama.cpp-new/build/bin
status() { echo "$(date '+%F %T') $*" | tee -a "$ROOT/STATUS"; }
NAME=coldfusion-q4km
GGUF=$HOME/models/eval/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-Q4_K_M.gguf
status "insert: waiting for nemotron bfcl + coldfusion download"
while [ ! -e agentic/jobs/bfcl/nemotron35-lightning-iq4nl/DONE ] || systemctl --user is-active -q dl-coldfusion; do
  systemctl --user is-active -q llm-eval4 || break
  sleep 20
done
status "insert: pausing queue4"
systemctl --user stop llm-eval4b; systemctl --user stop llm-eval4
sleep 5
pkill -f "llama-server -m .* --port 8091" ; pkill -f "proxy.py 8092"
docker ps --format '{{.Names}}' | grep '__env' | xargs -r docker rm -f > /dev/null
if [ -s "$GGUF" ]; then
  out=results/$NAME; mkdir -p $out
  [ -s $out/bench.json ] || "$BIN/llama-bench" -m "$GGUF" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 4096 -n 128 -d 0,16384 -r 2 -o json > $out/bench.json 2> $out/bench.log
  status "$NAME: eval"
  python3 -I eval.py $NAME "$GGUF" >> $out/eval.log 2>&1 || status "$NAME: eval failed"
  python3 -I summarize.py > /dev/null 2>&1
  status "$NAME: done"
  grep -q "^$NAME" models4.tsv || printf '%s\t%s\t2\t131072,98304,65536\t\n' $NAME "$GGUF" >> models4.tsv
else
  status "$NAME: MISSING gguf"
fi
grep -q '^bonsai2-q2_0-stock' models4.tsv || printf 'bonsai2-q2_0-stock\t%s\t2\t262144,196608,131072\t\n' "$HOME/models/eval/Ternary-Bonsai-2-27B-Q2_0.gguf" >> models4.tsv
status "insert: restarting queue4"
systemctl --user reset-failed llm-eval4 llm-eval4b 2>/dev/null
systemd-run --user --unit=llm-eval4 --working-directory=$ROOT --setenv=PATH=$HOME/.local/bin:/usr/bin:/bin bash run4.sh
sleep 3
systemd-run --user --unit=llm-eval4b --working-directory=$ROOT --setenv=PATH=$HOME/.local/bin:/usr/bin:/bin bash run4b.sh
