#!/usr/bin/env bash
# Pause queue4, find max loadable ctx per model with -np 1 (q8_0 and q4_0 KV), restart queue4.
ROOT=~/llm-eval; cd $ROOT
S=~/llama.cpp-new/build/bin/llama-server
OUT=$ROOT/results/ctxprobe.tsv
systemctl --user stop llm-eval4b; systemctl --user stop llm-eval4; sleep 5
pkill -f "llama-server -m .* --port 8091"; pkill -f "proxy.py 8092"
docker ps --format '{{.Names}}' | grep '__env' | xargs -r docker rm -f > /dev/null
sleep 3
echo -e "model\tkv\tctx\tok\tvram_mib" > $OUT
try() { # name gguf kv ctx extra
  $S -m "$2" --host 127.0.0.1 --port 8093 -ngl 99 -fa on -c $4 -np 1 -ctk $3 -ctv $3 --jinja --no-webui $5 > /tmp/ctxprobe.log 2>&1 &
  local p=$! ok=0 v=0
  for _ in $(seq 180); do
    kill -0 $p 2>/dev/null || break
    if curl -sf http://127.0.0.1:8093/health | grep -q ok; then
      ok=1; v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
      # real allocation check: one short completion
      curl -sf http://127.0.0.1:8093/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}' > /dev/null || ok=0
      break
    fi
    sleep 1
  done
  kill $p 2>/dev/null; wait $p 2>/dev/null
  echo -e "$1\t$3\t$4\t$ok\t$v" >> $OUT
  return $((1-ok))
}
grep -v '^#' models4.tsv | while IFS=$'\t' read -r n g p c e; do
  [ -z "$n" ] && continue
  case $n in
    nemotron*) L="1048576 786432 524288 393216 262144";;
    nanbeige*) L="262144 196608 131072 98304 65536 49152";;
    *) L="262144 229376 196608 163840 131072 98304";;
  esac
  for kv in q8_0 q4_0; do
    for c in $L; do try "$n" "$g" $kv $c "$e" < /dev/null && break; done
  done
done
systemctl --user reset-failed llm-eval4 llm-eval4b 2>/dev/null
systemd-run --user --unit=llm-eval4 --working-directory=$ROOT --setenv=PATH=$HOME/.local/bin:/usr/bin:/bin bash run4.sh
sleep 3
systemd-run --user --unit=llm-eval4b --working-directory=$ROOT --setenv=PATH=$HOME/.local/bin:/usr/bin:/bin bash run4b.sh
echo "$(date '+%F %T') ctxprobe done, queue4 restarted" >> $ROOT/STATUS
