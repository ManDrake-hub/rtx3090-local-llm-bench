#!/usr/bin/env bash
# Lower llama-qwen ctx until the stress test passes with >=200 MiB min free VRAM.
set -u
U=~/.config/systemd/user/llama-qwen.service
OUT=~/llm-eval/results/stress.log
: > $OUT
cp $U $U.bak-prectxfix
for ctx in 147456 139264 131072; do
  sed -i -E "s/-c [0-9]+ -np 1/-c $ctx -np 1/" $U
  systemctl --user daemon-reload; systemctl --user restart llama-qwen
  for _ in $(seq 90); do curl -sf localhost:8080/health | grep -q ok && break; sleep 2; done
  echo "== ctx=$ctx $(date +%T) buf200fail=$(journalctl --user _PID=$(systemctl --user show llama-qwen -p MainPID --value) --no-pager | grep -c 'layer_inp_dev_alloc: failed')" >> $OUT
  python3 -I ~/llm-eval/stress.py $ctx >> $OUT 2>&1
  rc=$?; mf=$(grep -oP '^PASS.*minfree=\K[0-9]+' $OUT | tail -1)
  echo "== ctx=$ctx rc=$rc minfree=${mf:-?}" >> $OUT
  [ $rc = 0 ] && [ "${mf:-0}" -ge 200 ] && { echo "CHOSEN $ctx" >> $OUT; break; }
done
echo DONE >> $OUT
