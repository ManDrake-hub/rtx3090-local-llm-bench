#!/usr/bin/env bash
# Push eval progress to ntfy (topic llm-3090, Tailscale only). New lines in decide*.tsv immediately; BFCL progress every 15 min.
N=${NTFY_URL:-}
R=~/llm-eval/results
send() { curl -s -H "Title: ${2:-LLM 3090}" -H "Tags: bar_chart" -d "$1" $N > /dev/null; }
declare -A seen
for f in $R/decide.tsv $R/decide-bfcl.tsv $R/iq3.tsv; do seen[$f]=$(wc -l < $f 2>/dev/null || echo 0); done
t=0
while true; do
  for f in $R/decide.tsv $R/decide-bfcl.tsv $R/iq3.tsv; do
    c=$(wc -l < $f 2>/dev/null || echo 0)
    if [ "$c" -gt "${seen[$f]}" ]; then
      send "$(sed -n "$((seen[$f]+1)),${c}p" $f | sed 's/^# 2026-[0-9-]* //')"; seen[$f]=$c
    fi
  done
  if [ $((t % 90)) -eq 0 ] && [ $t -gt 0 ]; then
    send "$(python3 -I ~/llm-eval/notify-status.py 2>&1)" "Benchmark LLM - $(date +%H:%M)"
  fi
  if grep -q 'llama-qwen restored' $R/iq3.tsv 2>/dev/null; then
    s=$(cd ~/llm-eval && python3 -I agentic/summarize4.py 2>/dev/null | grep -E 'flashnext|q4-mtp|official|bonsai' | awk -F'|' '{print $2": BFCL "$3}')
    send "FINITO. $s" "Benchmark finiti"; exit 0
  fi
  t=$((t+1)); sleep 10
done
