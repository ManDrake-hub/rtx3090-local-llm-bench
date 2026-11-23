#!/usr/bin/env bash
# ExecStopPost: always runs, even if bench is killed
docker rm -f hyperink >/dev/null 2>&1
docker network rm hyperink-net >/dev/null 2>&1
U=$(cat ~/llm-eval/hyperink/ACTIVE_UNIT 2>/dev/null) || exit 0
sleep 5; systemctl --user start "$U"; rm -f ~/llm-eval/hyperink/ACTIVE_UNIT
echo "# $(date '+%F %T') restored $U" >> ~/llm-eval/hyperink/bench.tsv
