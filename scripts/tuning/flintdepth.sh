#!/usr/bin/env bash
ROOT=~/llm-eval; cd $ROOT
systemctl --user stop llm-eval4b; systemctl --user stop llm-eval4; sleep 5
pkill -f "llama-server -m .* --port 8091"; pkill -f "proxy.py 8092"
docker ps --format '{{.Names}}' | grep '__env' | xargs -r docker rm -f > /dev/null
sleep 3
B=~/llama.cpp-new/build/bin/llama-bench; G=$HOME/models/eval/flint-UD-Q4_K_XL.gguf
$B -m $G -ngl 99 -fa 1 -ncmoe 10 -ctk q8_0 -ctv q8_0 -p 2048 -n 128 -d 0,32768,131072,225000 -r 1 -o md > results/flint-depth-q8.md 2>results/flint-depth-q8.log
$B -m $G -ngl 99 -fa 1 -ncmoe 10 -ctk q4_0 -ctv q4_0 -p 2048 -n 128 -d 258000 -r 1 -o md > results/flint-depth-q4.md 2>results/flint-depth-q4.log
systemctl --user reset-failed llm-eval4 llm-eval4b 2>/dev/null
systemd-run --user --unit=llm-eval4 --working-directory=$ROOT --setenv=PATH=$HOME/.local/bin:/usr/bin:/bin bash run4.sh
sleep 3
systemd-run --user --unit=llm-eval4b --working-directory=$ROOT --setenv=PATH=$HOME/.local/bin:/usr/bin:/bin bash run4b.sh
echo "$(date '+%F %T') flint depth bench done, queue4 restarted" >> $ROOT/STATUS
