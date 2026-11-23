#!/usr/bin/env bash
# Qwen3.8-Flash-Next Q2_0 via Strata: speed + HumanEval, then restore llama-qwen.
set -u
S=~/.local/bin/strata
OUT=~/llm-eval/results/strata-flashnext.tsv
P=8094
log() { echo "# $(date '+%T') $*" >> "$OUT"; }
restore() { $S launch --stop >/dev/null 2>&1; systemctl --user start llama-qwen; log "llama-qwen restored"; }
trap restore EXIT
: > "$OUT"
systemctl --user stop llama-qwen; sleep 3
log "engine update"
$S update --yes > ~/strata-dl/update.log 2>&1 || { log "update FAILED: $(tail -3 ~/strata-dl/update.log | tr '\n' ' ')"; exit 1; }
log "launch"
$S launch --family qwen --model q2_0 --context 128k --parallel 1 --vision off --access local --port $P \
  --no-api-key --tune skip --yes --no-tui > ~/strata-dl/launch.log 2>&1 || { log "launch FAILED: $(tail -5 ~/strata-dl/launch.log | tr '\n' ' ')"; exit 1; }
for _ in $(seq 900); do curl -sf localhost:$P/health >/dev/null && break; sleep 2; done
curl -sf localhost:$P/health >/dev/null || { log "server not healthy"; exit 1; }
log "up; vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) ram=$(free -g | awk '/Mem/{print $3"G used"}')"
python3 -I ~/llm-eval/mtptest.py $P strata-q2 >> "$OUT" 2>&1
python3 -I ~/llm-eval/mtptest.py $P strata-q2-depth 32000 >> "$OUT" 2>&1
log "humaneval start"
EVAL_EXTERNAL_PORT=$P EVAL_PARALLEL=1 python3 -I ~/llm-eval/eval.py flashnext-q2_0-strata none > ~/strata-dl/eval.log 2>&1
log "humaneval rc=$? $(tail -2 ~/strata-dl/eval.log | tr '\n' ' ')"
log "DONE"
