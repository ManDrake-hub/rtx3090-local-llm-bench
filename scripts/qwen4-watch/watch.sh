#!/usr/bin/env bash
# Run by qwen4-watch.timer 3x/day. Looks for Qwen4-27B GGUF on Hugging Face; when found starts qwen4-bench.
set -u
D=~/llm-eval/qwen4
NTFY=${NTFY_URL:-}
ntfy() { curl -s -H "Title: $1" -H "Tags: ${3:-mag}" -d "$2" "$NTFY" >/dev/null 2>&1; }
log() { echo "$(date '+%F %T') $*" >> $D/watch.log; }
[ -e $D/DONE ] && { systemctl --user disable --now qwen4-watch.timer; exit 0; }
systemctl --user is-active -q qwen4-bench && { log "bench running"; exit 0; }

# repo + quant file pattern (priority: unsloth UD-Q4_K_XL, then official Qwen Q4_K_M)
FOUND=$(python3 -I - <<'EOF'
import json, re, urllib.request
def get(u):
    return json.load(urllib.request.urlopen(u, timeout=30))
cands = []
for author in ('unsloth', 'Qwen'):
    try:
        ms = get(f'https://huggingface.co/api/models?author={author}&search=Qwen4&limit=100')
    except Exception:
        continue
    for m in ms:
        if re.fullmatch(rf'{author}/Qwen4(\.\d+)?-27B(-Instruct)?-GGUF', m['id'], re.I):
            cands.append(m['id'])
for repo in cands:
    try:
        files = [s['rfilename'] for s in get(f'https://huggingface.co/api/models/{repo}')['siblings']]
    except Exception:
        continue
    for pat in ('UD-Q4_K_XL', 'Q4_K_M'):
        if any(pat in f and f.endswith('.gguf') and 'mmproj' not in f.lower() for f in files):
            print(repo, pat); raise SystemExit
EOF
)
if [ -z "$FOUND" ]; then log "not found"; exit 0; fi
log "found $FOUND"
[ -e $D/FOUND_NOTIFIED ] || { ntfy "Qwen4-27B is out!" "Found on HF: $FOUND. Starting download + benchmark vs Qwen3.8 (~3h, server :8080 down during tests)." tada; touch $D/FOUND_NOTIFIED; }
systemd-run --user --unit=qwen4-bench --collect bash $D/bench.sh $FOUND
