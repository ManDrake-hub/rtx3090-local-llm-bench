#!/usr/bin/env python3
"""Streaming speed bench, same method for any OpenAI server (decode excludes prefill; unique nonce => no prefix cache).
Usage: sbench.py <port> <label> <depths comma> [api key file]"""
import json, os, sys, time, urllib.request
PORT, LABEL, DEPTHS = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
KEY = open(os.path.expanduser(sys.argv[4])).read().strip() if len(sys.argv) > 4 else ''
CODE = open(os.path.expanduser('~/llm-eval/eval.py')).read()
P = {'codegen': 'Write a Python module implementing an LRU cache with TTL expiry, thread safety, and a decorator interface. Include docstrings and pytest tests.',
     'chat': 'Explain in detail how TCP congestion control works (slow start, AIMD, fast retransmit, CUBIC, BBR) and compare their trade-offs.',
     'edit': 'Here is a Python file:\n```python\n' + CODE[:12000] + '\n```\nRewrite the whole file adding type hints to every function. Output the full file.'}


def run(name, depth, temp):
    nonce = f'[{LABEL}-{name}-{depth}-{time.time_ns()}]\n'
    fill = ''
    if depth:
        fill = 'Background material (ignore unless relevant):\n'
        while len(fill) < depth * 3.3: fill += CODE
        fill = fill[:int(depth * 3.3)] + '\n\n'
    b = {'model': 'x', 'temperature': temp, 'top_p': 0.95, 'top_k': 20, 'max_tokens': 1200, 'stream': True,
         'stream_options': {'include_usage': True}, 'messages': [{'role': 'user', 'content': nonce + fill + P[name]}]}
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', json.dumps(b).encode(),
                                 {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + KEY})
    t0 = time.time(); first = last = None; u = {}
    for line in urllib.request.urlopen(req, timeout=3600):
        if not line.startswith(b'data: ') or b'[DONE]' in line: continue
        o = json.loads(line[6:])
        if o.get('usage'): u = o['usage']
        for c in o.get('choices', []):
            d = c.get('delta') or {}
            if d.get('content') or d.get('reasoning') or d.get('reasoning_content') or d.get('tool_calls'):
                last = time.time(); first = first or last
    n, p = u.get('completion_tokens', 0), u.get('prompt_tokens', 0)
    print(f'{LABEL}\t{name}\tdepth={p}\tttft={first - t0:.2f}s\tpp={p / (first - t0):.0f}\ttg={(n - 1) / max(last - first, 1e-6):.1f}\tn={n}', flush=True)


for d in DEPTHS:
    for name in (P if d == 0 else ['codegen']):
        try: run(name, d, 0.6)
        except Exception as e:
            print(f'{LABEL}\t{name}\tdepth~{d}\tFAIL {e!r} {e.read()[:200] if hasattr(e, "read") else ""}', flush=True)
