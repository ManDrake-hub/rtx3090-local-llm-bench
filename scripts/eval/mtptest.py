#!/usr/bin/env python3
"""Measure llama-server decode speed + draft acceptance on fixed prompts.
Usage: mtptest.py <port> <label> [depth_tokens]"""
import json, sys, time, urllib.request

PORT, LABEL = int(sys.argv[1]), sys.argv[2]
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 0

CODE = open(__file__.replace('mtptest.py', 'eval.py')).read()
PROMPTS = {
    'codegen': 'Write a Python module implementing an LRU cache with TTL expiry, thread safety, '
               'and a decorator interface. Include docstrings and pytest tests.',
    'chat': 'Explain in detail how TCP congestion control works (slow start, AIMD, fast retransmit, '
            'CUBIC, BBR) and compare their trade-offs.',
    'edit': 'Here is a Python file:\n```python\n' + CODE[:12000] + '\n```\n'
            'Rewrite the whole file adding type hints to every function. Output the full file.',
}


def run(prompt, max_tokens=1200):
    body = {'model': 'x', 'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'max_tokens': max_tokens,
            'messages': [{'role': 'user', 'content': prompt}],
            'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    t0 = time.time()
    o = json.load(urllib.request.urlopen(req, timeout=3600))
    if 'timings' in o:
        return o['timings']
    u = o.get('usage', {})  # external server without llama.cpp timings: wall-clock incl. prefill
    dt = time.time() - t0
    return {'prompt_per_second': 0, 'predicted_per_second': u.get('completion_tokens', 0) / dt,
            'predicted_n': u.get('completion_tokens', 0), 'prompt_n': u.get('prompt_tokens', 0)}


filler = ''
if DEPTH:
    unit = CODE
    filler = 'Background material (ignore unless relevant):\n'
    while len(filler) < DEPTH * 3.3:
        filler += unit
    filler = filler[:int(DEPTH * 3.3)] + '\n\n'

for name, p in PROMPTS.items():
    if DEPTH and name != 'codegen':
        continue
    t = run(filler + p)
    acc = ''
    if t.get('draft_n'):
        acc = f" draft_acc={t.get('draft_n_accepted', 0)}/{t['draft_n']} ({100 * t.get('draft_n_accepted', 0) / t['draft_n']:.0f}%)"
    print(f"{LABEL}\t{name}\tdepth={t.get('prompt_n', 0) + t.get('cache_n', 0)}\tpp={t['prompt_per_second']:.0f}"
          f"\ttg={t['predicted_per_second']:.1f}\tn={t['predicted_n']}{acc}", flush=True)
