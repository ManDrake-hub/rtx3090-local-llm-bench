#!/usr/bin/env python3
"""Summarize queue4 Harbor jobs -> ~/llm-eval/results/summary4.md"""
import glob, json, os

A = os.path.expanduser('~/llm-eval/agentic/jobs')
rows = {}
benches = []
for res in sorted(glob.glob(f'{A}/*/*/result.json')):
    job = os.path.dirname(res)
    bench, model = job.split('/')[-2:]
    if bench not in benches:
        benches.append(bench)
    try:
        st = json.load(open(res))['stats']
    except Exception:
        continue
    passed = total = 0
    for trial_res in glob.glob(f'{job}/*/result.json'):
        try:
            t = json.load(open(trial_res))
        except Exception:
            continue
        vr = (t.get('verifier_result') or {}).get('rewards') or {}
        r = vr.get('reward')
        total += 1
        if r is not None and float(r) >= 1.0:
            passed += 1
    mean = None
    for ev in st.get('evals', {}).values():
        m = ev.get('metrics') or []
        if m and 'mean' in m[0]:
            mean = m[0]['mean']
    done = 'yes' if os.path.exists(f'{job}/DONE') else 'partial'
    rows.setdefault(model, {})[bench] = (mean, passed, total, st.get('n_errored_trials', 0), done)

lines = ['| model | ' + ' | '.join(benches) + ' |', '|---|' + '---|' * len(benches)]
for model, d in sorted(rows.items()):
    cells = []
    for b in benches:
        if b not in d:
            cells.append('-')
            continue
        mean, p, t, err, done = d[b]
        ms = f'{mean*100:.1f}%' if mean is not None else '?'
        cells.append(f'{ms} ({p}/{t}, err {err}{"" if done == "yes" else ", partial"})')
    lines.append(f'| {model} | ' + ' | '.join(cells) + ' |')
out = '\n'.join(lines) + '\n'
open(os.path.expanduser('~/llm-eval/results/summary4.md'), 'w').write(out)
print(out)
