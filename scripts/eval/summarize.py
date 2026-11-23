#!/usr/bin/env python3
"""Build results/summary.md from per-model results."""
import json, os, re

ROOT = os.path.expanduser('~/llm-eval')
rows = []
for name in sorted(os.listdir(os.path.join(ROOT, 'results'))):
    d = os.path.join(ROOT, 'results', name)
    if not os.path.isdir(d):
        continue
    he = [json.loads(l) for l in open(os.path.join(d, 'humaneval.jsonl'))] if os.path.exists(os.path.join(d, 'humaneval.jsonl')) else []
    rf = [json.loads(l) for l in open(os.path.join(d, 'refusals.jsonl'))] if os.path.exists(os.path.join(d, 'refusals.jsonl')) else []
    bench = {}
    bp = os.path.join(d, 'bench.json')
    if os.path.exists(bp):
        try:
            for b in json.load(open(bp)):
                key = ('pp' if b.get('n_prompt') else 'tg') + ('@d' + str(b.get('n_depth')) if b.get('n_depth') else '')
                bench[key] = round(b.get('avg_ts', 0), 1)
        except Exception:
            pass
    npass = sum(r['pass'] for r in he)
    trunc = sum(1 for r in he if r.get('finish') == 'length')
    toks = [r['tokens'] for r in he if r.get('tokens')]
    rows.append((name, f"{npass}/{len(he)}", f"{100*npass/len(he):.1f}%" if he else '-', trunc,
                 int(sum(toks)/len(toks)) if toks else '-', f"{sum(r['refused'] for r in rf)}/{len(rf)}",
                 bench.get('pp', '-'), bench.get('tg', '-'), bench.get('pp@d16384', '-'), bench.get('tg@d16384', '-'),
                 'yes' if os.path.exists(os.path.join(d, 'EVAL_DONE')) else 'partial'))
hdr = '| model | HumanEval | pass% | truncated | avg tokens | refusals | pp4096 t/s | tg128 t/s | pp@16k | tg@16k | done |\n|' + '---|' * 11
lines = [hdr] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]
open(os.path.join(ROOT, 'results', 'summary.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
