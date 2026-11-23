#!/usr/bin/env python3
"""Readable progress + comparison text for ntfy (phone)."""
import glob, json, os, time

E = os.path.expanduser('~/llm-eval')
NICE = {
    'qwen38-q4-mtp': 'Qwen Q4+MTP', 'flashnext-iq3_xxs-noskip': 'FlashNext IQ3', 'flashnext-q2_0-strata': 'FlashNext', 'qwen38-official': 'Qwen Q5',
    'qwen38-coder390': 'Coder390', 'coldfusion-q4km': 'ColdFusion', 'bonsai2-q2_0-stock': 'Bonsai',
    'flint-q4kxl': 'Flint', 'nanbeige43-33b-iq4xs': 'Nanbeige', 'nemotron35-lightning-iq4nl': 'Nemotron',
}
NEW = ('flashnext-q2_0-strata', 'qwen38-q4-mtp', 'flashnext-iq3_xxs-noskip')


def he_score(name):
    f = f'{E}/results/{name}/humaneval.jsonl'
    if not os.path.exists(f):
        return None
    rows = [json.loads(l) for l in open(f) if l.strip()]
    return sum(1 for r in rows if r.get('pass')), len(rows)


def bfcl_score(name):
    d = f'{E}/agentic/jobs/bfcl/{name}'
    if not os.path.isdir(d):
        return None
    p = n = 0
    for t in glob.glob(f'{d}/*/result.json'):
        try:
            r = (json.load(open(t)).get('verifier_result') or {}).get('rewards', {}).get('reward')
        except Exception:
            continue
        n += 1
        p += r is not None and float(r) >= 1
    return p, n


out = [f'📍 Phase: ']
# phase
he_q = he_score('qwen38-q4-mtp')
bf_f, bf_q = bfcl_score('flashnext-q2_0-strata'), bfcl_score('qwen38-q4-mtp')
if he_q and he_q[1] < 164:
    out[0] += f'HumanEval Qwen Q4+MTP ({he_q[1]}/164)'
elif he_q and not os.path.exists(f'{E}/results/qwen38-q4-mtp/EVAL_DONE'):
    out[0] += 'refusal test Qwen Q4+MTP (12 prompts)'
elif bfcl_score('flashnext-iq3_xxs-noskip') and bfcl_score('flashnext-iq3_xxs-noskip')[1] < 123:
    out[0] += f"BFCL FlashNext IQ3 ({bfcl_score('flashnext-iq3_xxs-noskip')[1]}/123)"
elif bf_q and bf_q[1] < 123:
    out[0] += f'BFCL Qwen Q4+MTP ({bf_q[1]}/123)'
elif bf_f and bf_f[1] < 123:
    out[0] += f'BFCL FlashNext ({bf_f[1]}/123)'
elif bf_q and bf_q[1] >= 123:
    out[0] += 'done ✅'
else:
    out[0] += 'starting BFCL...'

# HumanEval
out += ['', '🧪 HUMANEVAL (code writing)']
for m in NEW:
    s = he_score(m)
    if s and m != 'flashnext-iq3_xxs-noskip':
        tag = '' if s[1] >= 164 else f'  ⏳ {s[1]}/164'
        out.append(f'▶ {NICE[m]}: {s[0]}/{s[1]} = {100*s[0]/s[1]:.1f}%{tag}')
old = sorted(((m, he_score(m)) for m in NICE if m not in NEW and he_score(m)), key=lambda x: -x[1][0])
out.append('   others: ' + ' · '.join(f'{NICE[m]} {s[0]}' for m, s in old[:5]) + '  (out of 164)')

# BFCL
out += ['', '🔧 BFCL (tool calls / agents)']
for m in NEW:
    s = bfcl_score(m)
    if not s or s[1] == 0:
        out.append(f'▶ {NICE[m]}: waiting')
    else:
        tag = '' if s[1] >= 123 else f'  ⏳ {s[1]}/123'
        out.append(f'▶ {NICE[m]}: {100*s[0]/s[1]:.1f}% ({s[0]}/{s[1]}){tag}')
oldb = [(m, bfcl_score(m)) for m in NICE if m not in NEW]
oldb = sorted([(m, s) for m, s in oldb if s and s[1] >= 120], key=lambda x: -x[1][0] / x[1][1])
out.append('   ranking:')
for i, (m, s) in enumerate(oldb[:5], 1):
    out.append(f'   {i}. {NICE[m]} {100*s[0]/s[1]:.1f}%')

out += ['', '⚡ SPEED (tokens/s)',
        '▶ FlashNext: 128 chat · 152 code · 149 at 31k',
        '▶ Qwen Q4+MTP: 72 chat · 89 code · 68 at 31k',
        '   Bonsai 68 · Qwen Q5 without MTP 37']
print('\n'.join(out))
