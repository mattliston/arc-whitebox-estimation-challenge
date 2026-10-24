"""TAG opus_engineer a44: a candidate session against a base session, network by network (same rank list, same order): final-layer
MSE change, bill change, charged-op change, cpu. usage: compare_sessions.py <candidate session> <base session>"""
import json
import sys
from pathlib import Path
E = Path(__file__).resolve().parents[1]


def rows(name):
    path = E/f'logs/quick_{name}.json'
    if path.exists():
        return json.load(open(path))['rows']
    return [json.loads(l) for l in open(E/f'logs/quick_{name}.log') if l.startswith('{"rank"')]


cand, base = rows(sys.argv[1]), {r['rank']: r for r in rows(sys.argv[2])}
ch, bills, ops, lower = [], [], [], 0
for r in cand:
    b = base[r['rank']]
    ch.append(100 * (r['mse'] / b['mse'] - 1)); bills.append(100 * (r['flops'] / b['flops'] - 1)); ops.append(r['ops'] - b['ops']); lower += r['mse'] < b['mse']
    print(f"{r['rank']:5d}  mse {1e9 * r['mse']:.5f}e-9 ({ch[-1]:+.2f}%)  bill {r['flops'] / 1e9:.3f} G ({bills[-1]:+.3f}%)  ops {r['ops']} ({ops[-1]:+d})  cpu {r.get('cpu_s')} / {b.get('cpu_s')}")
n = len(cand)
pooled = 100 * (sum(r['mse'] for r in cand) / sum(base[r['rank']]['mse'] for r in cand) - 1)
mean = sum(ch) / n
se = (sum((x - mean) ** 2 for x in ch) / (n - 1) / n) ** 0.5 if n > 1 else float('nan')
print(f'{n} networks: pooled MSE {pooled:+.2f}%, mean of changes {mean:+.2f}% (se {se:.2f}), {lower} lower; bill {sum(bills) / n:+.3f}% mean; charged ops {sum(ops) / n:+.0f} mean')
