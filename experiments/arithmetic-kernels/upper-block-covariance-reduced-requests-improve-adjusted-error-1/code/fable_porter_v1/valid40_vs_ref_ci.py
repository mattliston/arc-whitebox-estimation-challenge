#!/usr/bin/env python3
"""TAG fable-porter (PT-9): VALID40 gate report of a candidate against a REFERENCE candidate's run (both paired_eval_cached.py outputs on the
same ranks, e.g. S7-full vs S6b). Pooled raw ratio (sum MSE cand / sum MSE ref) and pooled adjusted ratio, each with a 95% paired bootstrap CI
over nets (percentile, 20000 draws, seed 0), wins, mean util, per-net adjusted change. usage: valid40_vs_ref_ci.py <cand.json> <ref.json>"""
import sys, json, random
C, R = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ranks = [r for r in C['ranks'] if 'mse' in C['results'].get(f'cand:{r}', {}) and 'mse' in R['results'].get(f'cand:{r}', {})]
c = [C['results'][f'cand:{r}'] for r in ranks]; f = [R['results'][f'cand:{r}'] for r in ranks]; n = len(ranks)
def pooled(key, idx): return sum(c[i][key] for i in idx) / sum(f[i][key] for i in idx)
rng = random.Random(0); boots = {k: [] for k in ('mse', 'adjusted', 'util')}
for _ in range(20000):
    idx = [rng.randrange(n) for _ in range(n)]
    for k in boots: boots[k].append(pooled(k, idx))
def ci(k): b = sorted(boots[k]); return b[int(0.025 * len(b))], b[int(0.975 * len(b)) - 1]
pct = lambda x: f"{100 * (x - 1):+.3f}%"
rel = sorted(100 * (c[i]['adjusted'] / f[i]['adjusted'] - 1) for i in range(n))
raw_w = sum(c[i]['mse'] < f[i]['mse'] for i in range(n)); raw_t = sum(c[i]['mse'] == f[i]['mse'] for i in range(n))
adj_w = sum(c[i]['adjusted'] < f[i]['adjusted'] for i in range(n))
uc, uf = sum(x['util'] for x in c) / n, sum(x['util'] for x in f) / n
print(f"TAG fable-porter PT-9: {C['candidate']} vs reference {R['candidate']} on {n} VALID40 nets (each net cold, own process, .venv)")
print(f"raw (pooled MSE ratio): {pct(pooled('mse', range(n)))}  95% CI [{pct(ci('mse')[0])}, {pct(ci('mse')[1])}]  wins {raw_w}/{n} (ties {raw_t})")
print(f"util: mean {uc:.6f} vs {uf:.6f} ({pct(uc / uf)}); pooled util (= pooled flops) {pct(pooled('util', range(n)))}  95% CI [{pct(ci('util')[0])}, {pct(ci('util')[1])}]")
print(f"adjusted (pooled): {pct(pooled('adjusted', range(n)))}  95% CI [{pct(ci('adjusted')[0])}, {pct(ci('adjusted')[1])}]  wins {adj_w}/{n}; per-net min {rel[0]:+.2f}% median {rel[n // 2]:+.2f}% max {rel[-1]:+.2f}%")
s = C['summary']; print(f"vs cached bank rows (#331824): adjusted {s['adj_ratio']:.5f}, raw {s['raw_ratio']:.5f}, util {s['util_cand']:.6f} vs {s['util_parent']:.6f}, adj wins {s['adj_wins']}/{s['n']}")
