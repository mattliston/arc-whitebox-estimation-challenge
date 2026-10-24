#!/usr/bin/env python3
"""TAG fable-physicist (OT-3): exact per-layer MAC model of the bank's shared-old384 tier (candidate.py _shared_old384_push/_read +
the card's diagonal / omitted-radial consumers), from the r35 dump sizes (fable_oldtier_economics_v1/bank_r35/layer_LL.npz), and the
same model for candidate exact / approximate representations. Units: 1 unit = 2^31 FLOPs = 2^30 MACs (dense, before Winograd).
Options: base | resp (rebuild only response rows) | norot (join rotation r^2 Nprev replaced by per-join-group r x r products) |
sub<A>_<r2> (cohorts aged >= A in a nested rank-r2 sub-frame) and combinations."""
import numpy as np, json
from pathlib import Path
D = Path(__file__).resolve().parents[2] / 'fable_oldtier_economics_v1' / 'bank_r35'
n, r = 1024, 384
rows = []
for l in range(5, 15):
    z = np.load(D / f'layer_{l:02d}.npz')
    coh = z['cohorts']; N = int(z['old_columns']); resp = len(z['response']); live = len(z['live'])
    prev_alive = int(np.load(D / f'layer_{l-1:02d}.npz')['alive'].sum())
    Nprev = int(coh[l - 6].sum()) if l > 5 else 0      # old columns before this join
    width = N - Nprev
    rows.append(dict(l=l, N=N, Nprev=Nprev, width=width, resp=resp, live=live, act=prev_alive, cohorts=coh[:l - 4].tolist()))
def cost(row, resp_only=False, norot=False, sub=None):
    l, N, Nprev, width, resp, live, act = (row[k] for k in ('l', 'N', 'Nprev', 'width', 'resp', 'live', 'act'))
    c = {}
    c['join'] = 2 * act * width * r + (2 * act * r * r + r ** 3 if Nprev else 0) + 2 * act * r * r + r * act * width + r * width * r + (r * act * r + 2 * r ** 3 if Nprev else 0)
    groups = l - 5 + 1
    c['rotation'] = (groups * r ** 3 + groups * n * r * r) if norot else r * r * Nprev
    c['basis_push'] = n * act * r
    if sub is None:
        c['rebuild'] = (resp if resp_only else n) * r * N
        c['inner'] = resp * N * r
    else:
        A, r2 = sub
        # cohorts aged >= A at layer l: born at layers <= l - A
        nsub = int(sum(w for (s, w) in row['cohorts'][:max(0, l - A + 1)])) if l - A + 1 > 0 else 0
        nsub = min(nsub, N); nmain = N - nsub
        c['rebuild'] = (resp if resp_only else n) * (r * nmain + r2 * nsub) + (n * r * r2 if nsub else 0)
        c['inner'] = resp * (nmain * r + nsub * r2) + (resp * r2 * r if nsub else 0)
        c['rotation'] = c['rotation'] * (nmain / max(N, 1)) + (r2 * r2 * nsub if nsub else 0) + (r * r * r2 if nsub else 0)
    c['plane'] = resp * r * live
    c['diag_omit'] = 5 * n * N
    return c
opts = dict(base={}, resp=dict(resp_only=True), norot=dict(norot=True), resp_norot=dict(resp_only=True, norot=True),
            sub9_256=dict(sub=(9, 256)), resp_norot_sub9_256=dict(resp_only=True, norot=True, sub=(9, 256)),
            resp_norot_sub8_256=dict(resp_only=True, norot=True, sub=(8, 256)))
U = 2 ** 30
tot = {}
for name, kw in opts.items():
    per = [cost(row, **kw) for row in rows]
    tot[name] = sum(sum(c.values()) for c in per) / U
    if name == 'base':
        print('layer  N  resp  | base units by part (join rot push rebuild inner plane diag)')
        for row, c in zip(rows, per):
            print(row['l'], row['N'], row['resp'], ' '.join(f"{k}:{v / U:.2f}" for k, v in c.items()), f"| total {sum(c.values()) / U:.2f}")
print(json.dumps({k: round(v, 1) for k, v in tot.items()}))
print(json.dumps({k: round(v / tot['base'], 3) for k, v in tot.items()}))
