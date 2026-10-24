"""TAG opus_engineer a53 (2026-10-10): the audit table. For each variant session (one approximation of the parent switched to exact),
against the parent's session on the same networks in the same order: final-layer MSE change per network and pooled (the session's own
truth), bill change, charged-op change; and how far the variant sits from torch (final layer, 688-truth): rms(native - torch) and the
gap native/torch - 1 split into noise + correlated parts, next to the parent's.
usage: audit_table.py <parent session> <torch tag, e.g. a_0mHpar> <variant session> [...]"""
import json
import sys
from pathlib import Path
import numpy as np
E = Path(__file__).resolve().parents[1]
O, P2 = E.parent, E.parents[1]


def rows(name):
    path = E/f'logs/quick_{name}.json'
    if path.exists():
        return {r['rank']: r for r in json.load(open(path))['rows']}
    return {r['rank']: r for r in (json.loads(l) for l in open(E/f'logs/quick_{name}.log') if l.startswith('{"rank"'))}


def torch_gap(session, tag, rank):
    tp = O/'trainer/run_M/results'/f'{tag}_r{rank:04d}.json'
    pp = E/f'logs/quick_{session}_pred_{rank}.npy'
    if not (tp.exists() and pp.exists()):
        return None
    truth = np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
    tor_all = np.asarray(json.load(open(tp))['prediction'], dtype=np.float64)
    nat_all = np.load(pp).astype(np.float64)
    tor, nat = tor_all[15], nat_all[15]
    e, d = tor - truth, nat - tor
    mt = float(np.mean(e * e))
    layers = ' '.join('%.2g' % (1e6 * float(np.sqrt(np.mean((nat_all[l] - tor_all[l]) ** 2)))) for l in range(16))
    return 1e6 * float(np.sqrt(np.mean(d * d))), 100 * float(np.mean(d * d)) / mt, 200 * float(np.mean(e * d)) / mt, layers


parent, tag = sys.argv[1], sys.argv[2]
base = rows(parent)
for v in sys.argv[3:]:
    cand = rows(v)
    ks = [k for k in cand if k in base]
    if not ks:
        print(f'{v}: no rows'); continue
    ch = ['%+.2f' % (100 * (cand[k]['mse'] / base[k]['mse'] - 1)) for k in ks]
    pooled = 100 * (sum(cand[k]['mse'] for k in ks) / sum(base[k]['mse'] for k in ks) - 1)
    bill = np.mean([100 * (cand[k]['flops'] / base[k]['flops'] - 1) for k in ks])
    ops = np.mean([cand[k]['ops'] - base[k]['ops'] for k in ks])
    imp = []
    for k in ks:   # implied cost: the power of the final-layer change over the parent's MSE (688-truth), the expected cost of a change unrelated to the error
        a, b = (E/f'logs/quick_{parent}_pred_{k}.npy'), (E/f'logs/quick_{v}_pred_{k}.npy')
        if a.exists() and b.exists():
            t = np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{k:04d}.npz')['mc_post'][15].astype(np.float64)
            pa, pb = np.load(a).astype(np.float64)[15], np.load(b).astype(np.float64)[15]
            imp.append('%.2f' % (100 * float(np.mean((pb - pa) ** 2)) / float(np.mean((pa - t) ** 2))))
    print(f'{v}: networks {ks}: final-layer MSE {"/".join(ch)}% (pooled {pooled:+.2f}%), implied {"/".join(imp)}%, bill {bill:+.3f}%, charged ops {ops:+.0f}')
    for k in ks:
        g, gp = torch_gap(v, tag, k), torch_gap(parent, tag, k)
        if g and gp:
            print(f'      {k} vs torch: rms {g[0]:.2f}e-6 (parent {gp[0]:.2f}), noise {g[1]:+.2f}% (parent {gp[1]:+.2f}), correlated {g[2]:+.2f}% (parent {gp[2]:+.2f})')
            print(f'         per layer (1e-6): {g[3]}')
            print(f'         parent          : {gp[3]}')
