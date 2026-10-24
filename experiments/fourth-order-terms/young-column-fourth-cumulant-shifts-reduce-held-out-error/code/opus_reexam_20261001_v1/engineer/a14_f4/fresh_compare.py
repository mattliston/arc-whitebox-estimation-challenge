"""TAG opus_engineer a14 F4 on fresh networks: a candidate's quick sessions against RRPI's native rows stored by the release lane
(O/release/runs/_native/a3efb3a10665_fc945b3a_a1 = ranks 651..390, a3efb3a10665_77a29add_a1 = ranks 293..259: the first 20 ranks of
O/trainer/cert/runB_val.txt, nobody trained or selected on them), same lists, same order, one process per list.
usage: fresh_compare.py <quick name> [<quick name> ...] [--base <quick name> ...]
Reads stored rows and predictions only. Per network: bill change, final-layer MSE change (organizers' truth), and the torch card's
F4 change on the same network (O/physicist/t3_pred_rank*_fresh24.npz, on_0 -> on_0.4) where stored. Pooled: bill, MSE with a
paired bootstrap over networks, adjusted score (MSE x bill), all-layer MSE.
With --base, the reference is those quick sessions (another build of mine) instead of RRPI's rows."""
import json
import sys
from pathlib import Path
import numpy as np

E = Path(__file__).resolve().parents[1]
O = E.parent
P2 = O.parent
args = sys.argv[1:]
names = args[:args.index('--base')] if '--base' in args else args
bases = args[args.index('--base') + 1:] if '--base' in args else []
ref = {}
if bases:
    for name in bases:
        for row in json.load(open(E/f'logs/quick_{name}.json'))['rows']:
            ref[row['rank']] = (row['flops'], E/f"logs/quick_{name}_pred_{row['rank']}.npy", row.get('ops'))
else:
    for d in ('a3efb3a10665_fc945b3a_a1', 'a3efb3a10665_77a29add_a1'):
        for line in open(O/'release/runs/_native'/d/'rows.jsonl'):
            row = json.loads(line)
            ref[row['rank']] = (row['flops'], O/'release/runs/_native'/d/f"pred_{row['rank']}.npy", None)
rows = []
print(f"{'rank':>5s} {'bill':>8s} {'MSE x1e9':>9s} {'MSE':>8s} {'all-layer':>9s} {'torch F4':>9s} {'ops':>7s}")
for name in names:
    for row in json.load(open(E/f'logs/quick_{name}.json'))['rows']:
        rank = row['rank']
        if rank not in ref:
            continue
        truth = np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
        y = np.load(E/f'logs/quick_{name}_pred_{rank}.npy').astype(np.float64)
        y0 = np.load(ref[rank][1]).astype(np.float64)
        m, m0 = np.mean((y[15] - truth[15]) ** 2), np.mean((y0[15] - truth[15]) ** 2)
        a, a0 = np.mean((y - truth) ** 2), np.mean((y0 - truth) ** 2)
        tf = O/f'physicist/t3_pred_rank{rank:04d}_fresh24.npz'
        torch_change = None
        if tf.exists():
            z = np.load(tf)
            t0, t1 = (np.mean((np.maximum(z[k][15], 0) - z['truth'][15]) ** 2) for k in ('on_0', 'on_0.4'))
            torch_change = 100 * (t1 / t0 - 1)
        rows.append((rank, row['flops'], ref[rank][0], m, m0, a, a0, torch_change))
        print(f"{rank:5d} {100*(row['flops']/ref[rank][0]-1):+7.3f}% {m*1e9:9.4f} {100*(m/m0-1):+7.2f}% {100*(a/a0-1):+8.2f}% "
              + (f'{torch_change:+8.2f}%' if torch_change is not None else '        -') + f" {row.get('ops', 0):7d}")
r = np.array([[x[1], x[2], x[3], x[4], x[5], x[6]] for x in rows], dtype=np.float64)
n = len(r)
pooled = lambda idx: (r[idx, 2].sum() / r[idx, 3].sum() - 1, r[idx, 0].sum() / r[idx, 1].sum() - 1,
                      (r[idx, 2] * r[idx, 0]).sum() / (r[idx, 3] * r[idx, 1]).sum() - 1, r[idx, 4].sum() / r[idx, 5].sum() - 1)
rng = np.random.default_rng(0)
boot = np.array([pooled(rng.integers(0, n, n)) for _ in range(4000)])
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
point = pooled(np.arange(n))
per = r[:, 2] / r[:, 3] - 1
print(f'{n} networks; reference: ' + (', '.join(bases) if bases else "RRPI's native rows (release lane)"))
for label, i in (('final-layer MSE, pooled', 0), ('bill, pooled', 1), ('adjusted (MSE x bill per network, summed)', 2), ('all-layer MSE, pooled', 3)):
    print(f'  {label:42s} {100*point[i]:+7.3f}%  [95% {100*lo[i]:+.3f}, {100*hi[i]:+.3f}]')
print(f'  final-layer MSE per network: mean {100*per.mean():+.3f}%, standard error {100*per.std(ddof=1)/np.sqrt(n):.3f}%, lower on {int((per < 0).sum())} of {n}, '
      f'best {100*per.min():+.2f}%, worst {100*per.max():+.2f}%')
torch = np.array([x[7] for x in rows if x[7] is not None])
if len(torch):
    mine = np.array([100 * (x[3] / x[4] - 1) for x in rows if x[7] is not None])
    print(f'  torch F4 on the same {len(torch)} networks: per-network mean {torch.mean():+.3f}%; native minus torch: mean {np.mean(mine - torch):+.3f}%, '
          f'sd {np.std(mine - torch, ddof=1):.3f}%, correlation {np.corrcoef(mine, torch)[0, 1]:.3f}')
