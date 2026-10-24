"""TAG opus-reexam: pooled summary of an arm table directory (rank*.log of e16b_k4_closed_loop.py): per network and pooled final-layer / all-layer change against the first arm.
usage: e51_summary.py <dir>"""
import sys, json, glob
import numpy as np
D = sys.argv[1]; rows = {}
for f in sorted(glob.glob(D + '/rank*.log')):
    for line in open(f):
        if line.startswith('{'):
            x = json.loads(line); rows.setdefault(x['arm'], {})[x['rank']] = (x['final_x1e9'], sum(x['per_layer_x1e9']))
base = next(iter(rows)); ranks = sorted(rows[base])
print('ranks', ranks)
for arm, d in rows.items():
    rk = [r for r in ranks if r in d]
    f = np.array([d[r][0] for r in rk]); b = np.array([rows[base][r][0] for r in rk]); fa = np.array([d[r][1] for r in rk]); ba = np.array([rows[base][r][1] for r in rk])
    per = 100 * (f / b - 1)
    print('%-10s n %d  pooled final %+.2f%%  mean of networks %+.2f%% (se %.2f)  all layers %+.2f%%  per network: %s' % (
        arm, len(rk), 100 * (f.sum() / b.sum() - 1), per.mean(), per.std(ddof=1) / max(len(rk), 1) ** 0.5 if len(rk) > 1 else float('nan'), 100 * (fa.sum() / ba.sum() - 1), ' '.join('%+.1f' % p for p in per)))
