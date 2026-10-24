"""Paired same-host VM32 scorer: candidate run dir vs reference run dir (same arithmetic as grade_linux_markline_v1:
cross-half U at the final layer, adjusted ratio = (u_c/u_r) * (flops_c/flops_r) per rank). Prints all32 / first7 / held25.
usage: python score_pair_v1.py <cand_run_dir> <ref_run_dir> [--plan <plan.json>] [--json out.json]
"""
import argparse, hashlib, json, math, statistics
from pathlib import Path
import numpy as np
ROOT = Path('/home/arcworker/arc-workspace')
DEFAULT_PLAN = ROOT / 'pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/runtime_first128_frozen_census_v1/NATIVE_PLAN_LINUX_v1.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
_TRUTH = {}

def truths(plan, rank):
    if rank not in _TRUTH:
        out = []
        for half in ('A', 'B'):
            rec = plan['rows'][str(rank)]['truth_' + half]
            path = ROOT / rec['path']
            with np.load(path, allow_pickle=False) as data:
                t = data[rec['key']].astype(np.float64)
            assert t.shape == (16, 1024) and np.isfinite(t).all()
            out.append(t)
        _TRUTH[rank] = out
    return _TRUTH[rank]

def collect(root):
    rows = {}
    for path in sorted(Path(root).glob('part*/rows.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('phase') != 'scored':
                continue
            rank = int(row['rank']); pred = path.parent / f'pred_{rank}.bin'
            if not pred.exists() or pred.stat().st_size != 65536:
                continue
            value = np.fromfile(pred, dtype='<f4').reshape(16, 1024).astype(np.float64)
            rows[rank] = dict(row=row, prediction=value, pred_bytes=pred.read_bytes())
    return rows

def summary(rs):
    if not rs:
        return dict(n=0)
    r = dict(n=len(rs), wins=int(sum(x['adjusted_pct'] < 0 for x in rs)))
    for key in ('raw_pct', 'bill_pct', 'adjusted_pct', 'wall_pct', 'requests_pct', 'puts_pct', 'f4_pct', 'residual_pct', 'backend_pct'):
        v = [x[key] for x in rs]
        r[key] = statistics.mean(v); r[key + '_se'] = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None
    return r

def main():
    p = argparse.ArgumentParser(); p.add_argument('cand'); p.add_argument('ref')
    p.add_argument('--plan', default=str(DEFAULT_PLAN)); p.add_argument('--json'); p.add_argument('--layers', action='store_true')
    a = p.parse_args(); plan = json.loads(Path(a.plan).read_text())
    cand, ref = collect(a.cand), collect(a.ref)
    ranks = [r for r in ref if r in cand]
    missing = [r for g in plan['groups'] for r in g if r not in cand]
    result = []
    for rank in ranks:
        c, b = cand[rank], ref[rank]; A, B = truths(plan, rank)
        u1 = np.mean((c['prediction'] - A) * (c['prediction'] - B), axis=1)
        u0 = np.mean((b['prediction'] - A) * (b['prediction'] - B), axis=1)
        fr = c['row']['flops'] / b['row']['flops']
        result.append(dict(rank=rank, raw_pct=float(100 * (u1[15] / u0[15] - 1)), bill_pct=100 * (fr - 1),
                           adjusted_pct=float(100 * (u1[15] / u0[15] * fr - 1)), u1=float(u1[15]), u0=float(u0[15]),
                           flops=c['row']['flops'], ref_flops=b['row']['flops'], identical=c['pred_bytes'] == b['pred_bytes'],
                           wall_s=c['row']['wall_s'], ref_wall_s=b['row']['wall_s'], residual_s=c['row']['residual_s'],
                           wall_pct=100 * (c['row']['wall_s'] / b['row']['wall_s'] - 1), requests_pct=100 * (c['row']['requests'] / b['row']['requests'] - 1),
                           puts_pct=100 * (c['row']['puts'] / b['row']['puts'] - 1), residual_pct=100 * (c['row']['residual_s'] / b['row']['residual_s'] - 1),
                           backend_pct=100 * (c['row']['backend_s'] / b['row']['backend_s'] - 1), f4_pct=100 * (c['row']['put_bytes']['<f4'] / b['row']['put_bytes']['<f4'] - 1),
                           requests=c['row']['requests'], ref_requests=b['row']['requests'],
                           sm40_calls=(c['row'].get('_ASTRA_SM40_STATS') or {}).get('calls'), ref_sm40_calls=(b['row'].get('_ASTRA_SM40_STATS') or {}).get('calls'),
                           layer_raw_pct=[100 * (float(x) / float(y) - 1) if y else None for x, y in zip(u1, u0)]))
    SLOW = (424, 320, 474, 672, 378, 310, 16, 623, 35, 548)
    slow = [r for r in result if r['rank'] in SLOW]
    slowctl = sorted(result, key=lambda r: -r['ref_wall_s'])[:10]   # the 10 slowest ranks of THIS pair's reference run
    slowctl_ratio = (sum(r['wall_s'] / r['ref_wall_s'] for r in slowctl) / len(slowctl)) if slowctl else None
    sm = [r['sm40_calls'] for r in result if r['sm40_calls'] is not None]; smr = [r['ref_sm40_calls'] for r in result if r['ref_sm40_calls'] is not None]
    rep = dict(cand=a.cand, ref=a.ref, n=len(result), missing=missing, identical=int(sum(r['identical'] for r in result)),
               sm40_calls_mean=(sum(sm) / len(sm)) if sm else None, ref_sm40_calls_mean=(sum(smr) / len(smr)) if smr else None,
               wall_max_rank_pct=max((r['wall_pct'] for r in result), default=None),
               slow10ctl_wall_ratio=slowctl_ratio, slow10ctl_ranks=[r['rank'] for r in slowctl],
               slow10ctl_wall_ratios=[round(r['wall_s'] / r['ref_wall_s'], 3) for r in slowctl],
               slow10=dict(n=len(slow), wall_pct=(sum(r['wall_pct'] for r in slow) / len(slow)) if slow else None,
                           requests_pct=(sum(r['requests_pct'] for r in slow) / len(slow)) if slow else None,
                           adjusted_pct=(sum(r['adjusted_pct'] for r in slow) / len(slow)) if slow else None),
               all=summary(result), first7=summary([r for r in result if r['rank'] in plan['first7']]),
               held25=summary([r for r in result if r['rank'] in plan['held25']]), rows=result)
    f = lambda s: 'n=%d adj %+.3f%% (se %.3f) raw %+.3f%% bill %+.3f%% wins %d' % (s['n'], s['adjusted_pct'], s['adjusted_pct_se'] or 0, s['raw_pct'], s['bill_pct'], s['wins']) if s['n'] else 'n=0'
    print('CAND', a.cand); print('REF ', a.ref)
    print('all32  ', f(rep['all'])); print('held25 ', f(rep['held25'])); print('first7 ', f(rep['first7']))
    print('identical preds: %d/%d; missing ranks: %s' % (rep['identical'], rep['n'], missing))
    print('wall %+.2f%% requests %+.2f%% puts %+.2f%% f4bytes %+.2f%% | max-rank wall %+.2f%% | slow10: wall %+.2f%% requests %+.2f%% adj %+.3f%%' % (
        rep['all']['wall_pct'], rep['all']['requests_pct'], rep['all']['puts_pct'], rep['all']['f4_pct'], rep['wall_max_rank_pct'],
        rep['slow10']['wall_pct'], rep['slow10']['requests_pct'], rep['slow10']['adjusted_pct']))
    print('per-rank wall%: ' + ' '.join('%d:%+.1f' % (r['rank'], r['wall_pct']) for r in result))
    print('sm40 calls mean cand %s ref %s' % (rep['sm40_calls_mean'], rep['ref_sm40_calls_mean']))
    print('backend %+.2f%% residual %+.2f%%' % (rep['all']['backend_pct'], rep['all']['residual_pct']))
    print('slow10ctl wall ratio %.3f over ranks %s: %s' % (rep['slow10ctl_wall_ratio'] or 0, rep['slow10ctl_ranks'], rep['slow10ctl_wall_ratios']))
    print('per-rank adj%: ' + ' '.join('%d:%+.2f' % (r['rank'], r['adjusted_pct']) for r in result))
    if a.layers and result:
        L = np.array([[x if x is not None else np.nan for x in r['layer_raw_pct']] for r in result])
        print('mean layer raw%: ' + ' '.join('%d:%+.2f' % (l, v) for l, v in enumerate(np.nanmean(L, axis=0))))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1) + '\n')

if __name__ == '__main__':
    main()
