"""TAG opus-release STAGE B: paired native validation of a candidate against a named parent on 40 networks: valid40 (the
historical validation set of fable_analyst_v1/split_train640.json) or fresh40 (newer ranks nobody trained or selected on),
numpy backend, one BLAS thread, one process at a time.

Both arms run the same rank lists in the same order, a fresh process per batch of 10 (bills depend on the cache history of the
process, so nothing else is comparable). Batches alternate parent/candidate. Each (source sha, rank list) run is stored once in
O/release/runs/_native/ and reused by later calls, so a parent is paid for once.
Reported: pooled final-layer MSE ratio (mean MSE over networks, candidate / parent), bill ratio, adjusted ratio (mean of
MSE x max(0.1, bill / 2^41)), wins (networks where the candidate's adjusted is lower), 95% bootstrap interval of the adjusted
change over networks (20,000 resamples), the worst single-network regression, and the CPU ratio in cycles and instructions.
Default gates are the bank gates used for S35: adjusted <= -1%, interval upper end < 0, wins >= 28 of 40, no network worse than
+10%. Override them for other purposes (e.g. a CPU variant: --max-adjusted-pct 2 --max-raw-pct 0.05 --min-wins 0 --ci-upper 100).
usage: stage_b_native40.py <candidate.py> <tag> --parent PATH [--set valid40|fresh40] [--ranks r1,r2,...] [--batch 10] [gate options]"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from common import BANK, FRESH40, R, RUNS, SERVER_PY, VALID40, guarded, run_dir, sha, verdict, write_json


def native(source, ranks, gate=True):
    """Rows and directory of one native batch; runs it only if no complete stored run exists."""
    text = ','.join(map(str, ranks))
    key = f"{sha(source)[:12]}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
    base = RUNS/'_native'
    base.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        out = base/f'{key}_a{attempt}'
        if not out.exists():
            break
        rows = [json.loads(x) for x in (out/'rows.jsonl').read_text().splitlines()] if (out/'rows.jsonl').exists() else []
        if [r['rank'] for r in rows] == list(ranks):
            return rows, out, True
        attempt += 1
    if gate:
        subprocess.run(['bash', str(R/'own_gate.sh'), '7', 'release'], check=True)
    g = guarded(base/f'{key}_a{attempt}_guard.json', 300 + 400*len(ranks), [SERVER_PY, '-B', R/'native_probe.py', source, text, out],
                base/f'{key}_a{attempt}.log')
    if not g['completed']:
        raise SystemExit(f"native batch {out.name} did not complete: {g['error']} (log {base}/{key}_a{attempt}.log)")
    return [json.loads(x) for x in (out/'rows.jsonl').read_text().splitlines()], out, False


def assess(p, c, resamples=20000, seed=20261001):
    mse = np.array([[x['mse'] for x in p], [x['mse'] for x in c]])
    util = np.array([[x['util'] for x in p], [x['util'] for x in c]])
    adj = mse*np.maximum(0.1, util)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, mse.shape[1], size=(resamples, mse.shape[1]))
    boot = lambda a: np.percentile(100*(a[1][idx].mean(1)/a[0][idx].mean(1) - 1), [2.5, 97.5]).tolist()
    pct = lambda a: float(100*(a[1].mean()/a[0].mean() - 1))
    per = 100*(adj[1]/adj[0] - 1)
    tot = lambda rows, k: sum(x[k] for x in rows)
    return dict(n=mse.shape[1], raw_pct=pct(mse), bill_pct=pct(util), adjusted_pct=pct(adj), raw_ci95_pct=boot(mse),
                adjusted_ci95_pct=boot(adj), adjusted_wins=int((adj[1] < adj[0]).sum()), raw_wins=int((mse[1] < mse[0]).sum()),
                max_regression_pct=float(per.max()), worst_rank=int(p[int(per.argmax())]['rank']),
                parent_mean_adjusted=float(adj[0].mean()), candidate_mean_adjusted=float(adj[1].mean()),
                parent_mean_mse=float(mse[0].mean()), candidate_mean_mse=float(mse[1].mean()),
                parent_mean_util=float(util[0].mean()), candidate_mean_util=float(util[1].mean()),
                cycles_pct=100*(tot(c, 'cycles')/tot(p, 'cycles') - 1),
                instructions_pct=100*(tot(c, 'instructions')/tot(p, 'instructions') - 1),
                cpu_seconds_pct=100*(tot(c, 'cpu_s')/tot(p, 'cpu_s') - 1),
                per_network=[dict(rank=a['rank'], raw_pct=100*(b['mse']/a['mse'] - 1), bill_pct=100*(b['util']/a['util'] - 1),
                                  adjusted_pct=float(q)) for a, b, q in zip(p, c, per)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate', type=Path)
    ap.add_argument('tag')
    ap.add_argument('--parent', type=Path, default=BANK)
    ap.add_argument('--ranks', default=None, help='comma-separated ranks; default: the set named by --set')
    ap.add_argument('--set', choices=('valid40', 'fresh40'), default='valid40', help='valid40 = the historical validation set; '
                    'fresh40 = 40 newer ranks nobody has trained or selected on (common.FRESH40)')
    ap.add_argument('--batch', type=int, default=10)
    ap.add_argument('--max-adjusted-pct', type=float, default=-1.0)
    ap.add_argument('--max-raw-pct', type=float, default=None)
    ap.add_argument('--ci-upper', type=float, default=0.0)
    ap.add_argument('--min-wins', type=int, default=28)
    ap.add_argument('--max-regression-pct', type=float, default=10.0)
    ap.add_argument('--no-gate', action='store_true')
    a = ap.parse_args()
    candidate, parent = a.candidate.resolve(), a.parent.resolve()
    out = run_dir(a.tag)
    ranks = [int(x) for x in a.ranks.split(',')] if a.ranks else (VALID40 if a.set == 'valid40' else FRESH40)
    batches = [ranks[i:i + a.batch] for i in range(0, len(ranks), a.batch)]
    rows, dirs, same_window = {'P': [], 'C': []}, {'P': [], 'C': []}, True
    for b, batch in enumerate(batches):
        for arm in ('PC' if b % 2 == 0 else 'CP'):
            got, where, cached = native(parent if arm == 'P' else candidate, batch, not a.no_gate)
            rows[arm] += got
            dirs[arm].append(str(where))
            same_window &= not cached
            print(f"  batch {b + 1}/{len(batches)} arm {arm}: {'stored' if cached else 'ran'} {where.name}: mean MSE "
                  f"{np.mean([x['mse'] for x in got]):.4e}, mean bill {np.mean([x['util'] for x in got]):.5f}", flush=True)
    m = assess(rows['P'], rows['C'])
    identical = all((Path(d)/f'pred_{r}.npy').read_bytes() == (Path(e)/f'pred_{r}.npy').read_bytes()
                    for d, e, batch in zip(dirs['P'], dirs['C'], batches) for r in batch)
    gates = dict(adjusted=m['adjusted_pct'] <= a.max_adjusted_pct, ci_upper=m['adjusted_ci95_pct'][1] < a.ci_upper,
                 wins=m['adjusted_wins'] >= a.min_wins, regression=m['max_regression_pct'] <= a.max_regression_pct)
    if a.max_raw_pct is not None:
        gates['raw'] = m['raw_pct'] <= a.max_raw_pct
    passed = all(gates.values())
    text = (f"{m['n']} networks ({'given ranks' if a.ranks else a.set}) vs {parent.parent.name}: pooled MSE {m['raw_pct']:+.3f}% "
            f"[{m['raw_ci95_pct'][0]:+.2f}, {m['raw_ci95_pct'][1]:+.2f}]"
            + (f" (limit <= {a.max_raw_pct:+.2f})" if a.max_raw_pct is not None else '')
            + f", bill {m['bill_pct']:+.3f}%, adjusted {m['adjusted_pct']:+.3f}% "
            f"[{m['adjusted_ci95_pct'][0]:+.2f}, {m['adjusted_ci95_pct'][1]:+.2f}] (limit <= {a.max_adjusted_pct:+.1f}, upper end < "
            f"{a.ci_upper:g}), wins {m['adjusted_wins']}/{m['n']} (limit >= {a.min_wins}), worst network {m['max_regression_pct']:+.2f}% "
            f"(rank {m['worst_rank']}, limit <= {a.max_regression_pct:+.0f}); predictions "
            f"{'byte-identical' if identical else 'differ'}; CPU instructions {m['instructions_pct']:+.2f}%, cycles "
            f"{m['cycles_pct']:+.2f}%{'' if same_window else ' (arms not in one window: stored runs reused)'}")
    result = dict(stage='B', candidate=str(candidate), candidate_sha256=sha(candidate), parent=str(parent),
                  parent_sha256=sha(parent), batches=batches, metrics=m, predictions_identical=identical, gates=gates,
                  passed=passed, runs=dirs, limits=dict(max_adjusted_pct=a.max_adjusted_pct, max_raw_pct=a.max_raw_pct,
                                                        ci_upper=a.ci_upper, min_wins=a.min_wins,
                                                        max_regression_pct=a.max_regression_pct))
    result['line'] = verdict('B native40', passed, text)
    write_json(out/'stage_b.json', result)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
