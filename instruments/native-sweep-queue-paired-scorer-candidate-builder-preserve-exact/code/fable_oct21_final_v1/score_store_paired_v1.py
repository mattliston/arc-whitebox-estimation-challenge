"""Paired store32 scorer: candidate run vs reference run on the store32 plan (truth_A/truth_B separate npz, key mc_post).
usage: score_store_paired_v1.py --root ROOT --plan PLAN --cand-run DIR --cand-key K --ref-run DIR --ref-key K --out OUT.json
Reports adjusted % (u*flops ratio) over all32, held25 and first7, wins, SE, raw/bill split, requests/puts/wall deltas.
"""
import argparse, hashlib, importlib.util, json, math, statistics, subprocess, sys
from pathlib import Path
import numpy as np
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def summ(rows, key='adjusted_pct'):
    v = [r[key] for r in rows]; return dict(n=len(v), mean=statistics.mean(v), se=statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None, lower=sum(x < 0 for x in v))
def main():
    ap = argparse.ArgumentParser()
    for n in ('root', 'plan', 'cand-run', 'cand-key', 'ref-run', 'ref-key', 'out'): ap.add_argument('--' + n, required=True)
    a = ap.parse_args(); root = Path(a.root)
    mark = root / 'pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1'
    sys.path[:0] = [str(mark), str(mark.parent / 'cloud_linux_capacity_v7')]
    spec = importlib.util.spec_from_file_location('grader', mark / 'grade_linux_markline_v1.py'); grader = importlib.util.module_from_spec(spec); spec.loader.exec_module(grader)
    plan = json.loads(Path(a.plan).read_text())
    cand, cs, _ = grader.collect(Path(a.cand_run), fresh=False); ref, rs, _ = grader.collect(Path(a.ref_run), fresh=False)
    assert cs == plan['candidates'][a.cand_key]['effective_sha256'] and rs == plan['candidates'][a.ref_key]['effective_sha256'], (cs, rs)
    ranks = sorted(set(cand) & set(ref)); rows = []
    for rk in ranks:
        ra, rb = plan['rows'][str(rk)]['truth_A'], plan['rows'][str(rk)]['truth_B']
        with np.load(root / ra['path'], allow_pickle=False) as z: A = z[ra['key']].astype(np.float64)
        with np.load(root / rb['path'], allow_pickle=False) as z: B = z[rb['key']].astype(np.float64)
        c, r = cand[rk]['prediction'], ref[rk]['prediction']
        u1 = float(np.mean((c[15] - A[15]) * (c[15] - B[15]))); u0 = float(np.mean((r[15] - A[15]) * (r[15] - B[15])))
        fr = cand[rk]['row']['flops'] / ref[rk]['row']['flops']
        rows.append(dict(rank=rk, raw_pct=100 * (u1 / u0 - 1), bill_pct=100 * (fr - 1), adjusted_pct=100 * (u1 / u0 * fr - 1),
                         requests_pct=100 * (cand[rk]['row']['requests'] / ref[rk]['row']['requests'] - 1), puts_pct=100 * (cand[rk]['row']['puts'] / ref[rk]['row']['puts'] - 1),
                         wall_pct=100 * (cand[rk]['row']['wall_s'] / ref[rk]['row']['wall_s'] - 1), residual_pct=100 * (cand[rk]['row']['residual_s'] / ref[rk]['row']['residual_s'] - 1), identical=bool(np.array_equal(c, r))))
    held = [r for r in rows if r['rank'] in set(plan['held25'])]; first = [r for r in rows if r['rank'] in set(plan['first7'])]
    out = dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(), candidate_sha256=cs, reference_sha256=rs, cand_run=a.cand_run, ref_run=a.ref_run, n=len(rows),
               all=summ(rows), held25=summ(held), first7=summ(first), raw=summ(rows, 'raw_pct'), bill=summ(rows, 'bill_pct'), requests=summ(rows, 'requests_pct'), puts=summ(rows, 'puts_pct'), wall=summ(rows, 'wall_pct'), residual=summ(rows, 'residual_pct'), identical=sum(r['identical'] for r in rows), rows=rows)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('n', 'all', 'held25', 'first7', 'raw', 'bill', 'requests', 'puts', 'wall', 'residual', 'identical')}))
if __name__ == '__main__': main()
