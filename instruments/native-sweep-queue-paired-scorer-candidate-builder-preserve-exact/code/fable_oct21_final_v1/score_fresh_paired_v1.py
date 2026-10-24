"""Lean paired fresh-cohort scorer: candidate run vs reference run on the same cold singleton plan rows.
usage: score_fresh_paired_v1.py --root ROOT --plan PLAN --cand-run DIR --cand-key K --ref-run DIR --ref-key K [--plan-ref PLAN_REF] --out OUT.json
Per seed: u = mean((pred-A)*(pred-B)) at the final layer (cross-half U statistic), adjusted ratio = (u_c/u_r)*(flops_c/flops_r).
"""
import argparse, hashlib, importlib.util, json, math, statistics, subprocess, sys
from pathlib import Path
import numpy as np
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap = argparse.ArgumentParser()
    for n in ('root', 'plan', 'cand-run', 'cand-key', 'ref-run', 'ref-key', 'out'): ap.add_argument('--' + n, required=True)
    ap.add_argument('--plan-ref', default=None)
    a = ap.parse_args(); root = Path(a.root)
    mark = root / 'pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1'
    sys.path[:0] = [str(mark), str(mark.parent / 'cloud_linux_capacity_v7')]
    spec = importlib.util.spec_from_file_location('grader', mark / 'grade_linux_markline_v1.py'); grader = importlib.util.module_from_spec(spec); spec.loader.exec_module(grader)
    plan = json.loads(Path(a.plan).read_text()); plan_ref = json.loads(Path(a.plan_ref).read_text()) if a.plan_ref else plan
    cand, cs, cm = grader.collect(Path(a.cand_run), fresh=False); ref, rs, rm = grader.collect(Path(a.ref_run), fresh=False)
    assert cs == plan['candidates'][a.cand_key]['effective_sha256'] and rs == plan_ref['candidates'][a.ref_key]['effective_sha256'], (cs, rs)
    seeds = [g[0] for g in plan['groups']]; assert set(cand) == set(ref) == set(seeds)
    rows = []
    for seed in seeds:
        ra, rb = plan['rows'][str(seed)]['truth_A'], plan['rows'][str(seed)]['truth_B']; assert ra['path'] == rb['path']
        tp = root / ra['path']; assert sha(tp) == ra['sha256']
        with np.load(tp, allow_pickle=False) as z:
            A, B = z[ra['key']].astype(np.float64), z[rb['key']].astype(np.float64); assert int(z['seed']) == seed
        c, r = cand[seed]['prediction'], ref[seed]['prediction']
        u1 = float(np.mean((c[15] - A[15]) * (c[15] - B[15]))); u0 = float(np.mean((r[15] - A[15]) * (r[15] - B[15])))
        fr = cand[seed]['row']['flops'] / ref[seed]['row']['flops']
        rows.append(dict(rank=seed, raw_pct=100 * (u1 / u0 - 1), bill_pct=100 * (fr - 1), adjusted_pct=100 * (u1 / u0 * fr - 1), u1=u1, u0=u0,
                         flops=cand[seed]['row']['flops'], ref_flops=ref[seed]['row']['flops'], identical=bool(np.array_equal(c, r))))
    s = dict(n=len(rows), lower=sum(r['adjusted_pct'] < 0 for r in rows))
    for k in ('raw_pct', 'bill_pct', 'adjusted_pct'):
        v = [r[k] for r in rows]; s[k] = statistics.mean(v); s[k + '_se'] = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None
    out = dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(), candidate_sha256=cs, reference_sha256=rs, cand_run=a.cand_run, ref_run=a.ref_run, summary=s, rows=rows)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(dict(summary=s, identical=[r['identical'] for r in rows])))
if __name__ == '__main__': main()
