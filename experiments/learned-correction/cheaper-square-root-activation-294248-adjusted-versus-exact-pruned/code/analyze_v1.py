"""All four predeclared seeds, paired to saved bank337393 and saved EH14."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np
from run_v1 import checked_result,verify,utc

D = Path(__file__).resolve().parent


def load_module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--if-complete',action='store_true')
    args = p.parse_args()
    m = json.loads((D/'MANIFEST_v1.json').read_text())
    verify(m)
    paths = [D/'runs'/f'drop14_seed{seed}' for seed in m['seeds']]
    if not all((p/'GUARD_v1.json').exists() and (p/'run.log').exists()
               and 'EVAL_DONE' in (p/'run.log').read_text().splitlines() for p in paths):
        if args.if_complete:
            print(json.dumps(dict(status='COHORT_PENDING')))
            return 0
        raise AssertionError('All four completed guards required.')
    pd,rd = Path(m['parent_dir']),Path(m['reference_dir'])
    pc = load_module('drop14_parent_controller',pd/'run_v1.py')
    rc = load_module('drop14_bank_controller',rd/'run_partition_v2.py')
    pm,rm = json.loads((pd/'MANIFEST_v1.json').read_text()),json.loads((rd/'MANIFEST_v2.json').read_text())
    pc.verify(pm);rc.verify(rm)
    assert pm['settings'] == m['settings'] and pm['evaluator_sha256'] == m['evaluator_sha256']
    assert rm['evaluator_sha256'] == m['evaluator_sha256']
    rows = []
    for seed,path in zip(m['seeds'],paths):
        cand = checked_result(path,seed,m)
        parent = pc.checked_result(pd/'runs'/f'eh14_seed{seed}',seed,'eh14',pm)
        bp = rd/'runs'/f'bank_seed{seed}'
        rc.checked_result(bp,seed,'bank',rm)
        bank = json.loads(next((bp/'predictions').glob('*.json')).read_text())
        truth = next(t for t in m['truths'] if t['seed']==seed)
        with np.load(truth['path'],allow_pickle=False) as z:
            A,B = z['mc_post_A'][15].astype('float64'),z['mc_post_B'][15].astype('float64')
        for record in (cand,parent,bank):
            pred = Path(record['prediction_path'])
            assert hashlib.sha256(pred.read_bytes()).hexdigest() == record['prediction_sha256']
            y = np.load(pred,allow_pickle=False)
            assert y.shape==(16,1024) and np.isfinite(y).all()
            u = float(np.mean((y[15]-A)*(y[15]-B)))
            assert u==record['cross_half_U'] and u>0
        uc,up,ub = (r['cross_half_U'] for r in (cand,parent,bank))
        fc,fp,fb = (r['flops'] for r in (cand,parent,bank))
        rows.append(dict(seed=seed,candidate_U=uc,parent_U=up,bank_U=ub,
                         candidate_bill=fc,parent_bill=fp,bank_bill=fb,
                         raw_vs_337393_pct=100*(uc/ub-1),bill_vs_337393_pct=100*(fc/fb-1),
                         adjusted_vs_337393_pct=100*(uc*fc/(ub*fb)-1),
                         raw_vs_eh14_pct=100*(uc/up-1),bill_vs_eh14_pct=100*(fc/fp-1),
                         adjusted_vs_eh14_pct=100*(uc*fc/(up*fp)-1),
                         static_release_correction_vs_337393_pct=100*(uc*(fc-m['measured_reset_extra_flops'])/(ub*fb)-1)))
    fields = [k for k in rows[0] if k.endswith('_pct')]
    summaries = {}
    for key in fields:
        xs = [r[key] for r in rows]
        summaries[key] = dict(n=4,mean_pct=statistics.mean(xs),se_pct=statistics.stdev(xs)/2,lower=sum(x<0 for x in xs))
    inc = summaries['adjusted_vs_eh14_pct']
    report = dict(utc=utc(),status='PAIRED_FRESH4_NOT_CURRENT32_QUALIFICATION',reference_submission_id=337393,
                  candidate_sha256=m['candidate_sha256'],parent_sha256=m['parent_sha256'],seeds=m['seeds'],
                  rows=rows,summaries=summaries,
                  predeclared_next_stage_hint=inc['mean_pct']<0 and (inc['mean_pct']<-.02 or inc['lower']>=3),
                  limitations=m['limitations'])
    tmp = D/f'RESULT_FRESH4_v1.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    tmp.replace(D/'RESULT_FRESH4_v1.json')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
