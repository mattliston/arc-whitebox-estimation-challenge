"""Four predeclared paired seeds, actual bank and current-parent saved outputs."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

import numpy as np
from run_v1 import checked_result, verify

D = Path(__file__).resolve().parent


def stats(xs):
    return dict(n=len(xs), mean_pct=statistics.mean(xs), se_pct=statistics.stdev(xs)/math.sqrt(len(xs)),
                lower=sum(x < 0 for x in xs))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--if-complete', action='store_true')
    args = p.parse_args()
    m = json.loads((D/'MANIFEST_v1.json').read_text())
    verify(m)
    refdir = Path(m['reference_dir'])
    reference = json.loads((refdir/'MANIFEST_v2.json').read_text())
    spec = importlib.util.spec_from_file_location('edge_coadapt_reference_controller', refdir/'run_partition_v2.py')
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    rc.verify(reference)
    assert reference['settings'][0] == m['settings'][0]
    assert reference['evaluator_sha256'] == m['evaluator_sha256']
    assert reference['candidate_sha256'] == m['parent_sha256']
    def ready(path):
        return (path/'GUARD_v1.json').exists() and (path/'run.log').exists() and 'EVAL_DONE' in (path/'run.log').read_text().splitlines()
    missing = [str(refdir/'runs'/f'{role}_seed{seed}') for seed in m['seeds'] for role in ('bank', 'screen')
               if not ready(refdir/'runs'/f'{role}_seed{seed}')]
    if missing:
        if args.if_complete:
            print(json.dumps(dict(status='REFERENCE_COHORT_PENDING', missing=missing)), flush=True)
            return 0
        raise AssertionError(missing)
    saved = {}
    for seed in m['seeds']:
        for role in ('bank', 'screen'):
            path = refdir/'runs'/f'{role}_seed{seed}'
            rc.checked_result(path, seed, role, reference)
            records = [json.loads(p.read_text()) for p in (path/'predictions').glob('*.json')]
            record = next(r for r in records if r['setting_index'] == 0)
            saved[(seed, role)] = record
    result = []
    pending = []
    reset_charge = m['measured_reset_extra_flops']
    for role in ('eh14', 'eh15'):
        paths = [D/'runs'/f'{role}_seed{seed}' for seed in m['seeds']]
        if not all(ready(path) for path in paths):
            pending.append(role)
            continue
        rows = []
        for seed, path in zip(m['seeds'], paths):
            cand = checked_result(path, seed, role, m)
            bank, parent = saved[(seed, 'bank')], saved[(seed, 'screen')]
            truth = next(t for t in m['truths'] if t['seed'] == seed)
            with np.load(truth['path'], allow_pickle=False) as z:
                A, B = z['mc_post_A'][15].astype('float64'), z['mc_post_B'][15].astype('float64')
            for record in (cand, bank, parent):
                pred = Path(record['prediction_path'])
                assert hashlib.sha256(pred.read_bytes()).hexdigest() == record['prediction_sha256']
                y = np.load(pred, allow_pickle=False)
                assert y.shape == (16, 1024) and np.isfinite(y).all()
                u = float(np.mean((y[15]-A)*(y[15]-B)))
                assert u == record['cross_half_U'] and u > 0
            uc, ub, up = (r['cross_half_U'] for r in (cand, bank, parent))
            fc, fb, fp = (r['flops'] for r in (cand, bank, parent))
            rows.append(dict(seed=seed, candidate_U=uc, bank_U=ub, parent_U=up,
                             candidate_bill=fc, bank_bill=fb, parent_bill=fp,
                             raw_vs_337393_pct=100*(uc/ub-1), bill_vs_337393_pct=100*(fc/fb-1),
                             adjusted_vs_337393_pct=100*(uc*fc/(ub*fb)-1),
                             raw_vs_parent_pct=100*(uc/up-1), bill_vs_parent_pct=100*(fc/fp-1),
                             adjusted_vs_parent_pct=100*(uc*fc/(up*fp)-1),
                             static_release_bill_correction_vs_337393_pct=100*(uc*(fc-reset_charge)/(ub*fb)-1),
                             candidate_metadata_sha256=hashlib.sha256(json.dumps(cand,sort_keys=True).encode()).hexdigest()))
        fields = ('raw_vs_337393_pct', 'bill_vs_337393_pct', 'adjusted_vs_337393_pct',
                  'raw_vs_parent_pct', 'bill_vs_parent_pct', 'adjusted_vs_parent_pct',
                  'static_release_bill_correction_vs_337393_pct')
        result.append(dict(variant=role, candidate_sha256=m['variants'][role]['sha256'], rows=rows,
                           summaries={field:stats([r[field] for r in rows]) for field in fields}))
    if not result:
        if args.if_complete:
            print(json.dumps(dict(status='EDGE_COHORT_PENDING', variants=pending)), flush=True)
            return 0
        raise AssertionError(pending)
    report = dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
                  status='PAIRED_FRESH4_NOT_CURRENT32_QUALIFICATION', reference_submission_id=337393,
                  seeds=m['seeds'], variants=result, pending_variants=pending,
                  reset_extra_flops_measured=reset_charge,
                  interpretation='Primary ratios use actual scored bills. Parent comparison uses identical cache-reset settings. Separately labelled release correction removes only the measured GQ cache construction charge; no MSE or runtime adjustment.',
                  limitations=m['limitations'])
    temp = D/f'RESULT_FRESH4_v1.{os.getpid()}.tmp'
    temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temp.replace(D/'RESULT_FRESH4_v1.json')
    print(json.dumps({**report, 'variants':[{k:v for k,v in r.items() if k != 'rows'} for r in result]}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
