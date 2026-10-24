"""Actual paired fresh-bank ratios only; no old-mean additivity or release authority."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

import numpy as np

D = Path(__file__).resolve().parent


def summary(values):
    return dict(mean_pct=statistics.mean(values), se_pct=statistics.stdev(values)/math.sqrt(len(values)),
                lower=sum(v < 0 for v in values), n=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--if-complete', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((D/'MANIFEST_v2.json').read_text())
    paths = [(seed, role, D/'runs'/('%s_seed%d' % (role, seed)))
             for seed in manifest['seeds'] for role in ('bank', 'screen')]
    absent = [str(p) for _, _, p in paths if not (p/'GUARD_v1.json').exists() or not (p/'run.log').exists()
              or 'EVAL_DONE' not in (p/'run.log').read_text().splitlines()]
    if absent:
        if args.if_complete:
            print(json.dumps(dict(status='PAIRED_FRESH_PENDING', missing=absent)), flush=True)
            return 0
        raise AssertionError(absent)
    from run_partition_v2 import checked_result, verify
    verify(manifest)
    records = {}
    for seed, role, path in paths:
        checked_result(path, seed, role, manifest)
        records[(seed, role)] = [json.loads(p.read_text()) for p in sorted((path/'predictions').glob('*.json'))]
    paired = [[] for _ in manifest['settings']]
    for seed in manifest['seeds']:
        truth = next(t for t in manifest['truths'] if t['seed'] == seed)
        with np.load(truth['path'], allow_pickle=False) as z:
            ta, tb = z['mc_post_A'][15].astype('float64'), z['mc_post_B'][15].astype('float64')
        for role in ('bank', 'screen'):
            for meta in records[(seed, role)]:
                payload = Path(meta['prediction_path']).read_bytes()
                assert hashlib.sha256(payload).hexdigest() == meta['prediction_sha256']
                prediction = np.load(meta['prediction_path'], allow_pickle=False)
                assert prediction.shape == (16, 1024) and np.isfinite(prediction).all()
                u = float(np.mean((prediction[15]-ta)*(prediction[15]-tb)))
                assert u == meta['cross_half_U'] and u > 0
        bank = records[(seed, 'bank')][0]
        default = records[(seed, 'screen')][0]
        for index, meta in enumerate(records[(seed, 'screen')]):
            u_ratio = meta['cross_half_U']/bank['cross_half_U']
            bill_ratio = meta['flops']/bank['flops']
            own_ratio = meta['cross_half_U']*meta['flops']/(default['cross_half_U']*default['flops'])
            paired[index].append(dict(seed=seed, setting_index=index, setting=meta['setting'],
                                     bank_U=bank['cross_half_U'], candidate_U=meta['cross_half_U'],
                                     bank_bill=bank['flops'], candidate_bill=meta['flops'],
                                     raw_pct=100*(u_ratio-1), bill_pct=100*(bill_ratio-1),
                                     adjusted_vs_337393_pct=100*(u_ratio*bill_ratio-1),
                                     adjusted_vs_same_screen_default_pct=100*(own_ratio-1)))
    settings = []
    for index, rows in enumerate(paired):
        settings.append(dict(index=index, label=manifest['labels'][index], setting=manifest['settings'][index],
                             adjusted_vs_337393=summary([r['adjusted_vs_337393_pct'] for r in rows]),
                             raw_vs_337393=summary([r['raw_pct'] for r in rows]),
                             bill_vs_337393=summary([r['bill_pct'] for r in rows]),
                             incremental_vs_screen_default=summary([r['adjusted_vs_same_screen_default_pct'] for r in rows]),
                             rows=rows))
    result = dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(),
                  status='PAIRED_FRESH4_COMPLETE_NOT_CURRENT32_QUALIFICATION',
                  reference_submission_id=337393, reference_sha256=manifest['bank_sha256'],
                  candidate_sha256=manifest['candidate_sha256'], seeds=manifest['seeds'], settings=settings,
                  limitations=manifest['limitations'])
    temp = D/('RESULT_FRESH4_v1.%d.tmp' % os.getpid())
    temp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temp.replace(D/'RESULT_FRESH4_v1.json')
    print(json.dumps(dict(status=result['status'], reference=337393,
                         settings=[{k:v for k,v in r.items() if k not in ('rows','setting')} for r in settings]), indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
