#!/usr/bin/env python3
"""Prospective strictly-negative grading; no partial scientific data are loaded."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
O=HERE.parents[2]
FRESH=list(range(7100632,7100648))
CONF=list(range(7100592,7100608))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
POLICY_FOLDER = O / 'lead/runs/gain_floor_policy_20261013_v1'
ADDENDUM_SHA256 = 'd91844c5eb99fba8949233b364a29894b3ee2f1213756fb5a4e1e255bfbcbb5f'

def load_policy():
    assert sha(POLICY_FOLDER / 'PROSPECTIVE_ADDENDUM_v1.json') == ADDENDUM_SHA256
    spec = json.loads((POLICY_FOLDER / 'ACTIVATION_v1.json').read_text())
    assert spec['status'] == 'root_registered_prospectively'
    assert spec['addendum_sha256'] == ADDENDUM_SHA256
    assert set(spec['eligible']) <= {'T151', 'T152', 'metric', 'weighted'}
    for name, expected in spec['legacy_decision_sha256'].items():
        path = O / name
        assert (sha(path) if path.exists() else None) == expected, 'Legacy decision changed after registration: ' + name
    for item in spec['eligible']:
        assert spec['complete_at_registration'][item] is False, 'Completed experiments require independent continuation'
    for k in (151, 152):
        name = f'physicist/t{k}_confirm.txt'
        if f'T{k}' in spec['eligible']:
            assert name in spec['legacy_decision_sha256'] and spec['legacy_decision_sha256'][name] is None
    return spec

def policy():
    spec=load_policy()
    assert 'metric' in spec['eligible'], 'Root has not registered metric prospectively'
    return spec

def path_for(run,arm,seed):
    return HERE/run/'results'/f'k4metric_{arm}_r{seed}.json'

def read(run,arm,seed):
    p=path_for(run,arm,seed)
    assert not p.with_suffix('.error').exists(), 'Worker error: '+str(p)
    if not p.exists():return None
    row=json.loads(p.read_text())
    assert row['rank']==seed and row['id']==p.stem
    assert len(row['mse_layers'])==16 and all(math.isfinite(x) and x>0 for x in row['mse_layers'])
    assert math.isfinite(row['mse']) and row['mse']>0
    return row

def coverage(arms,confirmation=False):
    seeds=CONF if confirmation else FRESH
    run='run_confirm' if confirmation else 'run_screen'
    counts={arm:sum(path_for(run,arm,s).exists() for s in seeds) for arm in arms}
    refs=[O/'physicist/t141_k4l_ext/results'/f'ph141_r{s}_c.json' for s in seeds] if confirmation else [path_for(run,'z',s) for s in seeds]
    return dict(complete=all(n==16 for n in counts.values()) and all(p.exists() for p in refs),counts=counts,references=sum(p.exists() for p in refs))

def changes(arm,confirmation=False):
    assert coverage([arm],confirmation)['complete'], 'No partial scientific data permitted'
    out=[]
    for seed in (CONF if confirmation else FRESH):
        a=read('run_confirm' if confirmation else 'run_screen',arm,seed)
        b=json.loads((O/'physicist/t141_k4l_ext/results'/f'ph141_r{seed}_c.json').read_text()) if confirmation else read('run_screen','z',seed)
        assert a['rank']==b['rank']==seed and math.isfinite(b['mse']) and b['mse']>0
        value=math.log(a['mse']/b['mse']);assert math.isfinite(value);out.append(value)
    return out

def stats(values):
    return dict(n=len(values),mean_log_pct=100*statistics.mean(values),se_pct=100*statistics.stdev(values)/math.sqrt(len(values)),lower=sum(x<0 for x in values))

def smoke():
    a=read('run_smoke','z',7100584)
    b=json.loads((O/'physicist/t141_k4l_ext/results/ph141_r7100584_c.json').read_text())
    checks={key:a[key]==b[key] for key in ('mse','mse_layers','final_prediction')} if a else {}
    return dict(complete=a is not None,checks=checks,bitexact=bool(checks) and all(checks.values()))

def screen():
    policy()
    present=coverage(('h','f'))
    if not present['complete']:return dict(**present,winner=None)
    assert smoke()['bitexact'], 'Control must reproduce C'
    arms={arm:stats(changes(arm)) for arm in ('h','f')}
    winner=min(('h','f'),key=lambda arm:(arms[arm]['mean_log_pct'],arm!='h'))
    if not arms[winner]['mean_log_pct']<0:winner=None
    return dict(complete=True,smoke_bitexact=True,arms=arms,winner=winner,criterion='strict_negative_no_minimum_gain')

def confirm():
    decision=screen();winner=decision['winner']
    assert decision['complete'] and winner, 'No eligible complete screen winner'
    present=coverage([winner],True)
    if not present['complete']:return dict(**present,winner=winner)
    fresh,oos=changes(winner),changes(winner,True)
    full,separate=stats(fresh+oos),stats(oos)
    return dict(complete=True,winner=winner,all32=full,confirmation=separate,passed=full['mean_log_pct']<0 and separate['mean_log_pct']<0)

if __name__=='__main__':
    mode=sys.argv[1];assert mode in ('smoke','screen','confirm')
    print(json.dumps(globals()[mode](),indent=2,allow_nan=False))
