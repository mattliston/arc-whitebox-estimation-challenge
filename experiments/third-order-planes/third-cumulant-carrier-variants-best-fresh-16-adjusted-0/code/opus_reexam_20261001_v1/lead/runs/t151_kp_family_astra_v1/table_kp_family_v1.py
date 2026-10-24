"""Read-only full-family table; requires explicit root clarification activation.

Never imports the original table (its top level reads every available arm).
No result is opened until all16 fixed records for ('kp',) and C exist.
"""
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
O=HERE.parents[2]
P=O/'physicist'
R=P/'t151_carrier'
FRESH=list(range(7100608,7100624))
CONF=list(range(7100592,7100608))
ARMS=('kp',)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def activation():
    spec=json.loads((HERE/'ACTIVATION_v1.json').read_text())
    assert spec['status']=='root_registered_before_kp_result_read'
    assert spec['clarification_sha256']==sha(HERE/'SCHEDULING_CLARIFICATION_DRAFT_v1.json')
    assert spec['manifest_sha256']==sha(HERE/'SOURCE_MANIFEST_v1.json')
    assert spec['kp_scientific_results_unread_at_registration'] is True
    cfg=json.loads((HERE/'SCHEDULING_CLARIFICATION_DRAFT_v1.json').read_text())
    for rel,expected in cfg['sources'].items():assert sha(O/rel)==expected,rel
    for rel,expected in json.loads((HERE/'SOURCE_MANIFEST_v1.json').read_text())['files'].items():assert sha(HERE/rel)==expected,rel
    policy=json.loads((O/'lead/runs/gain_floor_policy_20261013_v1/ACTIVATION_v1.json').read_text())
    assert policy['status']=='root_registered_prospectively' and 'T151' in policy['eligible']
    assert not (P/'t151_confirm.txt').exists(),'Historical decision remains absent'
    return spec


def required_screen():
    return [R/'results'/f'ph151{x}_r{s}.json' for x in ('_ref',)+ARMS for s in FRESH]


def coverage(paths):
    errors=[str(p.with_suffix('.error')) for p in paths if p.with_suffix('.error').exists()]
    return dict(complete=all(p.exists() for p in paths) and not errors,
                records=sum(p.exists() for p in paths),required=len(paths),errors=errors)


def read(path,seed):
    d=json.loads(path.read_text())
    assert d['rank']==seed and d['id']==path.stem and d['mode']=='eval' and d['threads']==2
    assert len(d['mse_layers'])==16 and all(math.isfinite(x) and x>0 for x in d['mse_layers'])
    assert math.isfinite(d['mse']) and d['mse']>0
    return d


def stats(values):
    # Match v10's vectorized numpy mean/log arithmetic, including exact ties.
    mean=float(np.asarray(values).mean())
    return dict(n=len(values),mean_log=mean,mean_log_pct=100*mean,
                se_pct=float(100*statistics.stdev(values)/math.sqrt(len(values))),lower=int(sum(x<0 for x in values)))


def choose(means):
    assert set(means)==set(ARMS)
    winner=min(ARMS,key=lambda a:means[a])
    return winner if means[winner]<0 else None


def changes(arm,confirmation=False):
    seeds=CONF if confirmation else FRESH
    pairs=[(s,R/'results'/f'ph151{"x" if confirmation else ""}{arm}_r{s}.json',
            P/'t141_k4l_ext/results'/f'ph141_r{s}_c.json' if confirmation else R/'results'/f'ph151_ref_r{s}.json') for s in seeds]
    assert coverage([p for _,a,b in pairs for p in (a,b)])['complete']
    values=np.asarray([(read(a,s)['mse_layers'][-1],read(b,s)['mse_layers'][-1]) for s,a,b in pairs])
    return list(np.log(values[:,0]/values[:,1]))


def screen():
    activation()
    c=coverage(required_screen())
    if not c['complete']:return dict(**c,family=list(ARMS),winner=None)
    results={a:stats(changes(a)) for a in ARMS}
    winner=choose({a:results[a]['mean_log'] for a in ARMS})
    # The eventual independent all-family result must agree exactly.
    full=P/'t151_confirm_astra_v10.txt'
    if full.exists():
        selected=[a for a in full.read_text().split() if a in ARMS]
        assert selected==([winner] if winner else []),'Later full-T151 decision disagrees'
    priority=('before_t152' if winner and results[winner]['mean_log']<=-.0015 else 'normal')
    return dict(complete=True,family=list(ARMS),fresh=FRESH,confirmation=CONF,
                results=results,winner=winner,priority=priority,
                rule='per-dial minimum, original kp tie order, complete16 each, strictly negative',
                scientific_inputs={str(p.relative_to(O)):sha(p) for p in required_screen()})


def confirm():
    d=screen();assert d['complete'] and d['winner'],'No eligible full-family screen winner'
    w=d['winner'];paths=[p for s in CONF for p in (R/'results'/f'ph151x{w}_r{s}.json',P/'t141_k4l_ext/results'/f'ph141_r{s}_c.json')]
    c=coverage(paths)
    if not c['complete']:return dict(**c,winner=w)
    fresh,oos=changes(w),changes(w,True);full,separate=stats(fresh+oos),stats(oos)
    # The supervisor uses equal-sized block means for its all32 gate.
    combined_mean=(d['results'][w]['mean_log']+separate['mean_log'])/2
    return dict(complete=True,winner=w,all32=full,confirmation=separate,
                supervisor_all32_mean_log=combined_mean,
                passed=combined_mean<0 and separate['mean_log']<0)


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('screen','confirm')
    print(json.dumps(globals()[sys.argv[1]](),indent=2,allow_nan=False))
