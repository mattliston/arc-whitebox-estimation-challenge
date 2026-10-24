"""Prospective paired-log grading after fixed positive bill; complete pools only."""
import json
import hashlib
import math
from pathlib import Path
import statistics
import sys
HERE=Path(__file__).resolve().parent
O=HERE.parents[2]
BILL_EXTRA=1261596
BILL_BASE=257354920614
BILL_FRACTION=BILL_EXTRA/BILL_BASE
BILL_LOG=math.log1p(BILL_FRACTION)
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

def prereg():
    policy=load_policy()
    assert 'weighted' in policy['eligible'], 'Root has not registered weighted prospectively'
    d=json.loads((HERE/'PREREG_v2.json').read_text())
    assert d['status']=='registered_by_root_before_evaluations'
    assert d['arms']=={'z':0.0,'q':0.25,'h':0.5} and d['floor']==1e-30
    assert d['screen_seeds']==list(range(7101096,7101112))
    assert d['confirmation_seeds']==list(range(7100592,7100608))
    assert d['allocation_receipt_sha256']==sha(HERE/'ALLOCATION_RECEIPT_v1.json')
    assert d['harness_manifest_sha256']==sha(HERE/'harness_manifest_v3.json')
    assert d['gain_policy_sha256']==ADDENDUM_SHA256
    assert d['criterion']=='strict_negative_paired_log_after_fixed_bill'
    assert d['bill_extra_flops']==BILL_EXTRA and d['bill_baseline_flops']==BILL_BASE
    return d

def coverage(arms,confirmation=False):
    config=prereg();seeds=config['confirmation_seeds' if confirmation else 'screen_seeds']
    run='run_confirm' if confirmation else 'run_screen'
    counts={arm:sum((HERE/run/'results'/f'wfit_{arm}_r{s}.json').exists() for s in seeds) for arm in arms}
    refs=[O/'physicist/t141_k4l_ext/results'/f'ph141_r{s}_c.json' for s in seeds] if confirmation else [HERE/run/'results'/f'wfit_z_r{s}.json' for s in seeds]
    return dict(complete=all(n==16 for n in counts.values()) and all(p.exists() for p in refs),counts=counts,references=sum(p.exists() for p in refs))

def read(run, arm, seed):
    path = HERE/run/'results'/f'wfit_{arm}_r{seed}.json'
    if path.with_suffix('.error').exists():
        raise RuntimeError('Worker error: '+str(path.with_suffix('.error')))
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    assert row['rank'] == seed and row['id'] == path.stem
    assert len(row['mse_layers']) == 16 and all(math.isfinite(x) and x > 0 for x in row['mse_layers'])
    assert math.isfinite(row['mse']) and row['mse'] > 0
    return row


def smoke():
    a = read('run_smoke','z',7100584)
    b = json.loads((O/'physicist/t141_k4l_ext/results/ph141_r7100584_c.json').read_text())
    checks = {key: a[key] == b[key] for key in ('mse','mse_layers','final_prediction')} if a else {}
    return dict(complete=a is not None, checks=checks, bitexact=bool(checks) and all(checks.values()))


def changes(arm, confirmation=False):
    config = prereg()
    assert coverage([arm],confirmation)['complete'], 'No partial scientific data permitted'
    seeds = config['confirmation_seeds' if confirmation else 'screen_seeds']
    out = []
    for seed in seeds:
        a = read('run_confirm' if confirmation else 'run_screen',arm,seed)
        if confirmation:
            path = O/'physicist/t141_k4l_ext/results'/f'ph141_r{seed}_c.json'
            b = json.loads(path.read_text()) if path.exists() else None
        else:
            b = read('run_screen','z',seed)
        if a is None or b is None:
            continue
        assert b['rank'] == seed and math.isfinite(b['mse']) and b['mse'] > 0
        out.append(math.log(a['mse_layers'][-1]/b['mse_layers'][-1]))
    return out


def stats(values):
    return dict(n=len(values), mean_log_pct=100*statistics.mean(values),
                se_pct=100*statistics.stdev(values)/math.sqrt(len(values)), lower=sum(x<0 for x in values))


def net_stats(values):
    row=stats(values)
    row.update(fixed_bill_fraction=BILL_FRACTION,mean_net_log_pct=100*(statistics.mean(values)+BILL_LOG))
    return row

def screen():
    prereg()
    present=coverage(('q','h'))
    if not present['complete']:return dict(**present,winner=None)
    assert smoke()['bitexact'], 'Control did not reproduce C'
    arms={arm:net_stats(changes(arm)) for arm in ('q','h')}
    winner=min(('q','h'),key=lambda arm:(arms[arm]['mean_net_log_pct'],arm!='q'))
    if not arms[winner]['mean_net_log_pct']<0:winner=None
    return dict(complete=True,arms=arms,winner=winner,criterion='strict_negative_paired_log_after_fixed_bill')

def confirm():
    decision=screen();winner=decision['winner']
    assert decision['complete'] and winner, 'No complete-screen winner'
    present=coverage([winner],True)
    if not present['complete']:return dict(**present,winner=winner)
    fresh,oos=changes(winner),changes(winner,True)
    full,separate=net_stats(fresh+oos),stats(oos)
    return dict(complete=True,winner=winner,all32=full,confirmation=separate,
                passed=full['mean_log_pct']<0 and full['mean_net_log_pct']<0 and separate['mean_log_pct']<0)

if __name__=='__main__':
    mode=sys.argv[1];assert mode in ('smoke','screen','confirm')
    print(json.dumps(globals()[mode](),indent=2,allow_nan=False))
