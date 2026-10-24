"""Prepared, unlaunched exact-code extension. ONE numerical process at a time.
Run from a managed foreground exec session only after root assigns one slot:
  .venv/bin/python -B extend_astra_compute_v1.py --run-authorized NEW_RUN_NAME
Fresh output directory, no overwrites, no submission, no decision-file writes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

D = Path(__file__).resolve().parent
O = D.parent
P2 = O.parent
REPO = P2.parents[3]
PY = REPO / '.venv/bin/python'
RT = P2 / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
PARENT = O / 'engineer/a62_modules/s49/s54k1tw_x/candidate.py'
ARM = D / 'astra_compute_v1_arm/candidate.py'
PINS = {
    PARENT: 'db2d50756c5f34438e3c49cabfce272ebdfa4337d248d78fb775e211e2641563',
    ARM: '2aaf7aa0fd8c66ba63b895b2742a4c7640ee2c50835cd689ac9dce1f581108cb',
    D / 'fragment_astra_compute_v1.py': '2ea4441025f1edc350f863abff37e150e48972ac02db2f96f3973abc9385cf4b',
}
SEEDS = (7100137, 7100600, 7100601)

def stamp():
    return subprocess.check_output(['/bin/date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()

def exclusive(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def event(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')

def team(message):
    with (O / 'TEAM_LOG.md').open('a') as f:
        f.write('[astra] ' + stamp() + ' ' + message + '\n')

def resources():
    pressure = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=15)
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', pressure).group(1))
    text = subprocess.check_output(['/bin/ps', '-axo', 'pid=,ppid=,rss=,comm='], text=True, timeout=5)
    workers = []
    for line in text.splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4 and int(fields[2]) > 1048576 and 'python' in Path(fields[3]).name.lower():
            workers.append(dict(pid=int(fields[0]), ppid=int(fields[1]), rss_kib=int(fields[2])))
    # Truth producer gets a conservative extra slot even when below the RSS threshold.
    return dict(stamp=stamp(), free_percent=free, numerical_workers=workers,
                numerical_count=len(workers), truth_reserve=1, counted=len(workers)+1)

def run_native(home, seed, name, candidate, env):
    before = resources()
    event(home / 'resources.jsonl', dict(stage='before', seed=seed, arm=name, **before))
    if before['free_percent'] < 50 or before['counted'] >= 10:
        raise RuntimeError('No authorized free slot or memory below 50%; nothing launched')
    stem = home / ('native_%d_%s' % (seed, name))
    result = Path(str(stem) + '.json')
    log = Path(str(stem) + '.log')
    guard = Path(str(stem) + '_guard.json')
    assert all(not p.exists() for p in (result, log, guard, Path(str(stem)+'_pred.npz')))
    command = [str(PY), '-B', str(O/'release/guard_footprint_v3.py'),
               '--record', str(guard), '--seconds', '8100', '--rss-backstop', '7500000000',
               str(RT), '-B', str(D/'run_cert_v2.py'), str(candidate), str(seed), str(result)]
    with log.open('x') as stream:
        child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        owner = dict(stamp=stamp(), seed=seed, arm=name, owned_guard_pid=child.pid, command=command)
        event(home/'ownership.jsonl', owner)
        print(json.dumps(owner), flush=True)
        previous_print = time.monotonic()
        try:
            while child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    current = resources()
                    event(home/'resources.jsonl', dict(stage='running', seed=seed, arm=name, **current))
                    if current['free_percent'] < 50 or current['counted'] > 10:
                        raise RuntimeError('Resource limit while running; stop only this owned guard')
                    if time.monotonic()-previous_print >= 30:
                        print(json.dumps(dict(seed=seed, arm=name, **current)), flush=True)
                        previous_print = time.monotonic()
        except BaseException:
            # guard catches KeyboardInterrupt and its finally retires only its own child group.
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                child.wait(timeout=30)
            raise
    g = json.loads(guard.read_text())
    assert child.returncode == 0 and g['completed'] and g['exit_code'] == 0
    assert not g.get('error') and not g['remaining_rows'] and g['whole_group_retired']
    return result, guard

def compare(home, seed, baseline, arm, guards):
    import numpy as np
    bd, ad = (json.loads(p.read_text()) for p in (baseline, arm))
    assert len(bd['rows']) == len(ad['rows']) == 1
    assert bd['sha256'] == PINS[PARENT] and ad['sha256'] == PINS[ARM]
    b, a = bd['rows'][0], ad['rows'][0]
    bp = np.load(str(baseline).replace('.json', '_pred.npz'))[str(seed)]
    ap = np.load(str(arm).replace('.json', '_pred.npz'))[str(seed)]
    assert bp.shape == ap.shape == (16, 1024)
    exact = [bp[i].tobytes() == ap[i].tobytes() for i in range(16)]
    stats = a['stats']['_ASTRA_COMPUTE_STATS']
    credit = b['flops'] - a['flops']
    checks = dict(predictions_16_exact=all(exact), mse_16_exact=b['mse_layers']==a['mse_layers'],
                  final_mse_exact=b['mse_final']==a['mse_final'], pred_sha_exact=b['pred_sha']==a['pred_sha'],
                  seed_exact=b['seed']==a['seed']==seed,
                  bill_counter_exact=credit==stats['saved_copy_elements'], positive_credit=credit>0,
                  predictions_finite=bool(np.isfinite(bp).all() and np.isfinite(ap).all()))
    report = dict(seed=seed, pass_all=all(checks.values()), checks=checks,
                  predictions_by_layer=exact, bill_credit=credit,
                  bill_change_pct=100*(a['flops']/b['flops']-1), request_delta=a['ops']-b['ops'],
                  extra_latency_ms=0.005*(a['ops']-b['ops']), stats=stats,
                  parent_certificate=str(baseline), arm_certificate=str(arm), guards=[str(p) for p in guards])
    exclusive(home/('comparison_%d.json' % seed), report)
    print(json.dumps(report), flush=True)
    if not report['pass_all']:
        raise RuntimeError('Exact certificate failed; no later network or integration is authorized')
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-authorized', action='store_true')
    parser.add_argument('new_run_name')
    args=parser.parse_args()
    assert args.run_authorized, 'Root must assign one slot before --run-authorized is used'
    assert re.fullmatch(r'astra_compute_v1_ext_[a-zA-Z0-9_]+', args.new_run_name)
    for p, expected in PINS.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected, p
    prior=json.loads((D/'astra_compute_v1_arm136_comparison.json').read_text())
    assert prior['pass_all']
    home=D/args.new_run_name
    home.mkdir(exist_ok=False)
    exclusive(home/'prereg.json', dict(stamp=stamp(), seeds=list(SEEDS),
              purpose='exact implementation certification; no model selection',
              parent_sha=PINS[PARENT], arm_sha=PINS[ARM], numerical_slots=1))
    env=dict(os.environ, PB_ALL='1', STATS='_ASTRA_COMPUTE_STATS')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
                'VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
        env[key]='1'
    team('Compute-copy exact-code extension begins in one authorized slot: paired fired parent and arm on 7100137, 7100600, 7100601, sequential guarded BLAS1; no model selection. Every pair requires all 16 prediction bytes/MSEs exact, positive bill credit equal to its own counter, and clean guards. Output '+str(home.relative_to(O))+'.')
    reports=[]
    try:
        for seed in SEEDS:
            base,bg=run_native(home,seed,'parent',PARENT,env)
            arm,ag=run_native(home,seed,'arm',ARM,env)
            reports.append(compare(home,seed,base,arm,(bg,ag)))
        exclusive(home/'summary.json', dict(pass_all=True, reports=reports))
        team('Compute-copy exact-code extension passed its three additional paired networks; every layer prediction and MSE exact, bill counters exact. Output '+str(home.relative_to(O))+'/summary.json. Slot released; no integration or submission performed.')
    except BaseException as error:
        exclusive(home/'stopped.json', dict(stamp=stamp(), error=repr(error), completed_pairs=reports))
        team('Compute-copy exact-code extension stopped: '+repr(error)+'. Output '+str(home.relative_to(O))+'/stopped.json; no further native run or integration.')
        raise

if __name__=='__main__':
    main()
