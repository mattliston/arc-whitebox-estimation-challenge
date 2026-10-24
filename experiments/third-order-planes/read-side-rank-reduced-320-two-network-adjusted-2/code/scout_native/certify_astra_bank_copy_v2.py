"""HELD finite nine-evaluation bank-copy certificate; explicit root slot required.
One fresh process per fixed candidate/seed. Never infer baseline equivalence.
No production, seal, amend, submission, or automatic retry capability.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

S = Path(__file__).resolve().parent
O = S.parent
P2 = O.parent
REPO = P2.parents[3]
PY = REPO / '.venv/bin/python'
RT = P2 / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
PROTOCOL = S / 'astra_bank_copy_NATIVE_PROTOCOL_HELD_v2.json'
JUDGE = S / 'run_cert_astra_bank_copy_v2.py'
GUARD = O / 'release/guard_footprint_v3.py'
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'VECLIB_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
STATS = ('_ASTRA_BCC_STATS', '_ASTRA_BCC_CTX', '_ASTRA_BCC_RIGHT',
         '_ASTRA_COMPUTE_STATS', '_ASTRA_UBOWN_STATS', '_ASTRA_CLUSTER_STATS',
         '_UBC_STATS', '_UBX_STATS', '_UBX_RC_STATS', '_UBX_STATE', '_UBX_RCACHE',
         '_UBC_RUN', '_UBC_CONTEXT', '_UBX_CTX', '_UBX_RC', '_ASTRA_CLUSTER_CTX',
         '_OTT_CONTEXT', '_DCR_STASH', '_DCR_CALLS', '_DCR_OPEN')
UB_STATS = ('_ASTRA_COMPUTE_STATS', '_ASTRA_UBOWN_STATS', '_ASTRA_CLUSTER_STATS', '_UBC_STATS', '_UBX_STATS', '_UBX_RC_STATS')
COUNTERS = ('calls', 'expand_elements', 'join_elements', 'outer_elements',
            'destination_elements', 'dcr_elements', 'rowmap_elements',
            'lazy_elements', 'newborn_elements')
OWNED_RETIREMENT_CONFIRMED = True
SEEDS = (7100136, 7100137, 7100600, 7100601)
ORDER = [('parent', 7100136), ('noop', 7100136), ('arm', 7100136)] + [
    (kind, seed) for seed in SEEDS[1:] for kind in ('parent', 'arm')]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

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

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def resources():
    pressure = subprocess.check_output(['/usr/bin/memory_pressure'], text=True, timeout=15)
    free = int(re.search(r'System-wide memory free percentage: (\d+)%', pressure).group(1))
    output = subprocess.check_output(['/bin/ps', '-axo', 'pid=,ppid=,rss=,comm='], text=True, timeout=5)
    workers = []
    for line in output.splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4 and int(fields[2]) > 1048576 and 'python' in Path(fields[3]).name.lower():
            workers.append(dict(pid=int(fields[0]), ppid=int(fields[1]), rss_kib=int(fields[2])))
    return dict(utc=stamp(), free_percent=free, workers=workers, truth_reserve=1, counted=len(workers)+1)

def pins(protocol):
    for path, digest in protocol['pinned_files'].items():
        require(sha(Path(path)) == digest, 'Source pin mismatch: ' + path)
    for item in protocol['candidates'].values():
        require(sha(Path(item['path'])) == item['sha256'], 'Candidate pin mismatch')

def retirement(stats, kind):
    checks = {name:stats.get(name)==[None] for name in
              ('_UBC_RUN','_UBC_CONTEXT','_UBX_CTX','_UBX_RC','_ASTRA_CLUSTER_CTX',
               '_OTT_CONTEXT','_DCR_STASH')}
    checks.update(_DCR_CALLS=stats.get('_DCR_CALLS')==[], _DCR_OPEN=stats.get('_DCR_OPEN')==[False])
    if kind != 'parent':
        checks.update({name:stats.get(name)==[None] for name in ('_ASTRA_BCC_CTX','_ASTRA_BCC_RIGHT')})
    for name in ('_UBX_STATE','_UBX_RCACHE'):
        state=stats.get(name)
        checks[name] = (isinstance(state,dict) and state.get('run') is None and state.get('cache') == {}
                        and state.get('bytes') == 0 and (name!='_UBX_RCACHE' or state.get('owner') is None))
    ubc=stats.get('_UBC_STATS',{})
    checks['UBC_retired']=ubc.get('retired') is True and ubc.get('retained_cache_bytes')==0 and ubc.get('live_cache_bytes')==0
    checks['UBC_peak_cap']=0<=ubc.get('peak_cache_bytes',float('inf'))<=512000000
    checks['UBC_resident_cap']=0<=ubc.get('max_resident_bytes',float('inf'))<=512000000
    return checks

def credit_counter(stats):
    c=stats['_ASTRA_BCC_STATS']
    require(set(c)==set(COUNTERS), 'Bank-copy counter schema mismatch')
    require(all(isinstance(c[k],int) and c[k]>=0 for k in COUNTERS), 'Invalid bank-copy counter')
    return (c['expand_elements']+c['join_elements']+c['outer_elements']+c['lazy_elements']+c['newborn_elements']
            -c['destination_elements']-c['dcr_elements']-8*c['rowmap_elements'])

def run_native(home, protocol, kind, seed, env):
    global OWNED_RETIREMENT_CONFIRMED
    pins(protocol)
    # Pinned guard applies /usr/bin/nice -n10 once to its own native child.
    # Enforce the controller's base priority, avoiding accidental nice20.
    require(os.getpriority(os.PRIO_PROCESS,0)==0,'Controller must start at nice0; guard applies native nice10')
    before=resources()
    event(home/'resources.jsonl',dict(stage='before',kind=kind,seed=seed,**before))
    require(before['free_percent']>=50 and before['counted']<10, 'No free native slot or memory below50')
    stem=home/('native_%d_%s'%(seed,kind))
    result,log,guard=(Path(str(stem)+suffix) for suffix in ('.json','.log','_guard.json'))
    require(all(not p.exists() for p in (result,log,guard,Path(str(stem)+'_pred.npz'))), 'Fresh output required')
    candidate=protocol['candidates'][kind]
    env=dict(env,CANDIDATE_SHA256=candidate['sha256'])
    command=[str(PY),'-B',str(GUARD),'--record',str(guard),'--seconds','900',
             '--footprint-limit','7500000000','--rss-backstop','7500000000',
             str(RT),'-B',str(JUDGE),candidate['path'],str(seed),str(result)]
    started=time.monotonic()
    with log.open('x') as stream:
        child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,cwd=P2)
        OWNED_RETIREMENT_CONFIRMED = False
        owner=dict(controller_nice=0, guard_native_nice=10, guard_record=str(guard), utc=stamp(),kind=kind,seed=seed,controller_pid=os.getpid(),owned_guard_pid=child.pid,
                   command=command,candidate_sha256=candidate['sha256'],environment={k:env[k] for k in THREADS+('PB_ALL','CANDIDATE_SHA256')})
        event(home/'ownership.jsonl',owner)
        print(json.dumps(owner),flush=True)
        last_print=time.monotonic()
        try:
            while child.poll() is None:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    current=resources()
                    event(home/'resources.jsonl',dict(stage='running',kind=kind,seed=seed,**current))
                    require(current['free_percent']>=50 and current['counted']<=10,'Live resource bound exceeded; retire only owned guard')
                    if time.monotonic()-last_print>=30:
                        print(json.dumps(dict(kind=kind,seed=seed,**current)),flush=True)
                        last_print=time.monotonic()
        except BaseException as original_error:
            wait_error=None
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired as error:
                    wait_error=repr(error)
            try:
                stopped_guard=json.loads(guard.read_text())
                OWNED_RETIREMENT_CONFIRMED=(child.poll() is not None and stopped_guard.get('whole_group_retired') is True
                                            and stopped_guard.get('remaining_rows')==[])
            except (OSError,ValueError):
                OWNED_RETIREMENT_CONFIRMED=False
            exclusive(home/('interruption_%s_%d.json'%(kind,seed)),dict(
                utc=stamp(),owned_guard_pid=child.pid,returncode=child.poll(),wait_error=wait_error,
                original_error=repr(original_error),whole_group_retirement_verified=OWNED_RETIREMENT_CONFIRMED,
                slot_released=OWNED_RETIREMENT_CONFIRMED))
            if not OWNED_RETIREMENT_CONFIRMED:
                raise RuntimeError('Owned group retirement unconfirmed; retain slot reservation for root') from original_error
            raise
    elapsed=time.monotonic()-started
    g=json.loads(guard.read_text())
    OWNED_RETIREMENT_CONFIRMED=(child.poll() is not None and g.get('whole_group_retired') is True and g.get('remaining_rows')==[])
    require(child.returncode==0 and g.get('completed') is True and g.get('exit_code')==0
            and not g.get('error') and g.get('whole_group_retired') is True and g.get('remaining_rows')==[], 'Native guard incomplete or failed')
    require(g['max_footprint_bytes']<7500000000 and g['max_rss_bytes']<7500000000,'Guard footprint/RSS limit')
    pins(protocol)
    return dict(certificate=str(result),guard=str(guard),elapsed_s=elapsed,
                certificate_sha256=sha(result),prediction_npz_sha256=sha(result.with_name(result.stem+'_pred.npz')),
                guard_sha256=sha(guard),guard_max_rss_bytes=g['max_rss_bytes'],guard_max_footprint_bytes=g['max_footprint_bytes'])

def read_checked(item, protocol, kind, seed):
    import numpy as np
    result=Path(item['certificate']);d=json.loads(result.read_text())
    require(d['sha256']==protocol['candidates'][kind]['sha256'] and d['seeds']==[seed] and len(d['rows'])==1,'Result source/seed mismatch')
    require(d['judge_sha256']==protocol['pinned_files'][str(JUDGE)] and d['loader_sha256']==protocol['pinned_files'][str(O/'analyst/fresh/loader.py')],'Judge/loader mismatch')
    require(d['environment']['PB_ALL']=='1' and d['environment']['CANDIDATE_SHA256']==d['sha256']
            and all(d['environment'][k]=='1' for k in THREADS),'Native environment mismatch')
    require(d['setup']['pb_wall_s']==1e9 and d['setup']['wall_s']>=0,'Setup did not use full pinned prebind')
    with np.load(result.with_name(result.stem+'_pred.npz'),allow_pickle=False) as p:
        require(p.files==[str(seed)],'Prediction seed set mismatch')
        y=p[str(seed)]
    require(y.shape==(16,1024) and y.dtype==np.dtype('float32') and bool(np.isfinite(y).all()),'Invalid all16 prediction array')
    row=d['rows'][0]
    require(row['seed']==seed and isinstance(row['flops'],int) and 0<row['flops']<=2**41 and isinstance(row['ops'],int) and row['ops']>0,'Invalid row')
    require(len(row['mse_layers'])==16 and all(math.isfinite(x) and x>0 for x in row['mse_layers']) and row['mse_final']==row['mse_layers'][-1],'Invalid MSE vector')
    require(hashlib.sha256(y.tobytes()).hexdigest()[:16]==row['pred_sha'],'Prediction digest mismatch')
    retired=retirement(row['stats'],kind)
    require(all(retired.values()),'Context/cache retirement failure: '+repr(retired))
    if kind=='noop':
        require(credit_counter(row['stats'])==0 and all(v==0 for v in row['stats']['_ASTRA_BCC_STATS'].values()),'No-op counters not zero')
    return d,row,y,retired

def compare(home, protocol, parent_item, item, kind, seed):
    bd,b,bp,br=read_checked(parent_item,protocol,'parent',seed)
    ad,a,ap,ar=read_checked(item,protocol,kind,seed)
    exact=[bp[i].tobytes()==ap[i].tobytes() for i in range(16)]
    credit=b['flops']-a['flops']
    expected=credit_counter(a['stats'])
    checks=dict(all16_bytes=all(exact),all16_MSE=b['mse_layers']==a['mse_layers'],
                final_MSE=b['mse_final']==a['mse_final'],prediction_sha=b['pred_sha']==a['pred_sha'],
                setup_key_identity=all(bd['setup'][k]==ad['setup'][k] for k in ('pb_single_sha256','pb_single_count','pb_wall_s','cache_counts')),
                cache_stats_identity=all(b['stats'][k]==a['stats'][k] for k in UB_STATS),
                copy_counter_exact=credit==expected,contexts_retired=all(br.values()) and all(ar.values()))
    if kind=='noop':
        checks.update(flops_exact=credit==0,ops_exact=b['ops']==a['ops'])
    else:
        checks.update(positive_measured_gain=credit>0,bank_copy_reached=a['stats']['_ASTRA_BCC_STATS']['calls']>0)
    report=dict(kind=kind,seed=seed,pass_all=all(checks.values()),checks=checks,predictions_by_layer=exact,
                measured_flop_credit=credit,counter_credit=expected,bill_change_pct=100*(a['flops']/b['flops']-1),
                charged_ops_parent=b['ops'],charged_ops_arm=a['ops'],charged_ops_delta=a['ops']-b['ops'],
                nominal_5us_per_charged_op_delta_ms=.005*(a['ops']-b['ops']),
                actual_production_requests_measured=False,stats=a['stats'],retirement=ar,
                runtime=dict(parent_setup_s=bd['setup']['wall_s'],arm_setup_s=ad['setup']['wall_s'],
                             parent_predict_s=b['predict_wall_unrounded_s'],arm_predict_s=a['predict_wall_unrounded_s'],
                             production_wall_valid=False),parent=parent_item,arm=item)
    exclusive(home/('comparison_%s_%d.json'%(kind,seed)),report)
    print(json.dumps(dict(kind=kind,seed=seed,pass_all=report['pass_all'],checks=checks,
                         bill_change_pct=report['bill_change_pct'],charged_ops_delta=report['charged_ops_delta'])),flush=True)
    require(report['pass_all'],'Required identity/price/setup/context gate failed; no further seeds')
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-authorized',action='store_true')
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--grant',required=True,type=Path)
    parser.add_argument('new_run_name')
    args=parser.parse_args()
    require(args.run_authorized,'Explicit root slot grant required')
    require(re.fullmatch(r'astra_bank_copy_native_[a-zA-Z0-9_]+',args.new_run_name),'Invalid fresh run name')
    require(sha(PROTOCOL)==args.protocol_sha256,'Protocol hash mismatch')
    protocol=json.loads(PROTOCOL.read_text());pins(protocol)
    require(protocol['fixed_order']==[list(x) for x in ORDER] and protocol['seeds']==list(SEEDS)
            and protocol['reuse_baselines']==[],'Fixed fresh protocol changed')
    grant=json.loads(args.grant.read_text())
    require(grant.get('approved') is True and grant.get('purpose')=='bank_copy_native'
            and grant.get('protocol_sha256')==sha(PROTOCOL) and grant.get('runner_sha256')==sha(Path(__file__))
            and grant.get('run_name')==args.new_run_name and grant.get('numerical_slots')==1,'No exact root slot grant')
    require(all(os.environ.get(k)=='1' for k in THREADS),'Outer controller BLAS1 required')
    home=S/args.new_run_name;home.mkdir(exist_ok=False)
    exclusive(home/'prereg.json',dict(utc=stamp(),controller_pid=os.getpid(),protocol_sha256=sha(PROTOCOL),
              runner_sha256=sha(Path(__file__)),root_grant=str(args.grant),root_grant_sha256=sha(args.grant),
              order=ORDER,candidates=protocol['candidates'],numerical_slots=1,baseline_reuse=False))
    env=dict(os.environ,PB_ALL='1',STATS=','.join(STATS),PYTHONDONTWRITEBYTECODE='1')
    env.update({k:'1' for k in THREADS})
    team('Bank-copy finite exact-code qualification begins in one explicitly granted slot: exact S56XS+UB held parent; fresh parent136 then mandatory no-op136 and arm136, followed by fixed fresh parent/arm137,600,601. All16 bytes/MSE, measured copy counter, setup/ops, context retirement and native guards required. No adaptive seeds or production claim. Output '+str(home.relative_to(O))+'.')
    reports=[];parents={};evaluations=[]
    try:
        for kind,seed in ORDER:
            item=run_native(home,protocol,kind,seed,env)
            evaluations.append(dict(kind=kind,seed=seed,**item))
            if kind=='parent':
                read_checked(item,protocol,kind,seed)
                parents[seed]=item
                exclusive(home/('parent_checked_%d.json'%seed),dict(seed=seed,passed=True,**item))
            else:
                reports.append(compare(home,protocol,parents[seed],item,kind,seed))
        arms=[r for r in reports if r['kind']=='arm']
        exclusive(home/'summary.json',dict(pass_all=True,reports=reports,evaluations=evaluations,
                  geometric_mean_bill_change_pct=100*(math.exp(sum(math.log1p(r['bill_change_pct']/100) for r in arms)/4)-1),
                  maximum_charged_op_delta=max(r['charged_ops_delta'] for r in arms),
                  protocol_sha256=sha(PROTOCOL),parent_sha256=protocol['candidates']['parent']['sha256'],
                  production_qualified=False,own_prebind_qualified=False,current_final_composition_qualified=False,
                  limitation='Later combined stack on latest best needs source audit, own census/prebind, exact identities, normal/slow production request/wall gates. Existing UB source remains immutable.'))
        team('Bank-copy finite exact-code qualification passed its no-op and fixed four paired networks. All16 bytes/MSE, copy counter, setup identity, cache/context and guard gates passed; one native slot released. Own prebind, latest-best composition and production gates remain. Output '+str(home.relative_to(O))+'/summary.json.')
    except BaseException as error:
        exclusive(home/'stopped.json',dict(utc=stamp(),error=repr(error),reports=reports,evaluations=evaluations,
                  owned_group_retirement_verified=OWNED_RETIREMENT_CONFIRMED,slot_released=OWNED_RETIREMENT_CONFIRMED))
        state=('Owned group retirement verified; slot released.' if OWNED_RETIREMENT_CONFIRMED else
               'Owned group retirement UNCONFIRMED; retain slot reservation and escalate to root.')
        team('Bank-copy qualification stopped at a required gate. '+state+' No extra networks authorized: '+repr(error)+'. Output '+str(home.relative_to(O))+'/stopped.json.')
        raise

if __name__=='__main__':
    main()
