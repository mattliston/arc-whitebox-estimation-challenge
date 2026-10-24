"""HELD fixed fresh parent/OFF/ON136137 qualification; root grant required.
No historical reuse or scientific gain selection on these price seeds. No fresh/OOS phase, integration or submission.
"""
import argparse,hashlib,importlib.util,json,math,os,signal
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
G=O/'scout_native/certify_astra_bank_copy_v2.py'
assert sha(G)=='34be6065a292e8436a3d07141b746c623b6abc6c937be9555f07471c90e6e381'
spec=importlib.util.spec_from_file_location('_private_arow_guard',G)
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
H=D/'owned_native_runner_v1.py'
assert sha(H)=='aae933ceb3e05eef837ec925069ced9298a80bf9d3c5ea51543083611dba3823'
exec(compile(H.read_text(),str(H),'exec'),guard.__dict__)
guard.JUDGE=D/'native_arow_price_judge_v1.py'
CAP=None
ORDER=[(role,seed) for seed in (7100136,7100137) for role in ('parent','OFF','ON')]
PRESERVED=('_ASTRA_COMPUTE_STATS','_UBC_STATS','_UBX_STATS','_UBX_RC_STATS',
           '_ASTRA_UBOWN_STATS','_ASTRA_CLUSTER_STATS','_ASTRA_BCC_STATS','_ASTRA_QL_STATS','_RQD_STATS')
STATS=PRESERVED+('_AROW_STATS','_UBX_STATE','_UBX_RCACHE','_UBC_RUN','_UBC_CONTEXT','_UBX_CTX','_UBX_RC',
       '_OTT_CONTEXT','_DCR_STASH','_DCR_CALLS','_DCR_OPEN','_ASTRA_BCC_CTX','_ASTRA_CLUSTER_CTX','_AROW_ACTIVE','_AROW_GATE')
write=guard.exclusive


def config(path,digest,grant_path):
    assert sha(path)==digest
    c=json.loads(Path(path).read_text())
    assert c['purpose']=='arow_fixed_price6'
    assert c['approved'] is True and c['reviewed_by']=='root'
    assert c['fixed_price_cap_log']==0.0005 and c['no_MSE_gain_selection_on_price_seeds'] is True
    assert c['fixed_order']==[list(x) for x in ORDER] and c['OFF_exact_all16_required'] is True
    assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
    assert c['candidates']['parent']==dict(path=str(O/'engineer/a62_modules/s49/s56ubq1tw_x/candidate.py'),
             sha256='c149e5800751892eb4d203341154ab2444e824a067ffaf791293a49e035ca0a3',mode=-1)
    assert c['native_reuse']==[] and c['new_native_calls_max']==6
    assert c['both_fresh_parent_and_OFF_required'] is True
    guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
    r=json.loads(Path(grant_path).read_text())
    assert r['approved'] is True and r['purpose']==c['purpose']
    assert r['protocol_sha256']==digest and r['runner_sha256']==sha(__file__)
    assert r['run_name']==c['run_name'] and r['numerical_slots']==1
    assert r['OFF_exact_all16_required'] is True and r['fixed_order']==c['fixed_order']
    assert all(os.environ.get(k)=='1' for k in guard.THREADS)
    assert os.getpriority(os.PRIO_PROCESS,0)==0
    assert [t['seed'] for t in c['truth_metadata']]==[7100136,7100137]
    for t in c['truth_metadata']:assert sha(t['path'])==t['sha256']
    return c


def prediction(path,seed):
    import numpy as np
    p=Path(path).with_name(Path(path).stem+'_pred.npz')
    with np.load(p,allow_pickle=False) as z:
        assert z.files==[str(seed)];y=z[str(seed)].copy()
    assert y.shape==(16,1024) and y.dtype==np.dtype('float32') and np.isfinite(y).all()
    return y


def contexts(st,role):
    for name in ('_UBC_RUN','_UBC_CONTEXT','_UBX_CTX','_UBX_RC','_OTT_CONTEXT','_DCR_STASH','_ASTRA_BCC_CTX','_ASTRA_CLUSTER_CTX'):
        assert st[name]==[None],name
    assert st['_DCR_CALLS']==[] and st['_DCR_OPEN']==[False]
    for name in ('_UBX_STATE','_UBX_RCACHE'):
        x=st[name];assert x['run'] is None and x['cache']=={} and x['bytes']==0
        if name=='_UBX_RCACHE':assert x['owner'] is None
    x=st['_UBC_STATS'];assert x['retired'] is True and x['live_cache_bytes']==x['retained_cache_bytes']==0
    assert 0<=x['max_resident_bytes']<=512000000 and 0<=x['peak_cache_bytes']<=512000000
    if role=='parent':
        assert st['_AROW_STATS'] is None and st['_AROW_ACTIVE'] is None and st['_AROW_GATE'] is None
    else:
        assert isinstance(st['_AROW_STATS'],dict) and st['_AROW_STATS']['retired'] is True
        assert st['_AROW_ACTIVE']==[False] and st['_AROW_GATE']==[None]


def current(c,item,role,seed):
    import numpy as np
    for key in ('certificate','guard'):assert sha(item[key])==item[key+'_sha256']
    p=Path(item['certificate']);assert sha(p.with_name(p.stem+'_pred.npz'))==item['prediction_npz_sha256']
    d=json.loads(p.read_text());g=json.loads(Path(item['guard']).read_text());source=c['candidates'][role]
    assert d['candidate']==source['path'] and d['sha256']==source['sha256'] and d['arow_mode']==source['mode']
    assert d['judge_sha256']==c['source_pins'][str(guard.JUDGE)]
    assert d['loader_sha256']==c['source_pins'][str(O/'analyst/fresh/loader.py')]
    assert d['environment']=={**{k:'1' for k in guard.THREADS},'PB_ALL':'1','CANDIDATE_SHA256':source['sha256'],'AROW_MODE':str(source['mode'])}
    assert g['command']==[str(guard.RT),'-B',str(guard.JUDGE),source['path'],str(seed),str(p)]
    assert g['completed'] is True and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] is True and g['remaining_rows']==[]
    assert 0<g['max_rss_bytes']<7500000000 and 0<=g['max_footprint_bytes']<7500000000
    assert d['setup']['pb_wall_s']==1e9 and math.isfinite(d['setup']['wall_s']) and d['setup']['wall_s']>=0
    y=prediction(p,seed)
    assert d['seeds']==[seed] and len(d['rows'])==1
    row=d['rows'][0];assert row['seed']==seed
    assert type(row['flops']) is int and .1*2**41<=row['flops']<=2**41
    assert type(row['ops']) is int and row['ops']>0
    assert len(row['mse_layers'])==16 and all(math.isfinite(v) and v>0 for v in row['mse_layers'])
    assert row['mse_final']==row['mse_layers'][-1] and hashlib.sha256(y.tobytes()).hexdigest()[:16]==row['pred_sha']
    assert row['stats']['_ASTRA_COMPUTE_STATS']['calls']==14
    assert row['stats']['_ASTRA_COMPUTE_STATS']['saved_copy_elements']>0
    contexts(row['stats'],role)
    truth=next(t for t in c['truth_metadata'] if t['seed']==seed)
    assert row['weights_before_sha256']==row['weights_after_sha256']==truth['weights_sha256']
    return dict(cert=d,row=row,pred=y,path=str(p),evidence=item)


def setup_equal(a,b,parent=False):
    keys=('pb_single_sha256','pb_single_count','pb_wall_s','cache_counts')
    assert all(a['cert']['setup'][k]==b['cert']['setup'][k] for k in keys)


def exact_checks(base,arm):
    a,b=arm['row'],base['row']
    return dict(all16_bytes=arm['pred'].tobytes()==base['pred'].tobytes(),
        all16_MSE=a['mse_layers']==b['mse_layers'],scalar_MSE=a['mse_final']==b['mse_final'],
        prediction_hash=a['pred_sha']==b['pred_sha'],
        carried_stats=all(a['stats'][k]==b['stats'][k] for k in PRESERVED))


def gate_OFF(base,off):
    setup_equal(base,off)
    a,b=off['row'],base['row'];checks=exact_checks(base,off)
    checks.update(flops=a['flops']==b['flops'],ops=a['ops']==b['ops'],
        no_ON_counters=a['stats']['_AROW_STATS']==dict(calls=0,changed_calls=0,events=[],retired=True))
    assert all(checks.values()),('OFF_identity_failed',checks)
    return checks


def route_gate(st):
    s=st['_AROW_STATS'];events=s['events']
    assert s['retired'] is True and s['calls']==3 and len(events)==3
    assert [e['layer'] for e in events]==[10,11,12]
    assert s['changed_calls']==sum(int(e['swapped']>0) for e in events)>0
    for e in events:
        assert e['rank']==320 and e['materialized'] is True
        assert e['baseline_kept']==e['selected_kept'] and e['tail_rows']==e['rows']-e['selected_kept']
        assert all(type(e[k]) is int and e[k]>=0 for k in ('rows','baseline_kept','selected_kept','tail_rows','protected','selected_protected','swapped'))
        assert e['selected_protected']<=e['protected']<=e['rows'] and e['swapped']<=e['selected_kept']
        assert e['reason'] in ('allocated','full_parent','protected_parent')
        if e['reason']=='allocated':assert e['selected_protected']==e['protected']<e['selected_kept']
        elif e['reason']=='protected_parent':assert e['protected']>=e['selected_kept'] and e['swapped']==0
        else:assert e['selected_kept']==e['rows'] and e['swapped']==0
    return s


def gate_ON(off,arm):
    # New physics. Only the unchanged early prefix is an identity gate; never
    # select this arm using its 136/137 MSE. Complete bill includes later shifts.
    setup_equal(off,arm)
    a,b=arm['row'],off['row'];stats=route_gate(a['stats'])
    checks=dict(layer0_to9_bytes=arm['pred'][:10].tobytes()==off['pred'][:10].tobytes(),
        layer0_to9_MSE=a['mse_layers'][:10]==b['mse_layers'][:10],
        final_changed=arm['pred'][-1].tobytes()!=off['pred'][-1].tobytes(),
        positive_finite_price=math.isfinite(a['flops']/b['flops']) and a['flops']>0 and b['flops']>0)
    report=dict(checks=checks,route=stats,measured_log_bill=math.log(a['flops']/b['flops']),
        bill_change_pct=100*(a['flops']/b['flops']-1),charged_ops_delta=a['ops']-b['ops'],
        runtime=dict(OFF_setup_s=off['cert']['setup']['wall_s'],ON_setup_s=arm['cert']['setup']['wall_s'],
                     OFF_predict_s=b['predict_wall_unrounded_s'],ON_predict_s=a['predict_wall_unrounded_s'],production_wall_valid=False),
        MSE_gain_not_used_for_selection=True,accuracy_unqualified=True)
    assert all(checks.values()),('ON_route_failed',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-authorized',action='store_true')
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--grant',type=Path,required=True);a=p.parse_args();assert a.run_authorized
    c=config(a.protocol,a.protocol_sha256,a.grant)
    run=D/c['run_name'];assert run.parent==D and run.name.startswith('run_arow_qualification_')
    run.mkdir(exist_ok=False)
    write(run/'PLAN_v1.json',dict(protocol_sha256=a.protocol_sha256,root_grant_sha256=sha(a.grant),controller_pid=os.getpid(),
          fixed_order=c['fixed_order'],expected_guards=[str(run/f'native_{seed}_{role}_guard.json') for role,seed in ORDER],
          OFF_exact_all16_required=True,reuse=[],no_scientific_phase=True))
    proto=dict(pinned_files=c['source_pins'],candidates=c['candidates']);checks=[];evaluations=[];cases={}
    try:
        for role,seed in ORDER:
            config(a.protocol,a.protocol_sha256,a.grant)
            env=dict(os.environ,PB_ALL='1',STATS=','.join(STATS),AROW_MODE=str(c['candidates'][role]['mode']),PYTHONDONTWRITEBYTECODE='1')
            item=guard.run_native(run,proto,role,seed,env);evaluations.append(dict(role=role,seed=seed,**item))
            got=current(c,item,role,seed);cases[(role,seed)]=got
            if role=='OFF':check=gate_OFF(cases[('parent',seed)],got)
            elif role=='ON':check=gate_ON(cases[('OFF',seed)],got)
            else:check=dict(valid_fresh_parent=True)
            checks.append(dict(seed=seed,role=role,check=check));write(run/f'CHECK_{role}_{seed}_v1.json',checks[-1])
            print(json.dumps(dict(completed_calls=len(evaluations),total=6,role=role,seed=seed,required_gate_pass=True)),flush=True)
        price=max(x['check']['measured_log_bill'] for x in checks if x['role']=='ON')
        price_pass=price<=c['fixed_price_cap_log']
        write(run/'QUALIFICATION_RETIRED_v1.json',dict(status='price6_complete_all_guards_retired',
              max_log_bill=price,price_cap_log=c['fixed_price_cap_log'],price_pass=price_pass,accuracy_unqualified=True,protocol_sha256=a.protocol_sha256,
              root_grant_sha256=sha(a.grant),evaluations=evaluations,checks=checks,OFF_exact_all16_required=True,all_guards_retired=True,
              owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,no_scientific_pass=True,
              no_followup_authorization=True,production_qualified=False))
    except BaseException as e:
        write(run/'STOPPED_v1.json',dict(error=repr(e),checks=checks,evaluations=evaluations,
              owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,
              slot_reservation_retained=not guard.OWNED_RETIREMENT_CONFIRMED,no_retry=True))
        raise


def interrupted(signum,frame):raise KeyboardInterrupt('Owned AROW controller signal%s'%signum)
if __name__=='__main__':
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:main()
    finally:signal.signal(signal.SIGTERM,previous)
