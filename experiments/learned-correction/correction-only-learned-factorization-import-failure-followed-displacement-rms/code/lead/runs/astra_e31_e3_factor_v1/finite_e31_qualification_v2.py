"""HELD remaining5 after source-only import assertion repair; explicit reuse grant required.
Exact already-completed fresh parent136 reuse only; no scientific followup.
"""
import argparse,hashlib,importlib.util,json,math,os,signal
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
G=O/'scout_native/certify_astra_bank_copy_v2.py'
assert sha(G)=='34be6065a292e8436a3d07141b746c623b6abc6c937be9555f07471c90e6e381'
spec=importlib.util.spec_from_file_location('_private_e31_guard',G)
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
H=D/'owned_native_runner_v1.py'
assert sha(H)=='aae933ceb3e05eef837ec925069ced9298a80bf9d3c5ea51543083611dba3823'
exec(compile(H.read_text(),str(H),'exec'),guard.__dict__)
guard.JUDGE=D/'native_e31_qualification_judge_v1.py'
CAP=0.001
ORDER=[('OFF',7100136),('ON',7100136),('parent',7100137),('OFF',7100137),('ON',7100137)]
PRESERVED=('_ASTRA_COMPUTE_STATS','_UBC_STATS','_UBX_STATS','_UBX_RC_STATS')
STATS=PRESERVED+('_ASTRA_E31_STATS','_ASTRA_E31_META','_ASTRA_E31_RUNTIME','_RQD_STATS',
       '_UBX_STATE','_UBX_RCACHE','_UBC_RUN','_UBC_CONTEXT','_UBX_CTX','_UBX_RC',
       '_OTT_CONTEXT','_DCR_STASH','_DCR_CALLS','_DCR_OPEN')
write=guard.exclusive


def config(path,digest,grant_path):
    assert sha(path)==digest
    c=json.loads(Path(path).read_text())
    assert c['purpose']=='e31_import_repair_continuation5'
    assert c['fixed_order']==[list(x) for x in ORDER] and c['prediction_error_cap']==CAP
    assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
    assert c['candidates']['parent']==dict(path=str(O/'engineer/a62_modules/s49/s56k1tw_x/candidate.py'),
             sha256='e7d85276a625611d098b305c59e5e527d9d0df7941b67044bc0928e97eb47141',mode=-1)
    assert c['new_native_calls_max']==5 and len(c['native_reuse'])==1
    assert c['native_reuse'][0]['role']=='parent' and c['native_reuse'][0]['seed']==7100136
    assert c['original_protocol_sha256']=='5c9bb9b1dab5fcd45e1c6a3630456403c591dcc956a242f7f6f9659e75ab0edb'
    retired=json.loads(Path(c['import_failure_retirement']['path']).read_text())
    assert sha(c['import_failure_retirement']['path'])==c['import_failure_retirement']['sha256']
    assert retired['managed_session_exit_code']==1 and retired['whole_groups_retired'] is True and retired['no_OFF_prediction_or_ON_call'] is True
    fullimport=json.loads(Path(c['full_import_proof']['path']).read_text())
    assert sha(c['full_import_proof']['path'])==c['full_import_proof']['sha256']
    assert fullimport['status']=='PASS_FULL_IMPORT_ONLY' and fullimport['model_evaluations']==0
    assert {v['role']:v['source_sha256'] for v in fullimport['records']}=={k:v['sha256'] for k,v in c['candidates'].items()}
    assert c['both_fresh_parent_and_OFF_required'] is True
    guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
    r=json.loads(Path(grant_path).read_text())
    assert r['approved'] is True and r['purpose']==c['purpose']
    assert r['protocol_sha256']==digest and r['runner_sha256']==sha(__file__)
    assert r['run_name']==c['run_name'] and r['numerical_slots']==1
    assert r['prediction_error_cap']==CAP and r['fixed_order']==c['fixed_order']
    assert r['reuse_completed_fresh_parent136_approved'] is True
    assert r['arithmetic_and_scientific_gates_unchanged'] is True
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
    for name in ('_UBC_RUN','_UBC_CONTEXT','_UBX_CTX','_UBX_RC','_OTT_CONTEXT','_DCR_STASH'):
        assert st[name]==[None],name
    assert st['_DCR_CALLS']==[] and st['_DCR_OPEN']==[False]
    for name in ('_UBX_STATE','_UBX_RCACHE'):
        x=st[name];assert x['run'] is None and x['cache']=={} and x['bytes']==0
        if name=='_UBX_RCACHE':assert x['owner'] is None
    x=st['_UBC_STATS'];assert x['retired'] is True and x['live_cache_bytes']==x['retained_cache_bytes']==0
    assert 0<=x['max_resident_bytes']<=512000000 and 0<=x['peak_cache_bytes']<=512000000
    if role=='parent':assert st['_ASTRA_E31_RUNTIME'] is None
    else:assert st['_ASTRA_E31_RUNTIME']==dict(active=False,layer=None,retired=True)


def current(c,item,role,seed):
    import numpy as np
    for key in ('certificate','guard'):assert sha(item[key])==item[key+'_sha256']
    p=Path(item['certificate']);assert sha(p.with_name(p.stem+'_pred.npz'))==item['prediction_npz_sha256']
    d=json.loads(p.read_text());g=json.loads(Path(item['guard']).read_text());source=c['candidates'][role]
    assert d['candidate']==source['path'] and d['sha256']==source['sha256'] and d['e31_mode']==source['mode']
    assert d['judge_sha256']==c['source_pins'][str(guard.JUDGE)]
    assert d['loader_sha256']==c['source_pins'][str(O/'analyst/fresh/loader.py')]
    assert d['environment']=={**{k:'1' for k in guard.THREADS},'PB_ALL':'1','CANDIDATE_SHA256':source['sha256'],'E31_MODE':str(source['mode'])}
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
    keys=('pb_single_sha256','pb_single_count','pb_wall_s')
    assert all(a['cert']['setup'][k]==b['cert']['setup'][k] for k in keys)
    ca=dict(a['cert']['setup']['cache_counts']);cb=dict(b['cert']['setup']['cache_counts'])
    if parent:
        assert '_ASTRA_E31_CONST' not in ca and cb.pop('_ASTRA_E31_CONST')==0
    assert ca==cb


def gate_OFF(base,off):
    setup_equal(base,off,parent=True)
    a,b=off['row'],base['row'];st=a['stats']
    checks=dict(all16_bytes=off['pred'].tobytes()==base['pred'].tobytes(),all16_MSE=a['mse_layers']==b['mse_layers'],
        scalar_MSE=a['mse_final']==b['mse_final'],prediction_hash=a['pred_sha']==b['pred_sha'],
        flops=a['flops']==b['flops'],ops=a['ops']==b['ops'],
        carried_stats=all(st[k]==b['stats'][k] for k in PRESERVED),
        no_ON_counters=st['_ASTRA_E31_STATS']==dict(on_calls=0,off_calls=0,Q=0,H=0,gaussian_reuse_calls=0,constant_flops=0),
        delegated_all14=st['_ASTRA_E31_META']==dict(dispatch_calls=0,row_gather_elements=0,delegated=14,elided_formation_elements=0,elided_gather_elements=0))
    assert all(checks.values()),('OFF_identity_failed',checks)
    return checks


def displacement_gate(base_y,arm_y,mse_layers):
    import numpy as np
    d=np.sqrt(np.mean((arm_y.astype(np.float64)-base_y.astype(np.float64))**2,axis=1))
    b=np.sqrt(np.asarray(mse_layers,dtype=np.float64))
    assert d.shape==b.shape==(16,) and np.isfinite(d).all() and np.isfinite(b).all() and (b>0).all()
    ratio=d/b
    return dict(cap=CAP,all_layers_pass=bool((ratio<=CAP).all()),ratios=ratio.tolist(),
                displacement_RMSE=d.tolist(),parent_RMSE=b.tolist())


def counter_credit(st):
    s=st['_ASTRA_E31_STATS'];m=st['_ASTRA_E31_META']
    assert set(s)==set(('on_calls','off_calls','Q','H','gaussian_reuse_calls','constant_flops'))
    assert set(m)==set(('dispatch_calls','row_gather_elements','delegated','elided_formation_elements','elided_gather_elements'))
    assert all(type(v) is int and v>=0 for d in (s,m) for v in d.values())
    assert {k:v for k,v in s.items() if k!='constant_flops'}==dict(on_calls=28,off_calls=0,Q=14*2228224,H=14*10240,gaussian_reuse_calls=28)
    assert s['constant_flops'] in (0,6,12)
    assert m==dict(dispatch_calls=14,row_gather_elements=14*9216,delegated=0,
                   elided_formation_elements=14*1024**2,elided_gather_elements=28*1024**2)
    return 16*s['Q']+68*s['H']-4*m['row_gather_elements']+m['elided_formation_elements']+m['elided_gather_elements']-s['constant_flops']


def gate_ON(off,arm):
    setup_equal(off,arm)
    a,b=arm['row'],off['row'];credit=b['flops']-a['flops'];counter=counter_credit(a['stats'])
    displacement=displacement_gate(off['pred'],arm['pred'],b['mse_layers'])
    checks=dict(layer0_bytes=off['pred'][0].tobytes()==arm['pred'][0].tobytes(),layer0_MSE=b['mse_layers'][0]==a['mse_layers'][0],
        carried_stats=all(a['stats'][k]==b['stats'][k] for k in PRESERVED),positive_credit=credit>0,
        actual_bill_exact_counter=credit==counter,displacement_cap=displacement['all_layers_pass'])
    report=dict(checks=checks,displacement=displacement,measured_flop_credit=credit,counter_credit=counter,
        bill_change_pct=100*(a['flops']/b['flops']-1),charged_ops_delta=a['ops']-b['ops'],
        final_MSE_change_pct=100*(a['mse_final']/b['mse_final']-1),
        final_log_net=math.log(a['mse_final']/b['mse_final'])+math.log(a['flops']/b['flops']),
        runtime=dict(OFF_setup_s=off['cert']['setup']['wall_s'],ON_setup_s=arm['cert']['setup']['wall_s'],
                     OFF_predict_s=b['predict_wall_unrounded_s'],ON_predict_s=a['predict_wall_unrounded_s'],production_wall_valid=False),
        qualification_not_scientific_pass=True)
    assert all(checks.values()),('ON_qualification_failed',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-authorized',action='store_true')
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--grant',type=Path,required=True);a=p.parse_args();assert a.run_authorized
    c=config(a.protocol,a.protocol_sha256,a.grant)
    run=D/c['run_name'];assert run.parent==D and run.name.startswith('run_e31_qualification_')
    run.mkdir(exist_ok=False)
    write(run/'PLAN_v1.json',dict(protocol_sha256=a.protocol_sha256,root_grant_sha256=sha(a.grant),controller_pid=os.getpid(),
          fixed_order=c['fixed_order'],expected_guards=[str(run/f'native_{seed}_{role}_guard.json') for role,seed in ORDER],
          prediction_error_cap=CAP,reuse=c['native_reuse'],no_scientific_phase=True))
    proto=dict(pinned_files=c['source_pins'],candidates=c['candidates']);checks=[];evaluations=[];cases={}
    try:
        retained=c['native_reuse'][0]
        cases[('parent',7100136)]=current(c,retained,'parent',7100136)
        write(run/'REUSED_PARENT136_VERIFIED_v1.json',dict(source_sha256=c['candidates']['parent']['sha256'],evidence=retained,explicit_root_reuse=True))
        for role,seed in ORDER:
            config(a.protocol,a.protocol_sha256,a.grant)
            env=dict(os.environ,PB_ALL='1',STATS=','.join(STATS),E31_MODE=str(c['candidates'][role]['mode']),PYTHONDONTWRITEBYTECODE='1')
            item=guard.run_native(run,proto,role,seed,env);evaluations.append(dict(role=role,seed=seed,**item))
            got=current(c,item,role,seed);cases[(role,seed)]=got
            if role=='OFF':check=gate_OFF(cases[('parent',seed)],got)
            elif role=='ON':check=gate_ON(cases[('OFF',seed)],got)
            else:check=dict(valid_fresh_parent=True)
            checks.append(dict(seed=seed,role=role,check=check));write(run/f'CHECK_{role}_{seed}_v1.json',checks[-1])
            print(json.dumps(dict(completed_calls=len(evaluations),total=5,role=role,seed=seed,required_gate_pass=True)),flush=True)
        write(run/'QUALIFICATION_RETIRED_v1.json',dict(status='qualification6_complete_one_retained_parent_plus_five_new',protocol_sha256=a.protocol_sha256,
              root_grant_sha256=sha(a.grant),evaluations=evaluations,reused_parent=c['native_reuse'],checks=checks,cap=CAP,all_guards_retired=True,
              owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,no_scientific_pass=True,
              no_followup_authorization=True,production_qualified=False))
    except BaseException as e:
        write(run/'STOPPED_v1.json',dict(error=repr(e),checks=checks,evaluations=evaluations,
              owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,
              slot_reservation_retained=not guard.OWNED_RETIREMENT_CONFIRMED,no_retry=True))
        raise


def interrupted(signum,frame):raise KeyboardInterrupt('Owned E31 controller signal%s'%signum)
if __name__=='__main__':
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:main()
    finally:signal.signal(signal.SIGTERM,previous)
