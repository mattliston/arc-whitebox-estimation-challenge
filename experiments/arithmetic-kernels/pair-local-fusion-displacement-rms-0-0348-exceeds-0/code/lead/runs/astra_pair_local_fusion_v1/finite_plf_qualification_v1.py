"""HELD four-call arithmetic qualification. Explicit root grant required.
No scientific fresh/OOS phase, automatic retry, integration or submission.
"""
import argparse,hashlib,importlib.util,json,math,os
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
G=O/'scout_native/certify_astra_bank_copy_v2.py'
assert sha(G)=='34be6065a292e8436a3d07141b746c623b6abc6c937be9555f07471c90e6e381'
spec=importlib.util.spec_from_file_location('private_plf_guard',G)
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
# Private instance of the pinned generic owned native runner, never bank main.
guard.JUDGE=D/'native_plf_qualification_judge_v1.py'
CAP=0.001
ORDER=[('OFF',7100136),('ON',7100136),('OFF',7100137),('ON',7100137)]
PRESERVED=('_ASTRA_COMPUTE_STATS','_UBC_STATS','_UBX_STATS','_UBX_RC_STATS')
STATS=('_PLF_STATS',)+PRESERVED+('_RQD_STATS','_UBX_STATE','_UBX_RCACHE','_UBC_RUN','_UBC_CONTEXT',
       '_UBX_CTX','_UBX_RC','_OTT_CONTEXT','_DCR_STASH','_DCR_CALLS','_DCR_OPEN')
write=guard.exclusive

def config(path,digest,grant_path):
    assert sha(path)==digest
    c=json.loads(Path(path).read_text())
    assert c['purpose']=='pair_local_fusion_qualification4'
    assert c['fixed_order']==[list(x) for x in ORDER] and c['prediction_error_cap']==CAP
    assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
    assert c['historical_parent_path']==str(O/'scout_native/astra_compute_v1_arm/candidate.py')
    assert c['candidates']['parent']['sha256']=='2aaf7aa0fd8c66ba63b895b2742a4c7640ee2c50835cd689ac9dce1f581108cb'
    assert c['historical_missing_setup_metadata_acknowledged'] is True
    assert c['both_new_OFF_required'] is True and c['fresh_native_scientific_calls_max']==64
    guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
    r=json.loads(Path(grant_path).read_text())
    assert r['approved'] is True and r['purpose']==c['purpose']
    assert r['protocol_sha256']==digest and r['runner_sha256']==sha(__file__)
    assert r['run_name']==c['run_name'] and r['numerical_slots']==1
    assert r['reuse_exception_approved'] is True and r['prediction_error_cap']==CAP
    assert r['fixed_order']==c['fixed_order']
    assert all(os.environ.get(k)=='1' for k in guard.THREADS)
    assert [x['seed'] for x in c['native_reuse']]==[7100136,7100137]
    for item in c['native_reuse']:
        for name in ('certificate','guard','prediction_npz'):
            assert sha(item[name])==item[name+'_sha256']
    return c

def prediction(path,seed):
    import numpy as np
    p=Path(path).with_name(Path(path).stem+'_pred.npz')
    with np.load(p,allow_pickle=False) as z:
        assert z.files==[str(seed)];y=z[str(seed)].copy()
    assert y.shape==(16,1024) and y.dtype==np.dtype('float32') and np.isfinite(y).all()
    return y

def row_valid(d,y,seed):
    assert d['seeds']==[seed] and len(d['rows'])==1
    r=d['rows'][0]
    assert r['seed']==seed and type(r['flops']) is int and 0<r['flops']<=2**41
    assert type(r['ops']) is int and r['ops']>0
    assert len(r['mse_layers'])==16 and all(math.isfinite(v) and v>=0 for v in r['mse_layers'])
    assert r['mse_final']==r['mse_layers'][-1]
    assert hashlib.sha256(y.tobytes()).hexdigest()[:16]==r['pred_sha']
    assert r['stats']['_ASTRA_COMPUTE_STATS']['calls']==14
    assert r['stats']['_ASTRA_COMPUTE_STATS']['saved_copy_elements']>0
    return r

def contexts(st):
    for name in ('_UBC_RUN','_UBC_CONTEXT','_UBX_CTX','_UBX_RC','_OTT_CONTEXT','_DCR_STASH'):
        assert st[name]==[None],name
    assert st['_DCR_CALLS']==[] and st['_DCR_OPEN']==[False]
    for name in ('_UBX_STATE','_UBX_RCACHE'):
        x=st[name];assert x['run'] is None and x['cache']=={} and x['bytes']==0
        if name=='_UBX_RCACHE':assert x['owner'] is None
    x=st['_UBC_STATS'];assert x['retired'] is True and x['live_cache_bytes']==x['retained_cache_bytes']==0
    assert 0<=x['max_resident_bytes']<=512000000 and 0<=x['peak_cache_bytes']<=512000000

def old_baseline(c,item):
    for name in ('certificate','guard','prediction_npz'):assert sha(item[name])==item[name+'_sha256']
    d=json.loads(Path(item['certificate']).read_text());g=json.loads(Path(item['guard']).read_text())
    assert d['sha256']==c['candidates']['parent']['sha256'] and d['candidate']==c['historical_parent_path']
    assert g['command']==item['guard_command']
    assert g['completed'] is True and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] is True and g['remaining_rows']==[]
    assert 0<g['max_rss_bytes']<7500000000 and 0<=g['max_footprint_bytes']<7500000000
    y=prediction(item['certificate'],item['seed']);r=row_valid(d,y,item['seed'])
    return dict(cert=d,row=r,pred=y,path=item['certificate'],evidence=item)

def current(c,item,role,seed):
    for key in ('certificate','guard'):assert sha(item[key])==item[key+'_sha256']
    p=Path(item['certificate']);assert sha(p.with_name(p.stem+'_pred.npz'))==item['prediction_npz_sha256']
    d=json.loads(p.read_text());g=json.loads(Path(item['guard']).read_text());source=c['candidates'][role]
    assert d['candidate']==source['path'] and d['sha256']==source['sha256'] and d['plf_mode']==source['mode']
    assert d['judge_sha256']==c['source_pins'][str(guard.JUDGE)]
    assert d['loader_sha256']==c['source_pins'][str(O/'analyst/fresh/loader.py')]
    assert d['environment']=={**{k:'1' for k in guard.THREADS},'PB_ALL':'1','CANDIDATE_SHA256':source['sha256'],'PLF_MODE':str(source['mode'])}
    expected=[str(guard.RT),'-B',str(guard.JUDGE),source['path'],str(seed),str(p)]
    assert g['command']==expected
    assert g['completed'] is True and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] is True and g['remaining_rows']==[]
    assert d['setup']['pb_wall_s']==1e9 and d['setup']['wall_s']>=0
    y=prediction(p,seed);r=row_valid(d,y,seed);contexts(r['stats'])
    truth=next(t for t in c['truth_metadata'] if t['seed']==seed)
    assert r['weights_before_sha256']==r['weights_after_sha256']==truth['weights_sha256']
    return dict(cert=d,row=r,pred=y,path=str(p),evidence=item)

def gate_OFF(base,off):
    a,b=off['row'],base['row'];counter=a['stats']['_PLF_STATS']
    checks=dict(all16_bytes=off['pred'].tobytes()==base['pred'].tobytes(),all16_MSE=a['mse_layers']==b['mse_layers'],
        scalar_MSE=a['mse_final']==b['mse_final'],prediction_hash=a['pred_sha']==b['pred_sha'],
        flops=a['flops']==b['flops'],ops=a['ops']==b['ops'],
        # Historical certificates recorded only this carried-compute dictionary;
        # other current cache metadata is anchored by each new OFF, never invented.
        carried_compute=a['stats']['_ASTRA_COMPUTE_STATS']==b['stats']['_ASTRA_COMPUTE_STATS'],
        delegation_count=counter==dict(off_calls=28,on_calls=0,Q=0,H=0,local_flop_credit=0,persistent_calls=0,gathered_calls=0))
    assert all(checks.values()),('new_OFF_historical_equivalence_failed',checks)
    return checks

def displacement_gate(base_y,arm_y,mse_layers):
    import numpy as np
    d=np.sqrt(np.mean((arm_y.astype(np.float64)-base_y.astype(np.float64))**2,axis=1))
    b=np.sqrt(np.asarray(mse_layers,dtype=np.float64))
    assert d.shape==b.shape==(16,) and np.isfinite(d).all() and np.isfinite(b).all()
    ratios=np.divide(d,b,out=np.zeros_like(d),where=b>0)
    ok=(d<=CAP*b)&((b>0)|(d==0))
    return dict(cap=CAP,all_layers_pass=bool(ok.all()),displacement_RMSE=[float(x) for x in d],
        parent_RMSE=[float(x) for x in b],ratios=[float(x) for x in ratios],zero_parent_requires_exact=True)

def gate_ON(off,arm):
    a,b=arm['row'],off['row'];s=a['stats']['_PLF_STATS'];credit=b['flops']-a['flops']
    displacement=displacement_gate(off['pred'],arm['pred'],b['mse_layers'])
    keys=('pb_single_sha256','pb_single_count','pb_wall_s','cache_counts')
    checks=dict(new_OFF_setup=all(off['cert']['setup'][k]==arm['cert']['setup'][k] for k in keys),
        layer0_bytes=off['pred'][0].tobytes()==arm['pred'][0].tobytes(),
        layer0_MSE=b['mse_layers'][0]==a['mse_layers'][0],
        carried_stats=all(a['stats'][k]==b['stats'][k] for k in PRESERVED),
        counter_schema=set(s)==set(('off_calls','on_calls','Q','H','local_flop_credit','persistent_calls','gathered_calls')),
        exact_shape_counter=s==dict(off_calls=0,on_calls=28,Q=14*2228224,H=14*10240,
             local_flop_credit=662265856,persistent_calls=14,gathered_calls=14),
        measured_positive_credit=credit>0,full_bill_counter_exact=credit==s['local_flop_credit'],
        displacement_cap=displacement['all_layers_pass'])
    report=dict(checks=checks,displacement=displacement,measured_flop_credit=credit,
        bill_change_pct=100*(a['flops']/b['flops']-1),charged_ops_delta=a['ops']-b['ops'],
        final_MSE_change_pct=(100*(a['mse_final']/b['mse_final']-1) if b['mse_final']>0 else None),
        final_log_net=(math.log(a['mse_final']/b['mse_final'])+math.log(a['flops']/b['flops']) if a['mse_final']>0 and b['mse_final']>0 else None),
        qualification_not_scientific_pass=True)
    assert all(checks.values()),('ON_qualification_failed',report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-authorized',action='store_true')
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--grant',type=Path,required=True)
    a=p.parse_args();assert a.run_authorized
    c=config(a.protocol,a.protocol_sha256,a.grant)
    run=D/c['run_name'];assert run.parent==D and run.name.startswith('run_plf_qualification_')
    run.mkdir(exist_ok=False)
    write(run/'PLAN_v1.json',dict(protocol_sha256=a.protocol_sha256,root_grant_sha256=sha(a.grant),
        controller_pid=os.getpid(),fixed_order=c['fixed_order'],expected_guards=[str(run/f'native_{seed}_{role}_guard.json') for role,seed in ORDER],
        reuse=c['native_reuse'],prediction_error_cap=CAP,no_scientific_phase=True))
    proto=dict(pinned_files=c['source_pins'],candidates=c['candidates'])
    checks=[];evaluations=[];off_by_seed={}
    try:
        baselines={item['seed']:old_baseline(c,item) for item in c['native_reuse']}
        for role,seed in ORDER:
            config(a.protocol,a.protocol_sha256,a.grant)
            env=dict(os.environ,PB_ALL='1',STATS=','.join(STATS),PLF_MODE=str(c['candidates'][role]['mode']),PYTHONDONTWRITEBYTECODE='1')
            item=guard.run_native(run,proto,role,seed,env)
            evaluations.append(dict(role=role,seed=seed,**item))
            got=current(c,item,role,seed)
            if role=='OFF':
                check=gate_OFF(baselines[seed],got);off_by_seed[seed]=got
            else:check=gate_ON(off_by_seed[seed],got)
            checks.append(dict(seed=seed,role=role,check=check))
            write(run/f'CHECK_{role}_{seed}_v1.json',checks[-1])
            print(json.dumps(dict(completed_calls=len(evaluations),total=4,role=role,seed=seed,required_gate_pass=True)),flush=True)
        write(run/'QUALIFICATION_RETIRED_v1.json',dict(status='qualification4_complete_all_guards_retired',
            protocol_sha256=a.protocol_sha256,root_grant_sha256=sha(a.grant),evaluations=evaluations,checks=checks,
            cap=CAP,all_guards_retired=True,owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,
            no_scientific_pass=True,no_followup_authorization=True,production_qualified=False))
        print(json.dumps(dict(complete=True,new_evaluations=4,all_guards_retired=True,no_followup_authorized=True)))
    except BaseException as e:
        write(run/'STOPPED_v1.json',dict(error=repr(e),checks=checks,evaluations=evaluations,
            owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,
            slot_reservation_retained=not guard.OWNED_RETIREMENT_CONFIRMED,no_retry=True))
        raise

if __name__=='__main__':main()
