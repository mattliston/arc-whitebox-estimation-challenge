"""HELD fixed fresh parent/OFF/ON136137 qualification; root grant required.
No historical reuse, scientific phase, production or submission. GTR/IOP stay ON; dense is incremental.
"""
import argparse,hashlib,importlib.util,json,math,os,signal,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
G=O/'scout_native/certify_astra_bank_copy_v2.py'
assert sha(G)=='34be6065a292e8436a3d07141b746c623b6abc6c937be9555f07471c90e6e381'
spec=importlib.util.spec_from_file_location('_private_dense_joint_guard',G)
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
H=D/'owned_native_runner_v1.py'
assert sha(H)=='aae933ceb3e05eef837ec925069ced9298a80bf9d3c5ea51543083611dba3823'
exec(compile(H.read_text(),str(H),'exec'),guard.__dict__)
guard.JUDGE=D/'native_dense_qualification_judge_HELD_v1.py'
CAP=None
ORDER=[(role,seed) for seed in (7100136,7100137) for role in ('parent','OFF','ON')]
PRESERVED=('_ASTRA_COMPUTE_STATS','_UBC_STATS','_UBX_STATS','_UBX_RC_STATS',
           '_ASTRA_UBOWN_STATS','_ASTRA_CLUSTER_STATS','_ASTRA_BCC_STATS','_ASTRA_QL_STATS','_GTR_STATS','_ASTRA_IOP_STATS','_ASZ_STATS','_HZT_STATS','_ASTRA_RLF_STATS','_TR_STATS')
STATS=PRESERVED+('_RQD_STATS','_IDC_STATS','_ASTRA_DENSE_STATS','_ASTRA_IOP_CTX','_ASTRA_IOP_CACHE','_UBX_STATE','_UBX_RCACHE','_UBC_RUN','_UBC_CONTEXT','_UBX_CTX','_UBX_RC',
       '_OTT_CONTEXT','_DCR_STASH','_DCR_CALLS','_DCR_OPEN','_ASTRA_BCC_CTX','_ASTRA_CLUSTER_CTX')
write=guard.exclusive


def config(path,digest,grant_path):
    assert sha(path)==digest
    c=json.loads(Path(path).read_text())
    assert c['status']=='REGISTERED' and c['purpose']=='dense_on_qualified_gtr_iop_exact_qualification6'
    assert c['fixed_order']==[list(x) for x in ORDER] and c['exact_all16_required'] is True
    assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
    manifest=D/'COMPOSITION_SOURCE_MANIFEST_HELD_v1.json'
    assert str(manifest) in c['source_pins']
    assert c['candidates']==json.loads(manifest.read_text())['candidates']
    assert c['candidates']['parent']==dict(path=str(D.parent/'astra_gtr_iop_joint_v1/candidate_joint_ON_v1.py'),sha256='348af4f4373756c2fa309c2f16d3a026d7d5e73c1c470e8c3894f86a4dc2dafd',mode=-1)
    assert c['native_reuse']==[] and c['new_native_calls_max']==6 and c['both_fresh_parent_and_OFF_required'] is True
    assert c['run_name']=='run_dense_joint_qualification_v1'
    assert c['qualification_scope']=='incremental_dense_only_GTR_IOP_remain_ON'
    assert c['allowed_counter_changes_ON']==['_IDC_STATS','_RQD_STATS','_ASTRA_DENSE_STATS']
    assert c['resource_policy']==dict(entry_free_percent_min=60,entry_counted_max_exclusive=8,logical_total_max=8,live_free_percent_min=50,live_counted_max=10,native_rss_footprint_max=7500000000)
    assert c['admission_wait_policy']==dict(poll_seconds=15,status_seconds=30,max_wait_seconds=1800,no_native_child_while_waiting=True,no_automatic_retry=True)
    required=(Path(__file__).resolve(),guard.JUDGE,G,H,guard.GUARD,guard.RT,guard.PY,O/'analyst/fresh/loader.py',manifest,D/'COMPOSITION_IMPORT_AUDIT_v1.json')
    assert all(str(p) in c['source_pins'] for p in required)
    guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
    r=json.loads(Path(grant_path).read_text())
    assert r['approved'] is True and r['reviewed_by']=='root' and r['purpose']==c['purpose']
    assert r['protocol_sha256']==digest and r['runner_sha256']==sha(__file__)
    assert r['run_name']==c['run_name'] and r['numerical_slots']==1 and r['logical_total_reserved']<=8
    assert r['exact_all16_required'] is True and r['fixed_order']==c['fixed_order']
    assert r['resource_policy']==c['resource_policy'] and r['admission_wait_policy']==c['admission_wait_policy']
    assert all(os.environ.get(k)=='1' for k in guard.THREADS) and os.getpriority(os.PRIO_PROCESS,0)==0
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
    assert isinstance(st['_GTR_STATS'],dict) and isinstance(st['_ASTRA_IOP_STATS'],dict)
    assert st['_ASTRA_IOP_CTX']==[None] and st['_ASTRA_IOP_CACHE']=={}
    assert st['_ASTRA_IOP_STATS']['live_entries']==0 and 0<=st['_ASTRA_IOP_STATS']['peak_entries']<=32
    for name in ('_IDC_STATS','_RQD_STATS'):
        assert set(st[name])=={'hits','misses'} and all(type(v)is int and v>=0 for v in st[name].values())
    if role=='parent':assert st['_ASTRA_DENSE_STATS'] is None
    else:assert set(st['_ASTRA_DENSE_STATS'])=={'l0_calls','cc_vv_layers'} and all(type(v)is int and v>=0 for v in st['_ASTRA_DENSE_STATS'].values())


def current(c,item,role,seed):
    import numpy as np
    for key in ('certificate','guard'):assert sha(item[key])==item[key+'_sha256']
    p=Path(item['certificate']);assert sha(p.with_name(p.stem+'_pred.npz'))==item['prediction_npz_sha256']
    d=json.loads(p.read_text());g=json.loads(Path(item['guard']).read_text());source=c['candidates'][role]
    assert d['candidate']==source['path'] and d['sha256']==source['sha256'] and d['dense_mode']==source['mode']
    assert d['judge_sha256']==c['source_pins'][str(guard.JUDGE)]
    assert d['loader_sha256']==c['source_pins'][str(O/'analyst/fresh/loader.py')]
    assert d['environment']=={**{k:'1' for k in guard.THREADS},'PB_ALL':'1','CANDIDATE_SHA256':source['sha256'],'DENSE_MODE':str(source['mode'])}
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
    counter_credit(row['stats']) # GTR and IOP stay ON in every role.
    cache_ownership(row['dense_cache_metadata'])
    truth=next(t for t in c['truth_metadata'] if t['seed']==seed)
    assert row['weights_before_sha256']==row['weights_after_sha256']==truth['weights_sha256']
    return dict(cert=d,row=row,pred=y,path=str(p),evidence=item)


def cache_ownership(m):
    assert m['post_cov32_absent'] is True and m['W5_scope_absent'] is True
    assert set(m['IDC'])=={'cc32','colsq','ott_n2'}
    for tag,entries in m['IDC'].items():
        assert 0<len(entries)<=(3 if tag=='ott_n2' else 1)
        for owner,value in entries:
            assert owner['dtype']==value['dtype']=='float32'
            assert owner['bytes']>=0 and value['bytes']>=0
            assert all(type(x)is int and x>=0 for x in owner['shape']+value['shape'])
            if tag=='cc32':assert owner['shape']==value['shape']==[1024,1024]
    assert type(m['RQD_entries'])is int and m['RQD_entries']>=0
    assert type(m['RQD_unique_array_bytes'])is int and m['RQD_unique_array_bytes']>=0

def setup_equal(a,b,parent=False):
    keys=('pb_single_sha256','pb_single_count','pb_wall_s')
    assert all(a['cert']['setup'][k]==b['cert']['setup'][k] for k in keys)
    ca,cb=a['cert']['setup']['cache_counts'],b['cert']['setup']['cache_counts']
    assert {k:v for k,v in ca.items() if k!='_ASTRA_IOP_CACHE'}=={k:v for k,v in cb.items() if k!='_ASTRA_IOP_CACHE'}
    assert ca.get('_ASTRA_IOP_CACHE',0)==cb.get('_ASTRA_IOP_CACHE',0)==0



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
        RQD_counters=a['stats']['_RQD_STATS']==b['stats']['_RQD_STATS'],
        IDC_counters=a['stats']['_IDC_STATS']==b['stats']['_IDC_STATS'],
        retained_cache_metadata=a['dense_cache_metadata']==b['dense_cache_metadata'],
        no_dense_calls=a['stats']['_ASTRA_DENSE_STATS']==dict(l0_calls=0,cc_vv_layers=0))
    assert all(checks.values()),('OFF_identity_failed',checks)
    return checks


def counter_credit(st):
    s=st['_ASTRA_IOP_STATS']
    keys={'calls','hzg','ks','ubx','hz7','saved_elements','prefix_operations_delta','cache_hits','cache_stores','cache_evictions','peak_entries','live_entries','fallback'}
    assert set(s)==keys and all(type(v) is int for v in s.values())
    assert all(v>=0 for k,v in s.items() if k!='prefix_operations_delta')
    assert s['calls']>0 and s['saved_elements']>0 and s['fallback']==0
    assert s['calls']==sum(s[k] for k in ('hzg','ks','ubx','hz7'))
    assert s['cache_hits']+s['cache_stores']==s['calls']
    assert 0<s['peak_entries']<=32 and s['live_entries']==0
    assert s['cache_evictions']<=s['cache_stores']
    g=st['_GTR_STATS']
    assert set(g)=={'calls','Q','H','credit','persistent_calls','gathered_calls'}
    assert all(type(v) is int and v>=0 for v in g.values())
    assert g['calls']>0 and g['Q']>0 and g['H']>0
    assert g['calls']==g['persistent_calls']+g['gathered_calls']
    assert g['credit']==2*g['Q']+16*g['H'] and g['credit']>0
    return s['saved_elements']+g['credit']



def gate_ON(off,arm):
    setup_equal(off,arm)
    a,b=arm['row'],off['row'];checks=exact_checks(off,arm)
    cache_ownership(a['dense_cache_metadata']);cache_ownership(b['dense_cache_metadata'])
    checks.update(positive_measured_credit=a['flops']<b['flops'],
        exact_dense_coverage=a['stats']['_ASTRA_DENSE_STATS']==dict(l0_calls=1,cc_vv_layers=15),
        same_IDC_owner_schema=a['dense_cache_metadata']['IDC']==b['dense_cache_metadata']['IDC'],
        no_RQD_retained_bytes_growth=a['dense_cache_metadata']['RQD_unique_array_bytes']<=b['dense_cache_metadata']['RQD_unique_array_bytes'])
    report=dict(checks=checks,measured_flop_credit=b['flops']-a['flops'],
        log_bill_ratio=math.log(a['flops']/b['flops']),bill_change_pct=100*(a['flops']/b['flops']-1),
        charged_ops_delta=a['ops']-b['ops'],
        preserved_GTR_credit=a['stats']['_GTR_STATS']['credit'],preserved_IOP_credit=a['stats']['_ASTRA_IOP_STATS']['saved_elements'],
        allowed_cache_counter_deltas={name:{k:a['stats'][name][k]-b['stats'][name][k] for k in ('hits','misses')} for name in ('_IDC_STATS','_RQD_STATS')},
        cache_metadata=dict(OFF=b['dense_cache_metadata'],ON=a['dense_cache_metadata']),
        runtime=dict(OFF_setup_s=off['cert']['setup']['wall_s'],ON_setup_s=arm['cert']['setup']['wall_s'],OFF_predict_s=b['predict_wall_unrounded_s'],ON_predict_s=a['predict_wall_unrounded_s'],production_wall_valid=False),
        incremental_credit_only=True,no_assumed_total_additive_credit=True)
    assert all(checks.values()),('ON_dense_qualification_failed',report)
    return report


def admit(run,role,seed):
    # A transient lack of prospective headroom is not a failed model case.
    # Hold this root reservation; launch no child until admitted, or stop once.
    started=time.monotonic();attempt=0;last_status=None
    while True:
        state=guard.resources();elapsed=time.monotonic()-started;attempt+=1
        okay=state['free_percent']>=60 and state['counted']<8 and elapsed<=1800
        record=dict(role=role,seed=seed,controller_pid=os.getpid(),attempt=attempt,
                    elapsed_seconds=elapsed,admitted=okay,**state)
        guard.event(run/'admission_wait.jsonl',record)
        if okay:
            guard.event(run/'admission.jsonl',record)
            return record
        if last_status is None or elapsed-last_status>=30:
            print(json.dumps(dict(status='waiting_for_prospective_headroom_no_native_child',
                  role=role,seed=seed,free_percent=state['free_percent'],counted=state['counted'],
                  elapsed_seconds=elapsed,slot_reserved=True)),flush=True)
            last_status=elapsed
        assert elapsed<1800, 'Bounded admission wait expired; no automatic retry'
        time.sleep(min(15,1800-elapsed))

def admissions(run):
    records=[json.loads(x)for x in (run/'admission.jsonl').read_text().splitlines()]
    assert [(x['role'],x['seed'])for x in records]==ORDER
    assert all(x['controller_pid']==os.getpid() and x['free_percent']>=60 and x['counted']<8 for x in records)
    attempts=[json.loads(x)for x in (run/'admission_wait.jsonl').read_text().splitlines()]
    assert [x for x in attempts if x['admitted']] == records
    assert all(x['admitted'] is True and 0<=x['elapsed_seconds']<=1800 for x in records)
    return dict(path=str(run/'admission.jsonl'),sha256=sha(run/'admission.jsonl'),records=records,wait_path=str(run/'admission_wait.jsonl'),wait_sha256=sha(run/'admission_wait.jsonl'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-authorized',action='store_true')
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--grant',type=Path,required=True);a=p.parse_args();assert a.run_authorized
    c=config(a.protocol,a.protocol_sha256,a.grant)
    run=D/c['run_name'];assert run.parent==D and run.name.startswith('run_dense_joint_qualification_')
    run.mkdir(exist_ok=False)
    write(run/'PLAN_v1.json',dict(protocol_path=str(a.protocol.resolve()),protocol_sha256=a.protocol_sha256,root_grant_path=str(a.grant.resolve()),root_grant_sha256=sha(a.grant),runner_sha256=sha(__file__),controller_pid=os.getpid(),resource_policy=c['resource_policy'],
          fixed_order=c['fixed_order'],expected_guards=[str(run/f'native_{seed}_{role}_guard.json') for role,seed in ORDER],
          exact_all16_required=True,reuse=[],no_scientific_phase=True))
    proto=dict(pinned_files=c['source_pins'],candidates=c['candidates']);checks=[];evaluations=[];cases={}
    try:
        for role,seed in ORDER:
            config(a.protocol,a.protocol_sha256,a.grant)
            env=dict(os.environ,PB_ALL='1',STATS=','.join(STATS),DENSE_MODE=str(c['candidates'][role]['mode']),PYTHONDONTWRITEBYTECODE='1')
            admit(run,role,seed)
            item=guard.run_native(run,proto,role,seed,env);evaluations.append(dict(role=role,seed=seed,**item))
            got=current(c,item,role,seed);cases[(role,seed)]=got
            if role=='OFF':check=gate_OFF(cases[('parent',seed)],got)
            elif role=='ON':check=gate_ON(cases[('OFF',seed)],got)
            else:check=dict(valid_fresh_parent=True)
            checks.append(dict(seed=seed,role=role,check=check));write(run/f'CHECK_{role}_{seed}_v1.json',checks[-1])
            print(json.dumps(dict(completed_calls=len(evaluations),total=6,role=role,seed=seed,required_gate_pass=True)),flush=True)
        admission_evidence=admissions(run)
        price=max(x['check']['log_bill_ratio'] for x in checks if x['role']=='ON')
        write(run/'QUALIFICATION_RETIRED_v1.json',dict(status='qualification6_complete_all_guards_retired',protocol_sha256=a.protocol_sha256,
              root_grant_sha256=sha(a.grant),plan_sha256=sha(run/'PLAN_v1.json'),admission_evidence=admission_evidence,ownership_sha256=sha(run/'ownership.jsonl'),evaluations=evaluations,checks=checks,exact_all16_required=True,all_guards_retired=True,
              owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,no_scientific_pass=True,
              no_followup_authorization=True,production_qualified=False,max_log_bill_ratio=price,incremental_dense_credit_only=True,root_controller_retirement_required=True))
    except BaseException as e:
        write(run/'STOPPED_v1.json',dict(error=repr(e),checks=checks,evaluations=evaluations,
              owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,
              slot_reservation_retained=not guard.OWNED_RETIREMENT_CONFIRMED,no_retry=True,root_controller_retirement_required=True))
        raise


def interrupted(signum,frame):raise KeyboardInterrupt('Owned dense GTR/IOP controller signal%s'%signum)
if __name__=='__main__':
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:main()
    finally:signal.signal(signal.SIGTERM,previous)
