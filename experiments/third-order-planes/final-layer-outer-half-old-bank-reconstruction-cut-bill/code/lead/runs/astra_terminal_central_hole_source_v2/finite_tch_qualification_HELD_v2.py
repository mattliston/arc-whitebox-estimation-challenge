"""HELD fixed native6 implementation/price packet. Requires a pinned root grant."""
import argparse, hashlib, importlib.util, json, math, os, signal, time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
G=O/'scout_native/certify_astra_bank_copy_v2.py'
assert sha(G)=='34be6065a292e8436a3d07141b746c623b6abc6c937be9555f07471c90e6e381'
spec=importlib.util.spec_from_file_location('_private_tch_guard',G)
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
H=D/'owned_native_runner_v1.py'
assert sha(H)=='aae933ceb3e05eef837ec925069ced9298a80bf9d3c5ea51543083611dba3823'
exec(compile(H.read_text(),str(H),'exec'),guard.__dict__)
guard.JUDGE=D/'native_tch_qualification_judge_v1.py'
ORDER=[(role,seed) for seed in (7100136,7100137) for role in ('parent','OFF','ON')]
PRESERVED=('_UBC_STATS','_UBX_STATS','_UBX_RC_STATS')
STATS=PRESERVED+('_TCH_STATS','_TCH_CTX','_UBX_STATE','_UBX_RCACHE','_UBC_RUN','_UBC_CONTEXT',
                '_UBX_CTX','_UBX_RC','_OTT_CONTEXT','_DCR_STASH','_DCR_CALLS','_DCR_OPEN')
RESOURCE=dict(entry_free_percent_min=60,entry_counted_max_exclusive=8,logical_total_max=8,
              live_free_percent_min=50,live_counted_max=10,native_rss_footprint_max=7500000000)
WAIT=dict(poll_seconds=15,status_seconds=30,max_wait_seconds=1800,
          no_native_child_while_waiting=True,no_automatic_retry=True)
write=guard.exclusive
HELPERS=D/'qualification_lifecycle_helpers_v1.py'
exec(compile(HELPERS.read_text(),str(HELPERS),'exec'))


def config(path,digest,grant_path):
    assert sha(path)==digest
    c=json.loads(Path(path).read_text());r=json.loads(Path(grant_path).read_text())
    assert c['purpose']=='terminal_central_hole_qualification6'
    assert c['fixed_order']==[list(x) for x in ORDER]
    assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
    assert c['new_native_calls_max']==6 and c['native_reuse']==[]
    assert c['qualification_selection']=='none_no_price_seed_accuracy_or_changed_count_selection'
    assert c['parent_OFF_all16_exact'] is True and c['ON_first15_exact_all16_finite'] is True
    assert c['run_name']=='run_tch_qualification_v1'
    assert c['resource_policy']==RESOURCE and c['admission_wait_policy']==WAIT
    assert c['candidates']['parent']==dict(path=str(O/'scout_native/k4liab_s51k2tw/candidate.py'),
           sha256='6a145fadb3224501e683e95424bd9ada0ff571041850c7d482969b742d8f8f41',mode=-1)
    assert c['candidates']['ON']['path']==str(D/'matched_c_tch_v2/candidate.py')
    assert c['candidates']['OFF']['path']==str(D/'matched_c_tch_OFF_v2/candidate.py')
    assert c['candidates']['ON']['mode']==1 and c['candidates']['OFF']['mode']==0
    assert set(c['candidates'])=={'parent','OFF','ON'}
    parent=Path(c['candidates']['parent']['path']).read_bytes()
    on=Path(c['candidates']['ON']['path']).read_bytes()
    off=Path(c['candidates']['OFF']['path']).read_bytes()
    assert on[:len(parent)]==parent and off==on+b'\n_TCH_ON = False\n'
    required=(Path(__file__).resolve(),guard.JUDGE,G,H,HELPERS,guard.GUARD,guard.RT,guard.PY,
              O/'analyst/fresh/loader.py',D/'SOURCE_MANIFEST_AND_PROTOCOL_v3.json',
              D/'terminal_central_hole_fragment_v2.py',D/'TINY_ALGEBRA_v2.json')
    assert all(str(p) in c['source_pins'] for p in required)
    guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
    assert r['approved'] is True and r['reviewed_by']=='root' and r['purpose']==c['purpose']
    assert r['protocol_sha256']==digest and r['runner_sha256']==sha(__file__)
    assert r['run_name']==c['run_name'] and r['numerical_slots']==1
    assert r['fixed_order']==c['fixed_order'] and r['resource_policy']==RESOURCE
    assert r['admission_wait_policy']==WAIT
    assert r['parent_OFF_all16_exact'] is True and r['ON_first15_exact_all16_finite'] is True
    assert all(os.environ.get(k)=='1' for k in guard.THREADS)
    assert os.getpriority(os.PRIO_PROCESS,0)==0
    assert [x['seed'] for x in c['truth_metadata']]==[7100136,7100137]
    for t in c['truth_metadata']:assert sha(t['path'])==t['sha256']
    return c


def contexts(st,role):
    for name in ('_UBC_RUN','_UBC_CONTEXT','_UBX_CTX','_UBX_RC','_OTT_CONTEXT','_DCR_STASH'):
        assert st[name]==[None],name
    assert st['_DCR_CALLS']==[] and st['_DCR_OPEN']==[False]
    for name in ('_UBX_STATE','_UBX_RCACHE'):
        x=st[name];assert x['run'] is None and x['cache']=={} and x['bytes']==0
        if name=='_UBX_RCACHE':assert x['owner'] is None
    x=st['_UBC_STATS'];assert x['retired'] is True and x['live_cache_bytes']==x['retained_cache_bytes']==0
    assert 0<=x['max_resident_bytes']<=512000000 and 0<=x['peak_cache_bytes']<=512000000
    if role=='parent':assert st['_TCH_STATS'] is None and st['_TCH_CTX'] is None
    else:assert isinstance(st['_TCH_STATS'],dict) and st['_TCH_CTX']==[None]


def current(c,item,role,seed):
    for key in ('certificate','guard'):assert sha(item[key])==item[key+'_sha256']
    p=Path(item['certificate']);assert sha(p.with_name(p.stem+'_pred.npz'))==item['prediction_npz_sha256']
    d=json.loads(p.read_text());g=json.loads(Path(item['guard']).read_text());source=c['candidates'][role]
    assert d['candidate']==source['path'] and d['sha256']==source['sha256'] and d['tch_mode']==source['mode']
    assert d['judge_sha256']==c['source_pins'][str(guard.JUDGE)]
    assert d['loader_sha256']==c['source_pins'][str(O/'analyst/fresh/loader.py')]
    assert d['environment']=={**{k:'1' for k in guard.THREADS},'PB_ALL':'1',
                             'CANDIDATE_SHA256':source['sha256'],'TCH_MODE':str(source['mode'])}
    assert g['command']==[str(guard.RT),'-B',str(guard.JUDGE),source['path'],str(seed),str(p)]
    assert g['completed'] is True and g['exit_code']==0 and g['error'] is None
    assert g['whole_group_retired'] is True and g['remaining_rows']==[]
    assert 0<g['max_rss_bytes']<7500000000 and 0<=g['max_footprint_bytes']<7500000000
    assert d['setup']['pb_wall_s']==1e9 and math.isfinite(d['setup']['wall_s']) and d['setup']['wall_s']>=0
    y=prediction(p,seed);assert d['seeds']==[seed] and len(d['rows'])==1
    row=d['rows'][0];assert row['seed']==seed
    assert type(row['flops']) is int and .1*2**41<=row['flops']<=2**41
    assert type(row['ops']) is int and row['ops']>0
    assert len(row['mse_layers'])==16 and all(math.isfinite(x) and x>0 for x in row['mse_layers'])
    assert row['mse_final']==row['mse_layers'][-1]
    assert hashlib.sha256(y.tobytes()).hexdigest()[:16]==row['pred_sha']
    assert row['tch_cleanup']==dict(ctx_clear=True,hcf_ctx_clear=True,hcf_rows_clear=True,k4l_final_clear=True)
    contexts(row['stats'],role)
    truth=next(t for t in c['truth_metadata'] if t['seed']==seed)
    assert row['weights_before_sha256']==row['weights_after_sha256']==truth['weights_sha256']
    return dict(cert=d,row=row,pred=y,path=str(p),evidence=item)


def gate_OFF(base,off):
    setup_equal(base,off)
    a,b=off['row'],base['row']
    checks=dict(all16_bytes=off['pred'].tobytes()==base['pred'].tobytes(),
                all16_MSE=a['mse_layers']==b['mse_layers'],flops=a['flops']==b['flops'],ops=a['ops']==b['ops'],
                prediction_hash=a['pred_sha']==b['pred_sha'],
                carried_stats=all(a['stats'][k]==b['stats'][k] for k in PRESERVED),
                routes_empty=a['routes']==[],
                arm_counters_zero=all(type(v) is int and v==0 for v in a['stats']['_TCH_STATS'].values()))
    assert all(checks.values()),('OFF_identity_failed',checks)
    return checks


def gate_ON(off,arm):
    setup_equal(off,arm)
    a,b=arm['row'],off['row'];st=a['stats']['_TCH_STATS'];routes=a['routes']
    checks=dict(first15_bytes=arm['pred'][:15].tobytes()==off['pred'][:15].tobytes(),
                first15_MSE=a['mse_layers'][:15]==b['mse_layers'][:15],
                counter_types=all(type(v) is int and v>=0 for v in st.values()))
    if st['fallback_small']:
        checks.update(small_fallback=st['fallback_small']==1 and st['calls']==0 and routes==[])
    else:
        m=st['rows'];k=min(m,max(64,((m+127)//128)*64))
        traces=[x for x in routes if x['kind']=='trace'];hcf=[x for x in routes if x['kind']=='hcf']
        checks.update(fixed_counts=st['calls']==1 and st['kept']==k and st['omitted']==m-k,
                      hcf_once=st['hcf_proxy_calls']==1 and hcf==[dict(kind='hcf',rows=m,keep=k,tail=m-k)],
                      trace_count=st['trace_blocks']==len(traces) and len(traces) in (1,2),
                      trace_layouts=all(x['U_shape'][0]==m and x['tail']==m-k
                          and x['U_shape'][1]==x['Fv_shape'][0] and x['Fu_shape'][0]==x['ru']
                          and x['Fu_shape'][1]==x['Fv_shape'][1] and x['lam_shape']==[2*x['Fu_shape'][1]]
                          and sum(2*width for start,width in x['blocks'])==x['lam_shape'][0] for x in traces))
    report=dict(checks=checks,routes=routes,stats=st,
                log_bill_ratio=math.log(a['flops']/b['flops']),bill_change_pct=100*(a['flops']/b['flops']-1),
                charged_ops_delta=a['ops']-b['ops'],measured_full_bill=True,
                price_seed_MSE_gain_not_compared=True,no_scientific_pass=True,
                changed_prediction_presence_not_a_gate=True)
    assert all(checks.values()),('ON_TCH_implementation_failed',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-authorized',action='store_true')
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--grant',type=Path,required=True);a=p.parse_args();assert a.run_authorized
    c=config(a.protocol,a.protocol_sha256,a.grant)
    run=D/c['run_name'];assert run.parent==D and run.name=='run_tch_qualification_v1'
    run.mkdir(exist_ok=False)
    write(run/'PLAN_v1.json',dict(protocol_path=str(a.protocol.resolve()),protocol_sha256=a.protocol_sha256,
          root_grant_path=str(a.grant.resolve()),root_grant_sha256=sha(a.grant),runner_sha256=sha(__file__),
          controller_pid=os.getpid(),resource_policy=RESOURCE,fixed_order=c['fixed_order'],
          expected_guards=[str(run/f'native_{seed}_{role}_guard.json') for role,seed in ORDER],
          parent_OFF_all16_exact=True,ON_first15_exact_all16_finite=True,reuse=[],no_scientific_phase=True))
    proto=dict(pinned_files=c['source_pins'],candidates=c['candidates']);checks=[];evaluations=[];cases={}
    try:
        for role,seed in ORDER:
            config(a.protocol,a.protocol_sha256,a.grant)
            env=dict(os.environ,PB_ALL='1',STATS=','.join(STATS),TCH_MODE=str(c['candidates'][role]['mode']),PYTHONDONTWRITEBYTECODE='1')
            admit(run,role,seed)
            item=guard.run_native(run,proto,role,seed,env);evaluations.append(dict(role=role,seed=seed,**item))
            got=current(c,item,role,seed);cases[(role,seed)]=got
            if role=='OFF':check=gate_OFF(cases[('parent',seed)],got)
            elif role=='ON':check=gate_ON(cases[('OFF',seed)],got)
            else:check=dict(valid_fresh_parent=True)
            checks.append(dict(seed=seed,role=role,check=check));write(run/f'CHECK_{role}_{seed}_v1.json',checks[-1])
            print(json.dumps(dict(completed_calls=len(evaluations),total=6,role=role,seed=seed,required_gate_pass=True)),flush=True)
        write(run/'QUALIFICATION_RETIRED_v1.json',dict(status='qualification6_complete_all_guards_retired',
              protocol_sha256=a.protocol_sha256,root_grant_sha256=sha(a.grant),plan_sha256=sha(run/'PLAN_v1.json'),
              admission_evidence=admissions(run),ownership_sha256=sha(run/'ownership.jsonl'),
              evaluations=evaluations,checks=checks,all_guards_retired=True,
              owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,no_scientific_pass=True,
              no_followup_authorization=True,production_qualified=False,
              max_log_bill_ratio=max(x['check']['log_bill_ratio'] for x in checks if x['role']=='ON'),
              price_seed_MSE_gain_not_compared=True,root_controller_retirement_and_price_freeze_required=True))
    except BaseException as e:
        write(run/'STOPPED_v1.json',dict(error=repr(e),checks=checks,evaluations=evaluations,
              owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,
              slot_reservation_retained=not guard.OWNED_RETIREMENT_CONFIRMED,no_retry=True,
              root_controller_retirement_required=True))
        raise


def interrupted(signum,frame):raise KeyboardInterrupt('Owned TCH controller signal%s'%signum)
if __name__=='__main__':
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:main()
    finally:signal.signal(signal.SIGTERM,previous)
