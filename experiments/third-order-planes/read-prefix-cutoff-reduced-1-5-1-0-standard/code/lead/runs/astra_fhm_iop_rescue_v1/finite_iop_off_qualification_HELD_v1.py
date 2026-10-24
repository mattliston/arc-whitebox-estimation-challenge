"""HELD fixed native2 IOP-OFF implementation/price packet. Requires a pinned root grant."""
import argparse, hashlib, importlib.util, json, math, os, signal, time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
G=O/'scout_native/certify_astra_bank_copy_v2.py'
assert sha(G)=='34be6065a292e8436a3d07141b746c623b6abc6c937be9555f07471c90e6e381'
spec=importlib.util.spec_from_file_location('_private_ivd_guard',G)
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
I=O/'lead/runs/astra_iop_view_dedup_v1'
H=I/'owned_native_runner_v1.py'
assert sha(H)=='aae933ceb3e05eef837ec925069ced9298a80bf9d3c5ea51543083611dba3823'
exec(compile(H.read_text(),str(H),'exec'),guard.__dict__)
guard.JUDGE=I/'native_ivd_qualification_judge_v2.py'
ORDER=[('ON',seed) for seed in (7100136,7100137)]
PRESERVED=('_UBC_STATS','_UBX_STATS','_UBX_RC_STATS','_ASTRA_COMPUTE_STATS',
           '_ASTRA_UBOWN_STATS','_ASTRA_CLUSTER_STATS','_ASTRA_BCC_STATS','_ASTRA_QL_STATS',
           '_RQD_STATS','_ASZ_STATS','_HZT_STATS','_ASTRA_RLF_STATS','_ASTRA_RLS_STATS',
           '_TR_STATS','_ASTRA_IOP_STATS','_GTR_STATS','_ASTRA_DENSE_STATS','_ASTRA_FHM_STATS')
STATS=PRESERVED+('_ASTRA_IVD_STATS','_ASTRA_IOP_CTX','_ASTRA_IOP_CACHE','_UBX_STATE','_UBX_RCACHE',
    '_UBC_RUN','_UBC_CONTEXT','_UBX_CTX','_UBX_RC','_OTT_CONTEXT','_DCR_STASH','_DCR_CALLS','_DCR_OPEN',
    '_ASTRA_BCC_CTX','_ASTRA_CLUSTER_CTX')
RESOURCE=dict(entry_free_percent_min=60,entry_counted_max_exclusive=8,logical_total_max=8,
              live_free_percent_min=50,live_counted_max=10,native_rss_footprint_max=7500000000)
WAIT=dict(poll_seconds=15,status_seconds=30,max_wait_seconds=1800,
          no_native_child_while_waiting=True,no_automatic_retry=True)
write=guard.exclusive
HELPERS=I/'qualification_lifecycle_helpers_v1.py'
exec(compile(HELPERS.read_text(),str(HELPERS),'exec'))


def config(path,digest,grant_path):
    assert sha(path)==digest
    c=json.loads(Path(path).read_text());r=json.loads(Path(grant_path).read_text())
    assert c['purpose']=='iop_off_qualification2'
    assert c['fixed_order']==[list(x) for x in ORDER]
    assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
    assert c['new_native_calls_max']==2 and c['reused_parent_calls']==2
    assert c['qualification_selection']=='none_no_price_seed_accuracy_or_changed_count_selection'
    assert c['all16_exact'] is True and c['IOP_counters_zero'] is True
    assert c['run_name']=='run_iop_off_qualification_v1'
    assert c['resource_policy']==RESOURCE and c['admission_wait_policy']==WAIT
    assert c['candidates']['parent']==dict(path=str(O/'lead/runs/astra_fhm_dense_delivery_v1/candidate_fhm_dense_ON_HELD_v1.py'),
           sha256='6f384b0942b2813e42f4094ea2c26a2777fb76a24395540aa3f698c0f76330f1',mode=-1)
    assert c['candidates']['ON']==dict(path=str(D/'candidate_fhm_dense_IOP_OFF_HELD_v1.py'),
           sha256='4e1bf080bb6751b487c62842e369a427633e0c4686f089fd3f8e6448bb416f0e',mode=-1)
    assert set(c['candidates'])=={'parent','ON'}
    parent=Path(c['candidates']['parent']['path']).read_bytes()
    on=Path(c['candidates']['ON']['path']).read_bytes()
    assert on==parent+b'\n\n'+(D/'fragment_IOP_OFF_HELD_v1.py').read_bytes()
    required=(Path(__file__).resolve(),guard.JUDGE,G,H,HELPERS,guard.GUARD,guard.RT,guard.PY,
              O/'analyst/fresh/loader.py',D/'IOP_OFF_SOURCE_AND_OPERATIONAL_PROOF_v1.json',
              D/'fragment_IOP_OFF_HELD_v1.py')
    assert all(str(p) in c['source_pins'] for p in required)
    guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
    reuse=c['parent_reuse'];assert reuse['roles_seeds']==[['parent',7100136],['parent',7100137]]
    a=json.loads(Path(reuse['acceptance_path']).read_text());q=json.loads(Path(reuse['retirement_path']).read_text())
    done=json.loads(Path(reuse['complete_path']).read_text())
    for pkey,hkey in [('acceptance_path','acceptance_sha256'),('retirement_path','retirement_sha256'),('complete_path','complete_sha256')]:
        assert sha(reuse[pkey])==reuse[hkey] and c['source_pins'][reuse[pkey]]==reuse[hkey]
    assert a['approved'] is True and a['reviewed_by']=='root' and a['complete_calls']==6
    assert a['all_fixed_qualification_checks_recomputed'] and a['OFF_all16_exact'] and a['ON_all16_exact']
    assert a['root_retirement_path']==reuse['retirement_path'] and a['root_retirement_sha256']==reuse['retirement_sha256']
    assert a['retirement_path']==reuse['complete_path'] and a['retirement_sha256']==reuse['complete_sha256']
    assert q['approved'] and q['reviewed_by']=='root' and q['actual_managed_exit_observed'] and q['exit_code']==0
    assert q['all_owned_PID_PPID_PGID_absent'] and done['all_guards_retired'] and done['owned_group_retirement_verified']
    expected=[x for x in done['evaluations'] if x['role']=='parent']
    assert c['native_reuse']==expected and [[x['role'],x['seed']] for x in expected]==reuse['roles_seeds']
    for x in expected:
        for k in ('certificate','guard'):assert c['source_pins'][x[k]]==x[k+'_sha256']
        pp=Path(x['certificate']);pp=pp.with_name(pp.stem+'_pred.npz')
        assert c['source_pins'][str(pp)]==x['prediction_npz_sha256']
    assert r['approved'] is True and r['reviewed_by']=='root' and r['purpose']==c['purpose']
    assert r['protocol_sha256']==digest and r['runner_sha256']==sha(__file__)
    assert r['run_name']==c['run_name'] and r['numerical_slots']==1
    assert r['fixed_order']==c['fixed_order'] and r['resource_policy']==RESOURCE
    assert r['admission_wait_policy']==WAIT and r['all16_exact'] is True and r['IOP_counters_zero'] is True
    assert r['parent_reuse_approved'] is True and r['parent_reuse']==reuse
    assert all(os.environ.get(k)=='1' for k in guard.THREADS)
    assert os.getpriority(os.PRIO_PROCESS,0)==0
    assert [x['seed'] for x in c['truth_metadata']]==[7100136,7100137]
    for t in c['truth_metadata']:assert sha(t['path'])==t['sha256']
    return c


def contexts(st,role):
    for name in ('_UBC_RUN','_UBC_CONTEXT','_UBX_CTX','_UBX_RC','_OTT_CONTEXT','_DCR_STASH','_ASTRA_BCC_CTX','_ASTRA_CLUSTER_CTX'):
        assert st[name]==[None],name
    assert st['_DCR_CALLS']==[] and st['_DCR_OPEN']==[False]
    for name in ('_UBX_STATE','_UBX_RCACHE'):
        x=st[name];assert x['run'] is None and x['cache']=={} and x['bytes']==0
        if name=='_UBX_RCACHE':assert x['owner'] is None
    x=st['_UBC_STATS'];assert x['retired'] is True and x['live_cache_bytes']==x['retained_cache_bytes']==0
    assert 0<=x['max_resident_bytes']<=512000000 and 0<=x['peak_cache_bytes']<=512000000
    assert isinstance(st['_ASTRA_IOP_STATS'],dict) and st['_ASTRA_IOP_CTX']==[None] and st['_ASTRA_IOP_CACHE']=={}
    assert st['_ASTRA_IOP_STATS']['live_entries']==0 and 0<=st['_ASTRA_IOP_STATS']['peak_entries']<=32
    assert st['_ASTRA_IVD_STATS'] is None


def current(c,item,role,seed):
    for key in ('certificate','guard'):assert sha(item[key])==item[key+'_sha256']
    p=Path(item['certificate']);assert sha(p.with_name(p.stem+'_pred.npz'))==item['prediction_npz_sha256']
    d=json.loads(p.read_text());g=json.loads(Path(item['guard']).read_text());source=c['candidates'][role]
    assert d['candidate']==source['path'] and d['sha256']==source['sha256'] and d['ivd_mode']==source['mode']
    assert d['judge_sha256']==c['source_pins'][str(guard.JUDGE)]
    assert d['loader_sha256']==c['source_pins'][str(O/'analyst/fresh/loader.py')]
    assert d['environment']=={**{k:'1' for k in guard.THREADS},'PB_ALL':'1',
                             'CANDIDATE_SHA256':source['sha256'],'IVD_MODE':str(source['mode'])}
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
    assert row['ivd_cleanup']==dict(iop_ctx_clear=True,iop_cache_empty=True,hcf_ctx_clear=True,hcf_rows_clear=True,k4l_final_clear=True)
    contexts(row['stats'],role)
    truth=next(t for t in c['truth_metadata'] if t['seed']==seed)
    assert row['weights_before_sha256']==row['weights_after_sha256']==truth['weights_sha256']
    return dict(cert=d,row=row,pred=y,path=str(p),evidence=item)


def reused_parent(c,seed):
    item=next(x for x in c['native_reuse'] if x['role']=='parent' and x['seed']==seed)
    return current(c,item,'parent',seed)


def gate_ON(base,arm):
    setup_equal(base,arm)
    a,b=arm['row'],base['row'];iop=b['stats']['_ASTRA_IOP_STATS']
    az=a['stats']['_ASTRA_IOP_STATS']
    checks=dict(all16_bytes=arm['pred'].tobytes()==base['pred'].tobytes(),
        all16_MSE=a['mse_layers']==b['mse_layers'],scalar_MSE=a['mse_final']==b['mse_final'],
        prediction_hash=a['pred_sha']==b['pred_sha'],
        carried_stats=all(a['stats'][k]==b['stats'][k] for k in PRESERVED if k!='_ASTRA_IOP_STATS'),
        IOP_counters_zero=set(az)==set(iop) and all(type(x) is int and x==0 for x in az.values()),
        no_IVD_stats=a['stats']['_ASTRA_IVD_STATS'] is None,
        restored_bill=a['flops']==b['flops']+iop['saved_elements'],
        restored_charged_ops=a['ops']==b['ops']-iop['prefix_operations_delta'])
    report=dict(checks=checks,log_bill_ratio=math.log(a['flops']/b['flops']),
        bill_change_pct=100*(a['flops']/b['flops']-1),charged_ops_delta=a['ops']-b['ops'],
        measured_full_bill=True,price_seed_MSE_gain_not_compared=True,no_scientific_pass=True,
        production_request_counts_unqualified=True)
    assert all(checks.values()),('IOP_OFF_implementation_failed',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-authorized',action='store_true')
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--grant',type=Path,required=True);a=p.parse_args();assert a.run_authorized
    c=config(a.protocol,a.protocol_sha256,a.grant)
    run=D/c['run_name'];assert run.parent==D and run.name=='run_iop_off_qualification_v1'
    run.mkdir(exist_ok=False)
    write(run/'PLAN_v1.json',dict(protocol_path=str(a.protocol.resolve()),protocol_sha256=a.protocol_sha256,
          root_grant_path=str(a.grant.resolve()),root_grant_sha256=sha(a.grant),runner_sha256=sha(__file__),
          controller_pid=os.getpid(),resource_policy=RESOURCE,fixed_order=c['fixed_order'],
          expected_guards=[str(run/f'native_{seed}_{role}_guard.json') for role,seed in ORDER],
          all16_exact=True,IOP_counters_zero=True,reuse=c['parent_reuse'],no_scientific_phase=True))
    proto=dict(pinned_files=c['source_pins'],candidates=c['candidates']);checks=[];evaluations=[];cases={}
    try:
        for role,seed in ORDER:
            config(a.protocol,a.protocol_sha256,a.grant)
            env=dict(os.environ,PB_ALL='1',STATS=','.join(STATS),IVD_MODE=str(c['candidates'][role]['mode']),PYTHONDONTWRITEBYTECODE='1')
            admit(run,role,seed)
            item=guard.run_native(run,proto,role,seed,env);evaluations.append(dict(role=role,seed=seed,**item))
            got=current(c,item,role,seed);check=gate_ON(reused_parent(c,seed),got)
            checks.append(dict(seed=seed,role=role,check=check));write(run/f'CHECK_{role}_{seed}_v1.json',checks[-1])
            print(json.dumps(dict(completed_calls=len(evaluations),total=2,role=role,seed=seed,required_gate_pass=True)),flush=True)
        write(run/'QUALIFICATION_RETIRED_v1.json',dict(status='qualification2_complete_all_guards_retired',
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


def interrupted(signum,frame):raise KeyboardInterrupt('Owned IOP-OFF controller signal%s'%signum)
if __name__=='__main__':
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:main()
    finally:signal.signal(signal.SIGTERM,previous)
