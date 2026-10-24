"""Fixed current4e GOP2 implementation/bill qualification2. Root grant and accepted4e controls required."""
import argparse,hashlib,importlib.util,json,math,os,signal,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2];T=D.parent/'astra_terminal_central_hole_source_v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path,expected):
 assert sha(path)==expected
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load('_gop2_current_control_reader',O/'lead/runs/astra_fhm_iop_rescue_v1/finite_iop_off_qualification_HELD_v1.py','dbe70076f692318999d5894de7bd77d25797a74e3f9b00fb4bb3288a6876c697')
G=O/'scout_native/certify_astra_bank_copy_v2.py'
guard=load('_gop2_private_guard',G,'34be6065a292e8436a3d07141b746c623b6abc6c937be9555f07471c90e6e381')
H=T/'owned_native_runner_v1.py';assert sha(H)=='aae933ceb3e05eef837ec925069ced9298a80bf9d3c5ea51543083611dba3823'
exec(compile(H.read_text(),str(H),'exec'),guard.__dict__)
guard.JUDGE=O/'lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py'
ORDER=[('ON',7100136),('ON',7100137)]
STATS=old.STATS+('_GOP2_STATS','_GOP2_ACTIVE')
RESOURCE=old.RESOURCE;WAIT=old.WAIT;write=guard.exclusive
HELPERS=T/'qualification_lifecycle_helpers_v1.py'
assert sha(HELPERS)=='7a8b8e61565cb19d28b466d2fe023686363c8646bf22327a7f1a348a34a5e8fd'
exec(compile(HELPERS.read_text(),str(HELPERS),'exec'))

def config(path,digest,gpath):
 assert sha(path)==digest
 c=json.loads(Path(path).read_text());g=json.loads(Path(gpath).read_text())
 assert c['purpose']=='gop2_current4e_qualification2'and c['fixed_order']==[list(x)for x in ORDER]
 assert c['new_native_calls_max']==2 and c['run_name']=='run_gop2_current_qualification_v1'
 assert c['selection']=='none_fixed_implementation_and_bill_only'
 assert g['selection']==c['selection']
 assert c['resource_policy']==RESOURCE and c['admission_wait_policy']==WAIT
 assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 assert set(c['candidates'])=={'ON'} and c['candidates']['ON']==dict(path=str(D/'candidate_current4e_GOP2_ON_HELD_v1.py'),sha256='0170697df8866cff2bc43b13fa7a41faf5a8d73a9c527985e12ac8a29a4c89e4',mode=1)
 guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
 assert g['approved']is True and g['reviewed_by']=='root'and g['purpose']==c['purpose']
 assert g['protocol_sha256']==digest and g['runner_sha256']==sha(__file__)
 assert g['run_name']==c['run_name']and g['numerical_slots']==1 and g['fixed_order']==c['fixed_order']
 assert g['resource_policy']==RESOURCE and g['admission_wait_policy']==WAIT
 assert g['accepted_current4e_control_reuse']==c['native_reuse']
 assert all(os.environ.get(k)=='1'for k in guard.THREADS)and os.getpriority(os.PRIO_PROCESS,0)==0
 assert [x['seed']for x in c['truth_metadata']]==[7100136,7100137]
 for t in c['truth_metadata']:assert sha(t['path'])==t['sha256']
 rp=Path(c['current4e_root_price_acceptance']);a=json.loads(rp.read_text())
 assert sha(rp)==c['source_pins'][str(rp)]and a['approved']is True and a['reviewed_by']=='root'and a['exit_code']==0
 assert a['candidate_sha256']=='4e1bf080bb6751b487c62842e369a427633e0c4686f089fd3f8e6448bb416f0e'
 assert a['all_owned_PID_PPID_PGID_absent']is True and a['all16_exact']is True and a['complete_calls']==2
 assert sha(a['owner_retirement_path'])==a['owner_retirement_sha256']
 return c

def controls(c):
 bc=json.loads(Path(c['current4e_protocol']).read_text())
 assert bc['candidates']['ON']['sha256']=='4e1bf080bb6751b487c62842e369a427633e0c4686f089fd3f8e6448bb416f0e'
 assert bc['truth_metadata']==c['truth_metadata']
 return {x['seed']:old.current(bc,x,'ON',x['seed'])for x in c['native_reuse']}

def current(c,item,seed):
 for k in ('certificate','guard'):assert sha(item[k])==item[k+'_sha256']
 p=Path(item['certificate']);assert sha(p.with_name(p.stem+'_pred.npz'))==item['prediction_npz_sha256']
 d=json.loads(p.read_text());g=json.loads(Path(item['guard']).read_text());src=c['candidates']['ON']
 assert d['candidate']==src['path']and d['sha256']==src['sha256']and d['gop2_mode']==1
 assert d['judge_sha256']==c['source_pins'][str(guard.JUDGE)]and d['loader_sha256']==c['source_pins'][str(O/'analyst/fresh/loader.py')]
 assert d['environment']=={**{k:'1'for k in guard.THREADS},'PB_ALL':'1','CANDIDATE_SHA256':src['sha256'],'GOP2_MODE':'1'}
 assert g['command']==[str(guard.RT),'-B',str(guard.JUDGE),src['path'],str(seed),str(p)]
 assert g['completed']is True and g['exit_code']==0 and g['error']is None and g['whole_group_retired']is True and g['remaining_rows']==[]
 assert 0<g['max_rss_bytes']<7500000000 and 0<=g['max_footprint_bytes']<7500000000
 assert d['setup']['pb_wall_s']==1e9 and math.isfinite(d['setup']['wall_s'])and d['setup']['wall_s']>=0
 y=prediction(p,seed);assert d['seeds']==[seed]and len(d['rows'])==1
 r=d['rows'][0];assert r['seed']==seed and type(r['flops'])is int and 0<r['flops']<=2**41 and type(r['ops'])is int and r['ops']>0
 assert len(r['mse_layers'])==16 and all(math.isfinite(v)and v>0 for v in r['mse_layers'])and r['mse_final']==r['mse_layers'][-1]
 assert hashlib.sha256(y.tobytes()).hexdigest()[:16]==r['pred_sha']
 assert r['gop2_cleanup']==dict(active_clear=True,post_cov32_absent=True,hcf_ctx_clear=True,hcf_rows_clear=True,k4l_final_clear=True)
 old.contexts(r['stats'],'ON')
 assert r['stats']['_ASTRA_DENSE_STATS']==dict(cc_vv_layers=15,l0_calls=1)
 assert all(v==0 for v in r['stats']['_ASTRA_IOP_STATS'].values())
 fs=r['stats']['_ASTRA_FHM_STATS'];assert fs['retired']is True and fs['layers']==list(range(1,15))and [x[0]for x in fs['changed_rows']]==list(range(1,15))
 assert r['stats']['_GTR_STATS']['calls']>0
 assert r['stats']['_GOP2_STATS']==dict(calls=15,layers=list(range(1,16)),energy_blocks=45,message_misses=15,retired=True,value_cache_entries=0)
 assert r['stats']['_GOP2_ACTIVE']==[False]
 t=next(t for t in c['truth_metadata']if t['seed']==seed)
 assert r['weights_before_sha256']==r['weights_after_sha256']==t['weights_sha256']
 return dict(cert=d,row=r,pred=y,evidence=item)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-authorized',action='store_true');p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--grant',type=Path,required=True)
 a=p.parse_args();assert a.run_authorized;c=config(a.protocol,a.protocol_sha256,a.grant);base=controls(c)
 run=D/c['run_name'];run.mkdir(exist_ok=False)
 write(run/'PLAN_v1.json',dict(protocol_path=str(a.protocol.resolve()),protocol_sha256=a.protocol_sha256,root_grant_path=str(a.grant.resolve()),root_grant_sha256=sha(a.grant),runner_sha256=sha(__file__),controller_pid=os.getpid(),fixed_order=c['fixed_order'],expected_guards=[str(run/f'native_{seed}_{role}_guard.json')for role,seed in ORDER],native_reuse=c['native_reuse'],fixed_implementation_bill_only=True))
 proto=dict(pinned_files=c['source_pins'],candidates=c['candidates']);evaluations=[];checks=[]
 try:
  for role,seed in ORDER:
   config(a.protocol,a.protocol_sha256,a.grant);admit(run,role,seed)
   env=dict(os.environ,PB_ALL='1',GOP2_MODE='1',STATS=','.join(STATS),PYTHONDONTWRITEBYTECODE='1')
   item=guard.run_native(run,proto,role,seed,env);evaluations.append(dict(role=role,seed=seed,**item));got=current(c,item,seed);setup_equal(base[seed],got)
   assert got['cert']['python_version']==base[seed]['cert']['python_version']
   assert got['pred'][0].tobytes()==base[seed]['pred'][0].tobytes()
   b=base[seed]['row'];r=got['row'];ratio=max(r['flops']/2**41,.1)/max(b['flops']/2**41,.1)
   check=dict(seed=seed,all16_finite=True,first_layer_exact=True,input_weights_immutable=True,fifteen_routes_and_cleanup=True,setup_identical=True,actual_full_bill=r['flops'],parent_bill=b['flops'],log_score_bill_ratio=math.log(ratio),bill_change_pct=100*(r['flops']/b['flops']-1),ops_delta=r['ops']-b['ops'],no_MSE_gain_selection=True,no_scientific_pass=True)
   checks.append(check);write(run/f'CHECK_ON_{seed}_v1.json',check)
   print(json.dumps(dict(completed_calls=len(evaluations),total=2,seed=seed,implementation_pass=True)),flush=True)
  assert len(checks)==2
  write(run/'QUALIFICATION_RETIRED_v1.json',dict(status='qualification2_complete_all_guards_retired',candidate_sha256=c['candidates']['ON']['sha256'],protocol_sha256=a.protocol_sha256,root_grant_sha256=sha(a.grant),plan_sha256=sha(run/'PLAN_v1.json'),admission_evidence=admissions(run),ownership_sha256=sha(run/'ownership.jsonl'),evaluations=evaluations,checks=checks,all_guards_retired=True,owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,actual_full_bill_comparison=True,max_log_bill_ratio=max(x['log_score_bill_ratio']for x in checks),no_scientific_pass=True,no_followup_authorization=True,production_qualified=False))
 except BaseException as e:
  write(run/'STOPPED_v1.json',dict(error=repr(e),evaluations=evaluations,checks=checks,owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,slot_reservation_retained=not guard.OWNED_RETIREMENT_CONFIRMED,no_retry=True));raise

def interrupted(signum,frame):raise KeyboardInterrupt('Owned GOP2 controller signal%s'%signum)
if __name__=='__main__':
 previous=signal.signal(signal.SIGTERM,interrupted)
 try:main()
 finally:signal.signal(signal.SIGTERM,previous)
