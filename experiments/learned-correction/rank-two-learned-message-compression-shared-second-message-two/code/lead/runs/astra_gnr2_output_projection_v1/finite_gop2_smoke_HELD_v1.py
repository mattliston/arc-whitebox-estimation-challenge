"""Fixed ON136/137 gross accuracy/bill smoke. Root grant and TCH control reuse required."""
import argparse,hashlib,importlib.util,json,math,os,signal,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2];T=D.parent/'astra_terminal_central_hole_source_v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path,expected):
 assert sha(path)==expected
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load('_gop2_tch_control_reader',T/'finite_tch_qualification_HELD_v3.py','ffdd0407bde3e57f95d75fe1f37782ffde0f83437ef751cd7bb4c502df1f61fd')
G=O/'scout_native/certify_astra_bank_copy_v2.py'
guard=load('_gop2_private_guard',G,'34be6065a292e8436a3d07141b746c623b6abc6c937be9555f07471c90e6e381')
H=T/'owned_native_runner_v1.py';assert sha(H)=='aae933ceb3e05eef837ec925069ced9298a80bf9d3c5ea51543083611dba3823'
exec(compile(H.read_text(),str(H),'exec'),guard.__dict__)
guard.JUDGE=D/'native_gop2_probe_judge_v1.py'
ORDER=[('ON',7100136),('ON',7100137)]
STATS=old.STATS+('_GOP2_STATS','_GOP2_ACTIVE')
RESOURCE=old.RESOURCE;WAIT=old.WAIT;write=guard.exclusive
HELPERS=T/'qualification_lifecycle_helpers_v1.py'
assert sha(HELPERS)=='7a8b8e61565cb19d28b466d2fe023686363c8646bf22327a7f1a348a34a5e8fd'
exec(compile(HELPERS.read_text(),str(HELPERS),'exec'))

def config(path,digest,gpath):
 assert sha(path)==digest
 c=json.loads(Path(path).read_text());g=json.loads(Path(gpath).read_text())
 assert c['purpose']=='gnn_output_projection_rank2_native_smoke2'and c['fixed_order']==[list(x)for x in ORDER]
 assert c['new_native_calls_max']==2 and c['run_name']=='run_gop2_smoke_v1'
 assert c['selection']==dict(first_seed_early_reject_adjusted_ratio_ge=1.02,complete_second_unless_early_reject=True,two_seed_geometric_adjusted_ratio_lt=1.0,diagnostic_only_not_gain_acceptance=True)
 assert g['selection']==c['selection']
 assert c['resource_policy']==RESOURCE and c['admission_wait_policy']==WAIT
 assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 assert set(c['candidates'])=={'ON'} and c['candidates']['ON']==dict(path=str(D/'matchedC_ON_UNVALIDATED_v1.py'),sha256='57d68548b90aff0adddf0337a316b241f1c82345405fede92fe0192254686150',mode=1)
 guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
 assert g['approved']is True and g['reviewed_by']=='root'and g['purpose']==c['purpose']
 assert g['protocol_sha256']==digest and g['runner_sha256']==sha(__file__)
 assert g['run_name']==c['run_name']and g['numerical_slots']==1 and g['fixed_order']==c['fixed_order']
 assert g['resource_policy']==RESOURCE and g['admission_wait_policy']==WAIT
 assert g['accepted_TCH_control_reuse']==c['native_reuse']
 assert all(os.environ.get(k)=='1'for k in guard.THREADS)and os.getpriority(os.PRIO_PROCESS,0)==0
 assert [x['seed']for x in c['truth_metadata']]==[7100136,7100137]
 for t in c['truth_metadata']:assert sha(t['path'])==t['sha256']
 for p in (c['TCH_root_price_acceptance'],c['TCH_root_retirement']):
  a=json.loads(Path(p).read_text());assert a['approved']is True and a['reviewed_by']=='root'and a['exit_code']==0
 return c

def controls(c):
 tc=json.loads(Path(c['TCH_registered_protocol']).read_text())
 assert tc['candidates']['parent']['sha256']=='6a145fadb3224501e683e95424bd9ada0ff571041850c7d482969b742d8f8f41'
 assert tc['truth_metadata']==c['truth_metadata']
 return {x['seed']:old.current(tc,x,'parent',x['seed'])for x in c['native_reuse']}

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
 old.contexts(r['stats'],'parent')
 assert r['stats']['_GOP2_STATS']==dict(calls=15,layers=list(range(1,16)),energy_blocks=45,message_misses=15,retired=True,value_cache_entries=0)
 assert r['stats']['_GOP2_ACTIVE']==[False]
 t=next(t for t in c['truth_metadata']if t['seed']==seed)
 assert r['weights_before_sha256']==r['weights_after_sha256']==t['weights_sha256']
 return dict(cert=d,row=r,pred=y,evidence=item)

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-authorized',action='store_true');p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--grant',type=Path,required=True)
 a=p.parse_args();assert a.run_authorized;c=config(a.protocol,a.protocol_sha256,a.grant);base=controls(c)
 run=D/c['run_name'];run.mkdir(exist_ok=False)
 write(run/'PLAN_v1.json',dict(protocol_path=str(a.protocol.resolve()),protocol_sha256=a.protocol_sha256,root_grant_path=str(a.grant.resolve()),root_grant_sha256=sha(a.grant),runner_sha256=sha(__file__),controller_pid=os.getpid(),fixed_order=c['fixed_order'],expected_guards=[str(run/f'native_{seed}_{role}_guard.json')for role,seed in ORDER],native_reuse=c['native_reuse'],fixed_gross_accuracy_diagnostic=True))
 proto=dict(pinned_files=c['source_pins'],candidates=c['candidates']);evaluations=[];checks=[]
 try:
  for role,seed in ORDER:
   config(a.protocol,a.protocol_sha256,a.grant);admit(run,role,seed)
   env=dict(os.environ,PB_ALL='1',GOP2_MODE='1',STATS=','.join(STATS),PYTHONDONTWRITEBYTECODE='1')
   item=guard.run_native(run,proto,role,seed,env);evaluations.append(dict(role=role,seed=seed,**item));got=current(c,item,seed);setup_equal(base[seed],got)
   assert got['cert']['python_version']==base[seed]['cert']['python_version']
   assert got['pred'][0].tobytes()==base[seed]['pred'][0].tobytes()
   b=base[seed]['row'];r=got['row'];ratio=max(r['flops']/2**41,.1)/max(b['flops']/2**41,.1)
   check=dict(seed=seed,all16_finite=True,first_layer_exact=True,input_weights_immutable=True,fifteen_routes_and_cleanup=True,setup_identical=True,actual_full_bill=r['flops'],parent_bill=b['flops'],log_score_bill_ratio=math.log(ratio),bill_change_pct=100*(r['flops']/b['flops']-1),ops_delta=r['ops']-b['ops'],mse_ratio=r['mse_final']/b['mse_final'],adjusted_ratio=(r['mse_final']/b['mse_final'])*ratio,log_adjusted_ratio=math.log(r['mse_final']/b['mse_final'])+math.log(ratio),diagnostic_only=True,no_scientific_pass=True)
   checks.append(check);write(run/f'CHECK_ON_{seed}_v1.json',check)
   print(json.dumps(dict(completed_calls=len(evaluations),total=2,seed=seed,implementation_pass=True)),flush=True)
   if seed==7100136 and check['adjusted_ratio']>=1.02:
    break
  early_reject=len(checks)==1
  assert not early_reject or checks[0]['adjusted_ratio']>=1.02
  mean_log=sum(x['log_adjusted_ratio']for x in checks)/len(checks)
  diagnostic=dict(status=('EARLY_REJECT_136_GE_2PCT'if early_reject else ('PASS_TO_REQUEST_INDEPENDENT_FRESH'if mean_log<0 else 'REJECT_TWO_SEED_GEOMEAN')),complete_calls=len(checks),fixed_two_seed_complete=not early_reject,mean_log_adjusted_ratio=mean_log,geometric_adjusted_change_pct=100*math.expm1(mean_log),eligible_to_request_fresh=(not early_reject and mean_log<0),gain_accepted=False,no_followup_authorization=True)
  write(run/'DIAGNOSTIC_DECISION_v1.json',diagnostic)
  write(run/'PROBE_RETIRED_v1.json',dict(status='fixed_smoke_owned_guards_retired_root_actualexit_required',diagnostic=diagnostic,protocol_sha256=a.protocol_sha256,root_grant_sha256=sha(a.grant),plan_sha256=sha(run/'PLAN_v1.json'),admission_evidence=admissions(run),ownership_sha256=sha(run/'ownership.jsonl'),evaluations=evaluations,checks=checks,all_guards_retired=True,owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,no_scientific_pass=True,no_followup_authorization=True))
 except BaseException as e:
  write(run/'STOPPED_v1.json',dict(error=repr(e),evaluations=evaluations,checks=checks,owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,slot_reservation_retained=not guard.OWNED_RETIREMENT_CONFIRMED,no_retry=True));raise

def interrupted(signum,frame):raise KeyboardInterrupt('Owned GOP2 controller signal%s'%signum)
if __name__=='__main__':
 previous=signal.signal(signal.SIGTERM,interrupted)
 try:main()
 finally:signal.signal(signal.SIGTERM,previous)
