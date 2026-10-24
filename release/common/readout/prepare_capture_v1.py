"""Create frozen six-source own first128 capture plans; no prediction or truth read."""
import argparse, ast, hashlib, json, os, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=HERE.parent;ORIG=D/'truth_response95_v1'
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
A=D.parent/'fable_oct19_kc_v1/agent_astra_joint_v1';PORT=A/'cloud_linux_capacity_v7';L=A/'cloud_linux_markline_v1'
sys.path.insert(0,str(PORT));from common_linux_v7 import thread_env,verified
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
PREREG='42c4f80e65348f149e4b64d8e91da29f0fb1995abb5a78f8004fe0623a4a5bd9'
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
HOOK='f7e2bf4da1ff67ae3249f70accfbc0bd5419135ec951b233ba4a7620dc6ee229'
HARNESS={'client_capture_v1.py':'b08da90f4ca07e4ea5ac81350152e305473784ab968d451a0b1275d40ba747d0',
 'run_group_capture_v1.py':'8039f47d19ac6b971e4db77aab127458b1e62c6d57a2617145791526c68500b5',
 'run_capture_v1.py':'8e6a23c406c039dfeff34f8b6b135b03c6cdd40d05f586f7b6a9dee9bdb83aca'}
def rec(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
def write(p,x):
 with p.open('x') as out:json.dump(x,out,indent=2,allow_nan=False);out.write('\n')
def clean_guard(path):
 g=read(path)
 assert g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors']
 assert g['guard_sha256']==GUARD and g['thread_provenance']['blas_threads']==1
 assert g['max_owned_swap_bytes']==0 and all(v==0 for v in g['cgroup_events_final'].values())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha256',required=True);args=ap.parse_args()
 assert sys.platform=='linux' and sha(HERE/'PREREG_v1.json')==PREREG
 review=read(HERE/'ROOT_PREPARE_REVIEW_v1.json')
 assert review['successful'] and review['one_prepare_authorized'] and review['script_sha256']==sha(__file__)
 manifest_path=HERE/'PARENT_SOURCES_v1.json';assert sha(manifest_path)==args.manifest_sha256==review['source_manifest_sha256']
 manifest=read(manifest_path);p=read(HERE/'PREREG_v1.json');train=p['train']
 assert manifest['successful'] and manifest['prereg_sha256']==PREREG and manifest['full90_exact_49bd_tensor_parity']
 assert manifest['all_new_adaptation_heads_fold_fit'] and manifest['shared_pretrained777a_prior_not_nested']
 families=manifest['families'];assert [f['key'] for f in families]==['fold0','fold1','fold2','fold3','fold4','full90']
 qualified=HERE/'parent_qualification_v1/QUALIFIED_v1.json';assert sha(qualified)==review['qualification_sha256']
 q=read(qualified);assert q['successful'] and q['all_six_official_cold_warm_arithmetic_identical'] and q['source_manifest_sha256']==sha(manifest_path)
 for row in review['support_guards']:
  path=Path(row['path']);assert sha(path)==row['sha256'];clean_guard(path)
 planp=ORIG/'PLAN_CAPTURE_v1.json';assert sha(planp)=='a6e841b1c5a09613e1d580f8eb49f68889eaddd1bed6f2ae73e8208f8d3e5375';plan=read(planp)
 assert [s for g in plan['groups'] for s in g]==train and len(plan['groups'])==30 and all(len(g)==3 for g in plan['groups'])
 designp=ORIG/'DESIGN_LOCAL_v1.json';assert sha(designp)==p['design_sha256'];design=read(designp)
 assert not set(train)&set(design['protected_seed_ids']) and design['weight_hash_disjointness']
 weights=[]
 for seed in train:
  row=plan['rows'][str(seed)];assert set(row)=={'weights'};wr=row['weights'];wp=verified(ROOT,wr)
  assert wr['raw_float32_sha256']==design['rows'][str(seed)]['raw_float32_sha256']
  assert wr['raw_float32_sha256'] not in design['protected_raw_weight_hashes']
  weights.append(dict(seed=seed,path=str(wp),sha256=sha(wp),raw_float32_sha256=wr['raw_float32_sha256']))
 hook=HERE/'first128_onpolicy_hook_v2.py';assert sha(hook)==HOOK;ast.parse(hook.read_text())
 assert sha(PORT/'guard_linux_v7.py')==GUARD
 harness=[]
 for name,pin in HARNESS.items():
  path=HERE/name;assert sha(path)==pin;ast.parse(path.read_text());harness.append(rec(path))
 fragment=ORIG/'fragment_mark_complete_line_v1.py';assert sha(fragment)=='ca5f41cad45a5a380b2782d4436d2149a9af50955c2af7bfc27746494e63b684';harness.append(rec(fragment))
 controls=[L/n for n in ('run_science_linux_markline_v1.py','run_group_linux_markline_v1.py','client_linux_markline_v1.py','common_linux_v7.py')]
 for path in controls:ast.parse(path.read_text())
 out=HERE/'capture_plans_v1';out.mkdir(exist_ok=False);outputs=[]
 for index,family in enumerate(families):
  key=family['key'];held=train[index::5] if index<5 else [];selected=[s for s in train if s not in held]
  assert family['train']==selected and family['validation']==held and family['parent_fit_target_seed_ids']==selected
  assert not set(selected)&set(held) and family['parent_fit_excludes_validation']
  for role in ('science','release'):
   path=verified(ROOT,family[role]);assert sha(path)==family[role]['effective_sha256']
   if key=='full90':assert sha(path)==p['parent_'+role+'_sha256']
  base_recipe=dict(utc=utc(),kind='CLEAN_FIVE_BLOCK_SOURCE_SPECIFIC_FIRST128',teacher=family['science'],source_family=family,
   source_manifest=rec(manifest_path),on_policy_prereg=rec(HERE/'PREREG_v1.json'),train_ranks=selected,validation_ranks=held,
   layers=list(range(1,16)),rows_per_layer=1024,target_free=True,uses_network_truth=False,diagnostic_only=True,
   hook=rec(hook),harness=harness,guard_unchanged=True,reference_submission=p['reference_verified_submission'],
   no_new_truths=True,no_warmup_each=True,one_predict_call_per_scored_rank=True)
  folder=out/key;folder.mkdir();items={}
  for tag,indices,diagnostic in [('control',[0],False),('anchor',[0],True),('remaining',list(range(1,30)),True)]:
   groups=[plan['groups'][i] for i in indices];ranks=[s for g in groups for s in g]
   pp=dict(plan);pp.update(utc=utc(),kind='FIRST128_ON_POLICY_'+key.upper()+'_'+tag.upper(),diagnostic_only=diagnostic,
    groups=groups,first7=ranks,held25=[],rows={str(s):plan['rows'][str(s)] for s in ranks},candidates={'teacher':family['science']},
    source_manifest_sha256=sha(manifest_path),on_policy_prereg_sha256=PREREG,original_group_indices=indices,source_family_key=key)
   path=folder/('PLAN_'+tag.upper()+'_v1.json');write(path,pp)
   item=dict(plan=rec(path),out=str(RUNS/('onpolicy_first128_'+key+'_'+tag+'_v1')),networks=len(ranks),groups=len(groups),
    original_group_indices=indices,audit_guard_path=str(RUNS/('onpolicy_first128_'+key+'_'+tag+'_audit_v1_GUARD_v1.json')))
   if diagnostic:
    rp=folder/('RECIPE_'+tag.upper()+'_v1.json');write(rp,dict(base_recipe,capture_ranks=ranks,original_group_indices=indices,stage=tag));item['recipe']=rec(rp)
   items[tag]=item
  env=dict(os.environ,**thread_env(1),ARC_BLAS_THREADS='1',ARC_FULL_STATE_RECIPE=str(ROOT/items['anchor']['recipe']['path']),ARC_FULL_STATE_RECIPE_SHA=items['anchor']['recipe']['sha256'])
  check=subprocess.run(['/home/arcworker/arc-envs/client/bin/python','-B',str(HERE/'client_capture_v1.py'),'--help'],env=env,capture_output=True,text=True)
  assert check.returncode==0,check.stderr
  outputs.append(dict(key=key,teacher=family['science'],outputs=items,client_numpy_free_help_pass=True))
 subprocess.run(['/home/arcworker/arc-envs/client/bin/python','-c',"import importlib.util;assert importlib.util.find_spec('numpy') is None"],check=True)
 ready=dict(utc=utc(),successful=True,source_manifest=rec(manifest_path),prereg_sha256=PREREG,hook=rec(hook),harness=harness,
  control_harness=[rec(path) for path in controls],original_plan=rec(planp),outputs=outputs,weights=weights,
  script_sha256=sha(__file__),guard_sha256=GUARD,guard_unchanged=True,groups_unchanged=True,
  total_capture_predictions=540,total_control_predictions=18,ceiling_predictions=558,actual_full_predict_calls=558,extra_warmup_calls=0,
  waves=[['fold0','fold1','fold2'],['fold3','fold4','full90']],jobs_per_family=4,max_capture_native=12,
  all_family_anchor_prediction_and_charged_invoice_parity_required=True,root_review_required_before_any_anchor=True,
  separate_root_review_required_before_remaining522=True,no_native_launched=True,no_truth_arrays_read=True,old_captures_untouched=True)
 write(out/'CAPTURE_READY_v1.json',ready)
 print(json.dumps({k:v for k,v in ready.items() if k not in ('weights','outputs')}),flush=True)
if __name__=='__main__':main()
