"""Audit one same-source anchor or remaining87 capture; labels never read."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import argparse,ast,gzip,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;FAMILY=HERE;D=HERE.parent
ROOT=Path('/home/arcworker/arc-workspace');V=D.parent/'astra_oct20_vm_frontier_v1'
sys.path.insert(0,str(V));import fit_frontier_v1 as f
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
PREREG='42c4f80e65348f149e4b64d8e91da29f0fb1995abb5a78f8004fe0623a4a5bd9'
def guardcheck(p,threads):
 g=read(p)
 assert g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors']
 assert g['guard_sha256']==GUARD and g['thread_provenance']['blas_threads']==threads
 assert g['cap_bytes']==6500000000 and g['reserve_bytes']==19000000000 and g['max_active']==36 and g['max_owned_swap_bytes']==0
 assert all(v==0 for v in g['cgroup_events_final'].values())
 return dict(path=str(p),sha256=sha(p),peak_bytes=g['peak_group_rss_bytes'],clean=True)
def arr(r,name,shape):
 value=np.asarray(r[name]['data'],dtype='f8');assert list(value.shape)==r[name]['shape']==list(shape) and np.isfinite(value).all(),name
 return value
def endpoint(item,source):
 pp=ROOT/item['plan']['path'];assert sha(pp)==item['plan']['sha256'];plan=read(pp);out=Path(item['out'])
 done=read(out/'COMPLETE.json');manifest=read(out/'RUN_MANIFEST.json')
 assert done['successful'] and done['plan_sha256']==sha(pp) and done['blas_threads']==4
 assert manifest['candidate_sha256']==source and manifest['plan_sha256']==sha(pp) and manifest['blas_threads']==4
 assert manifest['groups']==list(range(1,len(plan['groups'])+1)) and not manifest['warmup_each'] and manifest['max_active']==36
 return plan,out,done,manifest
def invoice_rows(path,group):
 rows=[json.loads(line) for line in path.read_text().splitlines()]
 assert len(rows)==len(group) and [r['rank'] for r in rows]==group
 assert all(r['phase']=='scored' and r['n_out']==16384 and r['nonfinite']==0 for r in rows)
 return {r['rank']:r for r in rows}
def charged(row):
 return {key:int(value[1]) for key,value in row['operations'].items() if int(value[1])!=0}
def delta32(P,l,state,hidden,first,sigma):
 ss,hh,ff,sig=[np.asarray(v,dtype='f4') for v in (state,hidden,first,sigma)]
 inner=((ss@P['head'][l]+hh@P['hidden_head'][l])+ff@P['first_hidden_head'][l])+P['head_b'][l]
 gain=np.float32(1)+np.float32(.25)*np.tanh(P['gain'][l:l+1])
 result=((sig*P['rms'][l])*inner)*gain
 assert result.dtype==np.dtype('f4') and np.isfinite(result).all()
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['fold0','fold1','fold2','fold3','fold4','full90'],required=True)
 ap.add_argument('--stage',choices=['anchor','remaining'],required=True);ap.add_argument('--ready-sha256',required=True);a=ap.parse_args()
 assert sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
 readyfile=FAMILY/'capture_plans_v1/CAPTURE_READY_v1.json';assert sha(readyfile)==a.ready_sha256;ready=read(readyfile)
 assert ready['successful'] and ready['prereg_sha256']==PREREG and ready['ceiling_predictions']==558
 assert sha(FAMILY/'PREREG_v1.json')==PREREG and sha(ROOT/ready['source_manifest']['path'])==ready['source_manifest']['sha256']
 matches=[r for r in ready['outputs'] if r['key']==a.family];assert len(matches)==1;family=matches[0]
 source=family['teacher']['effective_sha256'];sourcepath=ROOT/family['teacher']['path'];assert sha(sourcepath)==source
 literal,_=f.literals_and_functions(sourcepath.read_text());P=f.decode(literal['_GNN_LITERAL'])
 assert P['head'].shape==(16,64) and P['hidden_head'].shape==P['first_hidden_head'].shape==(16,128)
 item=family['outputs'][a.stage];plan,out,done,manifest=endpoint(item,source)
 rp=ROOT/item['recipe']['path'];assert sha(rp)==item['recipe']['sha256'];recipe=read(rp)
 assert recipe['teacher']==family['teacher'] and recipe['on_policy_prereg']['sha256']==PREREG
 assert recipe['source_manifest']==ready['source_manifest'] and recipe['hook']==ready['hook'] and recipe['target_free'] and not recipe['uses_network_truth']
 assert sha(ROOT/recipe['hook']['path'])==recipe['hook']['sha256']
 control=None;anchorproof=None;guards=[];pairs=[];files=[];predictions=[]
 if a.stage=='anchor':
  cplan,control,_,_=endpoint(family['outputs']['control'],source);assert cplan['groups']==plan['groups'] and len(plan['groups'])==1
 else:
  ai=family['outputs']['anchor'];anchor=Path(ai['out'])/'AUDIT_v1.json';ar=read(anchor)
  assert ar['successful'] and ar['stage']=='anchor' and ar['family']==a.family and ar['source_sha256']==source and ar['all3_anchor_predictions_byte_identical'] and ar['all3_anchor_charged_invoices_exact']
  assert ar['ready_sha256']==a.ready_sha256 and ar['prereg_sha256']==PREREG and ar['source_manifest_sha256']==ready['source_manifest']['sha256']
  ag=guardcheck(Path(ai['audit_guard_path']),1);anchorproof=dict(audit_path=str(anchor),audit_sha256=sha(anchor),guard=ag)
 for partindex,(group,original_index) in enumerate(zip(plan['groups'],item['original_group_indices']),1):
  part=out/('part%02d'%partindex);guards.append(guardcheck(out/('part%02d_GUARD.json'%partindex),4));mf=read(part/'manifest.json')
  assert mf['sha256']==source and mf['ranks']==group and mf['diagnostic_only'] and mf['environment']['thread_provenance']['blas_threads']==4 and not mf['warmup_each']
  diagnostic_rows=invoice_rows(part/'diagnostic_rows.jsonl',group)
  if control:
   oldpart=control/('part%02d'%partindex);guards.append(guardcheck(control/('part%02d_GUARD.json'%partindex),4));cmf=read(oldpart/'manifest.json')
   assert cmf['sha256']==source and cmf['ranks']==group and not cmf.get('diagnostic_only',False) and cmf['environment']['thread_provenance']['blas_threads']==4 and not cmf['warmup_each']
   plain_rows=invoice_rows(oldpart/'rows.jsonl',group)
  cap=part/'gnn_full_state_v1';meta=read(cap/'METADATA_v1.json');completion=read(cap/'CAPTURE_COMPLETE_v1.json')
  assert meta['source_sha256']==source and meta['hook_sha256']==recipe['hook']['sha256'] and meta['target_free']
  assert meta['source_family']==recipe['source_family'] and meta['source_manifest']==recipe['source_manifest'] and meta['on_policy_prereg']==recipe['on_policy_prereg']
  params=f.decode(meta['gnn_literal']);assert set(params)==set(P) and all(np.array_equal(params[k],P[k]) for k in P)
  assert completion['complete'] and completion['source_sha256']==source and completion['target_free'] and not completion['native_handles_retained']
  assert not completion['native_handles_or_frames_retained'] and completion['all_first_paired_once'] and completion['pending_cache_empty']
  assert completion['source_function_aliases_unchanged'] and completion['gc_settings_unchanged'] and completion['valid_layer0_returns_skipped']==0
  assert completion['direct_first128_count']==15*len(group)
  expected={(seed,'scored',l) for seed in group for l in range(1,16)}
  assert {(r['rank'],r['phase'],r['layer']) for r in completion['files']}==expected and len(completion['files'])==len(expected)
  for seed in group:
   pred=part/('diagnostic_pred_%d.bin'%seed);assert pred.stat().st_size==65536
   vv=np.fromfile(pred,dtype='<f4');assert vv.shape==(16384,) and np.isfinite(vv).all()
   assert diagnostic_rows[seed]['prediction_sha256']==sha(pred)
   predictions.append(dict(seed=seed,path=str(pred),sha256=sha(pred),original_group_index=original_index,invoice_path=str(part/'diagnostic_rows.jsonl'),invoice_sha256=sha(part/'diagnostic_rows.jsonl'),flops=int(diagnostic_rows[seed]['flops'])))
   if control:
    old=oldpart/('pred_%d.bin'%seed);assert old.stat().st_size==65536 and old.read_bytes()==pred.read_bytes()
    plain,diagnostic=plain_rows[seed],diagnostic_rows[seed]
    assert plain['prediction_sha256']==sha(old) and plain['flops']==diagnostic['flops'] and charged(plain)==charged(diagnostic)
    pairs.append(dict(seed=seed,prediction_sha256=sha(pred),control_path=str(old),byte_identical=True,charged_invoice_exact=True,
     plain_flops=int(plain['flops']),capture_flops=int(diagnostic['flops']),charged_operations=charged(plain),
     plain_requests=int(plain['requests']),capture_requests=int(diagnostic['requests']),request_difference=int(diagnostic['requests'])-int(plain['requests']),
     plain_invoice_sha256=sha(oldpart/'rows.jsonl'),capture_invoice_sha256=sha(part/'diagnostic_rows.jsonl'),diagnostic_runtime_not_evidence=True))
  for r in completion['files']:
   fp=cap/r['path'];assert fp.stat().st_size==r['bytes'] and sha(fp)==r['sha256']
   with gzip.open(fp,'rt') as stream:x=json.load(stream)
   seed,l=x['rank'],x['layer'];assert (seed,x['phase'],l)==(r['rank'],r['phase'],r['layer'])
   state=arr(x,'state',(1024,64));hidden=arr(x,'hidden2',(1024,128));first=arr(x,'first128',(1024,128));
   for value in (state,hidden,first):np.testing.assert_array_equal(value,value.astype('f4').astype('f8'))
   sigma=arr(x,'sigma',(1024,));gate=arr(x,'gate_feature',(1024,));delta=arr(x,'delta',(1024,))
   replay_state=hidden@P['enc4_w'].astype('f8').T+P['enc4_b'].astype('f8')
   state_error=float(np.max(np.abs(replay_state-state)));assert state_error<3e-5
   # Both existing hidden readouts, exact float32 expression order at captured own input.
   replay=delta32(P,l,state,hidden,first,sigma).astype('f8')
   delta_error=float(np.max(np.abs(replay-delta)));assert delta_error<1e-8+2e-5*float(np.max(np.abs(delta)))
   assert float(np.max(np.abs(gate)))<7.999
   files.append(dict(seed=seed,layer=l,path=str(fp),sha256=sha(fp),bytes=fp.stat().st_size,head_replay_max_abs=delta_error,state_replay_max_abs=state_error,original_group_index=original_index))
 assert len(predictions)==item['networks'] and len(files)==15*item['networks']
 if control:assert len(pairs)==3 and all(x['byte_identical'] and x['charged_invoice_exact'] for x in pairs)
 result=dict(utc=utc(),successful=True,family=a.family,stage=a.stage,source_sha256=source,prereg_sha256=PREREG,
  source_manifest_sha256=ready['source_manifest']['sha256'],ready_sha256=a.ready_sha256,plan_sha256=item['plan']['sha256'],recipe_sha256=item['recipe']['sha256'],
  completion_sha256=sha(out/'COMPLETE.json'),manifest_sha256=sha(out/'RUN_MANIFEST.json'),hook_sha256=recipe['hook']['sha256'],
  guards=guards,anchor_proof=anchorproof,prediction_pairs=pairs,predictions=predictions,files=files,
  all3_anchor_predictions_byte_identical=bool(control),all3_anchor_charged_invoices_exact=bool(control),remaining_uses_same_source_anchor_proof=bool(anchorproof),
  all_schema_hash_finite_checks_pass=True,current_hidden_and_first_heads_in_replay=True,direct_float32_parent_replay=True,first128_captured_directly_not_inverted=True,no_truth_arrays_read=True,
  no_science_score_claim=True,auditor_sha256=sha(__file__))
 with (out/'AUDIT_v1.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ('guards','prediction_pairs','predictions','files')}),flush=True)
if __name__=='__main__':main()
