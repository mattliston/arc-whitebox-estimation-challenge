"""One fixed hard-clip14 physical-output teacher fit; no network truth."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse, hashlib, importlib.util, json, resource, subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
TRAIN=(35,150,200,250);VALID=(259,277,287);LAYERS=tuple(range(8,15));RIDGE=1e-4
TEACHER='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
LOADER='ea5c8f131bfbe2c6cdd1bef4b7036fb0d1d3dd686ade8fa16c59b12a0e4d470d'
VERIFIED='5210ebea545c7b4e70984d731525023d5b488dcc057afa1290b6b249476c1cdc'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def solve(records,prior):
 G=sum(r['X'].T@(r['X']*r['weights'][:,None]) for r in records)/len(records)
 b=sum(r['X'].T@(r['weights']*r['target']) for r in records)/len(records)
 scale=np.maximum(np.sqrt(np.diag(G)),1e-12)
 matrix=G/scale[:,None]/scale[None,:]+RIDGE*np.eye(14)
 rhs=(b-G@prior)/scale
 dz=np.linalg.solve(matrix,rhs)
 assert np.max(np.abs(matrix@dz-rhs))<1e-8*max(1.,float(np.max(np.abs(rhs))))
 return (prior+dz/scale).astype('f4')
def error(records,out):
 rows=[dict(rank=r['rank'],layer=r['layer'],relative_mse=float(r['weights']@((r['X']@out-r['target'])**2))) for r in records]
 return dict(mean_relative_mse=float(np.mean([r['relative_mse'] for r in rows])),
             per_rank={str(k):float(np.mean([r['relative_mse'] for r in rows if r['rank']==k])) for k in sorted({r['rank'] for r in rows})},rows=rows)
def main():
 p=argparse.ArgumentParser()
 for name in ('captures','verification','physical-loader','out'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args(); assert sha(a.physical_loader)==LOADER and sha(a.verification)==VERIFIED
 prereg=json.loads((HERE/'PREREG_v1.json').read_text())
 assert prereg['train']==list(TRAIN) and prereg['validation']==list(VALID) and prereg['ridge']==RIDGE
 assert prereg['teacher_source_sha256']==TEACHER and not prereg['uses_network_truth']
 verify=json.loads(a.verification.read_text())
 assert verify['successful'] and verify['no_truth'] and verify['teacher_sha256']==TEACHER
 for g in verify['guards']:
  gp=Path(g['path']);assert sha(gp)==g['sha256']; gg=json.loads(gp.read_text())
  assert gg.get('successful') or gg.get('success') or gg.get('status')=='SUCCESS',list(gg)
 spec=importlib.util.spec_from_file_location('pinned_edge_physical_loader',a.physical_loader)
 h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 records_index={};meta=None;provenance=[]
 for mp in sorted(a.captures.glob('part*/edge_prepost/MANIFEST.json')):
  manifest=json.loads(mp.read_text());directory=mp.parent
  assert manifest['uses_network_truth'] is False and manifest['arithmetic_AST_identical_after_removing_capture']
  metap=directory/'METADATA_v1.json';assert sha(metap)==manifest['metadata_sha256']
  m=json.loads(metap.read_text())
  assert m['teacher_source_sha256']==TEACHER and m['hidden']==14 and m['message_width']==8
  assert m['recipe']['train_ranks']==list(TRAIN) and m['recipe']['validation_ranks']==list(VALID)
  if meta is None:meta=m
  else:assert meta==m
  for item in manifest['records']:
   rp=directory/item['path'];assert sha(rp)==item['sha256'];r=json.loads(rp.read_text())
   key=(r['rank'],r['layer']);assert key not in records_index
   records_index[key]=(directory,r)
  provenance.append(dict(path=str(mp),sha256=sha(mp),metadata_sha256=sha(metap)))
 assert set(records_index)=={(r,l) for r in TRAIN+VALID for l in LAYERS}
 a.out.mkdir(exist_ok=False)
 write(a.out/'START_v1.json',dict(utc=utc(),prereg_sha256=sha(HERE/'PREREG_v1.json'),
       source_sha256=sha(__file__),captures=provenance,uses_network_truth=False))
 def collect(ranks):
  records=[]
  for rank in ranks:
   for layer in LAYERS:
    directory,rec=records_index[rank,layer]
    r=h.physical_record(directory,rec,meta)
    r['X']=np.clip(r['zf'],-1.,1.)*r['af'][:,None]+np.clip(r['zr'],-1.,1.)*r['ar'][:,None]
    # Release preactivation arrays after the fixed feature map.
    del r['zf'],r['zr'],r['af'],r['ar'],r['G']
    records.append(r)
  return records
 tr=collect(TRAIN);prior=tr[0]['output']
 assert prior.shape==(14,) and all(np.array_equal(r['output'],prior) for r in tr)
 out=solve(tr,prior);before=error(tr,prior);after=error(tr,out)
 loo=[]
 for rank in TRAIN:
  kept=[r for r in tr if r['rank']!=rank];left=[r for r in tr if r['rank']==rank]
  beta=solve(kept,prior)
  loo.append(dict(rank=rank,before=error(left,prior),after=error(left,beta)))
 loo_before=float(np.mean([x['before']['mean_relative_mse'] for x in loo]))
 loo_after=float(np.mean([x['after']['mean_relative_mse'] for x in loo]))
 lower=sum(x['after']['mean_relative_mse']<x['before']['mean_relative_mse'] for x in loo)
 gate=bool(loo_after<=.5*loo_before and lower>=3 and loo_after<=.01)
 frozen=dict(utc=utc(),status='FROZEN_CLIP14_OUTPUT_NO_NATIVE_SCORE',source_sha256=sha(__file__),
      prereg_sha256=sha(HERE/'PREREG_v1.json'),teacher_source_sha256=TEACHER,hidden=14,
      output=out.tolist(),prior_output=prior.tolist(),train=TRAIN,layers=LAYERS,
      train_before=before,train_after=after,leave_one_network=loo,
      loo_before=loo_before,loo_after=loo_after,loo_ratio=loo_after/loo_before,loo_lower=lower,
      native_fidelity_gate_pass=gate,captures=provenance,physical_loader_sha256=LOADER,
      capture_verification_sha256=VERIFIED,uses_network_truth=False,
      held25_and_fresh_unread=True,no_validation_loaded_before_freeze=True)
 write(a.out/'FROZEN_COEFFICIENTS_v1.json',frozen)
 validation=None
 if gate:
  va=collect(VALID)
  validation=dict(before=error(va,prior),after=error(va,out))
 result=dict(utc=utc(),status='FIT_PASS_READY_FOR_QUALIFICATION' if gate else 'CLOSED_FIDELITY',
    frozen_sha256=sha(a.out/'FROZEN_COEFFICIENTS_v1.json'),native_fidelity_gate_pass=gate,
    train_before=before['mean_relative_mse'],train_after=after['mean_relative_mse'],
    loo_before=loo_before,loo_after=loo_after,loo_ratio=loo_after/loo_before,loo_lower=lower,
    validation=validation,validation_used_for_selection=False,
    output_norm=float(np.linalg.norm(out)),prior_norm=float(np.linalg.norm(prior)),
    output_change_norm=float(np.linalg.norm(out-prior)),uses_network_truth=False,
    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 write(a.out/'FIT_REPORT_v1.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='validation'},indent=2))
if __name__=='__main__':main()

