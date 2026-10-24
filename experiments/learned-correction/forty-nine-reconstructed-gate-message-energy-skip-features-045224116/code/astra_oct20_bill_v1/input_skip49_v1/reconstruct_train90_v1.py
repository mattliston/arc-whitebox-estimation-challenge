"""Reconstruct only frozen H49 inputs on a target-free third of TRAIN90."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='4'
import argparse,ast,gzip,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;S=HERE.parent/'truth_response95_v1';V=HERE.parents[1]/'astra_oct20_vm_frontier_v1'
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs');CAP=RUNS/'full_state_response_train90_v1'
sys.path.insert(0,str(V));import fit_frontier_v1 as f;import check_frontier_tiny_v6 as tiny
from replay_message49_v1 import replay,SOURCE_SHA
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
PREREG='302ac1525406ccb2d7e94c27fe4288f8c495e1ac34d41e0ab07a112579c41ae1';GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
def write(p,x):
 with Path(p).open('x') as stream:json.dump(x,stream,indent=2,allow_nan=False);stream.write('\n')
def arr(r,k):
 a=np.asarray(r[k]['data'],dtype='f4');assert list(a.shape)==r[k]['shape'] and np.isfinite(a).all();return a
def clean_guard(p):
 g=read(p);assert g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors'] and g['guard_sha256']==GUARD;return sha(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--chunk',type=int,required=True);args=ap.parse_args();assert args.chunk in (0,1,2)
 assert sys.platform=='linux' and sha(HERE/'PREREG_v1.json')==PREREG
 pre=read(HERE/'PREREG_v1.json');assert sha(HERE/'replay_message49_v1.py')==pre['replay_sha256']
 assert sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363' and sha(tiny.__file__)=='a6246110ed4e3aaeba9f49b51b9bb3bd855e6656fc3508e8703b6c1c45aac376'
 auditp=RUNS/'input_skip49_replay_audit_v3/AUDIT_v1.json';audit=read(auditp)
 assert audit['successful'] and audit['no_truth_arrays'] and audit['source_sha256']==SOURCE_SHA and audit['replay_sha256']==pre['replay_sha256']
 assert audit['prereg_sha256']==pre['replay_prereg_sha256'] and audit['script_sha256']=='c019f4e7fb0fd94b786d69cf2dd6ccf8002e23350e2bd23fb7c808e24b5d1653'
 assert audit['max_abs']<=2e-6 and audit['max_relative_l2']<=2e-6 and len(audit['rows'])==60
 ag=clean_guard(RUNS/'input_skip49_replay_audit_v3_GUARD.json')
 price=read(RUNS/'input_skip49_price_v1/PRICE_v1.json');assert price['successful'] and price['passes_price_gate'] and price['script_sha256']=='ba0d2b4d0f5bd095073913812d7e3d904c00a8ca9dd3e7e5ddbbd8a1c5fd34ca'
 pg=clean_guard(RUNS/'input_skip49_price_v1_GUARD.json')
 assert sha(S/'DESIGN_LOCAL_v1.json')==pre['design_sha256'];d=read(S/'DESIGN_LOCAL_v1.json');assert d['train']==pre['train'] and not set(d['train'])&set(d['protected_seed_ids'])
 assert sha(S/'PLAN_CAPTURE_v1.json')=='a6e841b1c5a09613e1d580f8eb49f68889eaddd1bed6f2ae73e8208f8d3e5375';plan=read(S/'PLAN_CAPTURE_v1.json')
 assert read(CAP/'COMPLETE.json')['successful'];rm=read(CAP/'RUN_MANIFEST.json');assert rm['candidate_sha256']==SOURCE_SHA and rm['blas_threads']==4 and rm['plan_sha256']==sha(S/'PLAN_CAPTURE_v1.json')
 source=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py';assert sha(source)==SOURCE_SHA;s=source.read_text();lit,_=f.literals_and_functions(s);P=f.decode(lit['_GNN_LITERAL'])
 E=ast.literal_eval([n.value for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_VMFR_ORIGINAL_W3_ENCODER' for t in n.targets)][-1])
 out=RUNS/('input_skip49_reconstruct90_part%d_v1'%(args.chunk+1));out.mkdir(exist_ok=False);rows=[]
 selected=d['train'][args.chunk::3];assert len(selected)==30
 for seed in selected:
  ix=d['train'].index(seed);part='part%02d'%(ix//3+1);assert seed in plan['groups'][ix//3];folder=CAP/part
  guard=clean_guard(CAP/(part+'_GUARD.json'));meta=read(folder/'gnn_full_state_v1/METADATA_v1.json');done=read(folder/'gnn_full_state_v1/CAPTURE_COMPLETE_v1.json')
  assert meta['source_sha256']==SOURCE_SHA and meta['hook_sha256']=='8a94ea60aab869827ad97cec4aa2aea531bfdbc5c5341e53038b7f2feea4a91b' and meta['target_free'] and done['complete'] and done['target_free']
  Q=f.decode(meta['gnn_literal']);assert all(np.array_equal(P[k],Q[k]) for k in P)
  wr=plan['rows'][str(seed)]['weights'];wp=ROOT/wr['path'];tr=d['rows'][str(seed)]
  assert sha(wp)==wr['sha256'] and wp.stat().st_size==wr['bytes'] and wr['raw_float32_sha256']==tr['raw_float32_sha256'] and wr['raw_float32_sha256'] not in d['protected_raw_weight_hashes']
  W=np.load(wp,allow_pickle=False,mmap_mode='r');assert W.shape==(16,1024,1024)
  caps=[];proof=[]
  for l in range(1,16):
   rr=[r for r in done['files'] if r['rank']==seed and r['phase']=='scored' and r['layer']==l];assert len(rr)==1;r=rr[0];p=folder/'gnn_full_state_v1'/r['path'];assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes']
   x=json.load(gzip.open(p,'rt'));assert x['rank']==seed and x['phase']=='scored' and x['layer']==l;caps.append(x);proof.append(dict(layer=l,sha256=sha(p)))
  states=[np.zeros((1024,64),'f4')]+[arr(x,'state') for x in caps[:-1]];gates=[arr(x,'gate_feature') for x in caps]
  h,info=replay(P,E,W,states,gates,s,tiny);h=np.stack(h);assert h.shape==(15,1024,49)
  p=out/('H49_seed%d.npz'%seed);np.savez_compressed(p,H49=h,seed=np.asarray(seed),source_sha256=np.asarray(SOURCE_SHA),columns=np.asarray(pre['columns']),raw_float32_sha256=np.asarray(wr['raw_float32_sha256']))
  rows.append(dict(seed=seed,path=str(p),sha256=sha(p),bytes=p.stat().st_size,weights_sha256=sha(wp),raw_float32_sha256=wr['raw_float32_sha256'],capture_guard_sha256=guard,captures=proof,replay=info))
  print(json.dumps(dict(utc=utc(),seed=seed,completed=len(rows),chunk=args.chunk)),flush=True)
 result=dict(utc=utc(),successful=True,chunk=args.chunk,train=selected,rows=rows,source_sha256=SOURCE_SHA,prereg_sha256=PREREG,replay_sha256=pre['replay_sha256'],script_sha256=sha(__file__),audit_sha256=sha(auditp),audit_guard_sha256=ag,price_guard_sha256=pg,no_truth_arrays=True,no_estimator_predictions=True)
 write(out/'RECON_COMPLETE_v1.json',result);print(json.dumps(dict(utc=utc(),successful=True,chunk=args.chunk,networks=len(rows))),flush=True)
if __name__=='__main__':main()
