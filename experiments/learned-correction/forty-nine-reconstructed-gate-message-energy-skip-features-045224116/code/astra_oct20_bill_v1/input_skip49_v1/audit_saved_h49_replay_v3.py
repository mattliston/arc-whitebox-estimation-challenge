"""Verify replay against 60 saved current-source H blocks, without truth arrays."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='4'
import ast,gzip,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;D=HERE.parent;V=D.parent/'astra_oct20_vm_frontier_v1'
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
sys.path.insert(0,str(V));import fit_frontier_v1 as f;import check_frontier_tiny_v6 as tiny
from replay_message49_v1 import replay,SOURCE_SHA
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def arr(x,k):
 a=np.asarray(x[k]['data'],dtype='f4');assert list(a.shape)==x[k]['shape'] and np.isfinite(a).all();return a
def capture(folder,seed,version,hook,P):
 meta=read(folder/('METADATA_'+version+'.json'));done=read(folder/('CAPTURE_COMPLETE_'+version+'.json'))
 assert meta['source_sha256']==SOURCE_SHA and meta['hook_sha256']==hook and meta['target_free'] and done['complete'] and done['target_free']
 Q=f.decode(meta['gnn_literal']);assert set(Q)==set(P) and all(np.array_equal(P[k],Q[k]) for k in P)
 data=[];proof=[]
 for l in range(1,16):
  records=[r for r in done['files'] if r['rank']==seed and r['layer']==l and r['phase']=='scored'];assert len(records)==1
  r=records[0];p=folder/r['path'];assert sha(p)==r['sha256']
  if 'bytes' in r:assert p.stat().st_size==r['bytes']
  x=json.load(gzip.open(p,'rt'));assert x['rank']==seed and x['layer']==l and x['phase']=='scored';data.append(x);proof.append(dict(layer=l,sha256=sha(p)))
 return data,proof
def main():
 assert sys.platform=='linux' and sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
 assert sha(tiny.__file__)=='a6246110ed4e3aaeba9f49b51b9bb3bd855e6656fc3508e8703b6c1c45aac376'
 pre=read(HERE/'REPLAY_PREREG_v1.json');assert pre['train']==[35,150,200,250] and pre['columns']==[5]+list(range(30,78))
 assert sha(HERE/'replay_message49_v1.py')==pre['replay_sha256'] and pre['no_truth_arrays']
 source=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py';assert sha(source)==SOURCE_SHA
 s=source.read_text();literal,_=f.literals_and_functions(s);P=f.decode(literal['_GNN_LITERAL'])
 E=ast.literal_eval([n.value for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_VMFR_ORIGINAL_W3_ENCODER' for t in n.targets)][-1])
 ready=read(D/'propagated_state_response_v1/CAPTURE_READY_v1.json');plan=read(ready['plan']);assert plan['groups']==[[35,150,200],[250]]
 full=RUNS/'full_state_response_train4_v1';old=RUNS/'current_msg8_gnn_capture_v1';out=RUNS/'input_skip49_replay_audit_v3';out.mkdir(exist_ok=False)
 rows=[];proof=[]
 for gi,group in enumerate(plan['groups'],1):
  part='part%02d'%gi
  for base in (full,old):
   g=read(base/(part+'_GUARD.json'));assert g['successful'] and g['whole_group_retired'] and g['child_exit']==0 and not g['cleanup_errors'] and g['guard_sha256']=='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
   m=read(base/'RUN_MANIFEST.json');assert m['candidate_sha256']==SOURCE_SHA and m['blas_threads']==4 and m['setup']=='paced' and not m['warmup_each']
  for seed in group:
   p0=full/part/('diagnostic_pred_%d.bin'%seed);p1=old/part/('diagnostic_pred_%d.bin'%seed);assert p0.read_bytes()==p1.read_bytes()
   a,ap=capture(full/part/'gnn_full_state_v1',seed,'v1','d5292105219e14fdb9edf04a476888e844e9f518ff428bae675fa146fdc3e1b5',P)
   b,bp=capture(old/part/'gnn_current_msg8_v1',seed,'v2','07ae3f9ff87827ab4bb569de929af9a3ded296730d8b46be23b407bf677b856a',P)
   wr=plan['rows'][str(seed)]['weights'];wp=ROOT/wr['path'];assert sha(wp)==wr['sha256'];W=np.load(wp,allow_pickle=False,mmap_mode='r');assert W.shape==(16,1024,1024)
   states=[np.zeros((1024,64),'f4')]+[arr(x,'state') for x in a[:-1]];gates=[arr(x,'gate_feature') for x in a]
   got,info=replay(P,E,W,states,gates,s,tiny)
   for l,(x,y,h) in enumerate(zip(a,b,got),1):
    ix=y['rows'];assert ix==list(range(seed%8,1024,8))
    for key in ('state','hidden2','sigma','delta'):np.testing.assert_array_equal(arr(x,key)[ix],arr(y,key))
    expected=arr(y,'H')[:,pre['columns']];np.testing.assert_array_equal(h[ix,0],expected[:,0]);difference=h[ix]-expected
    mx=float(np.max(np.abs(difference)));rel=float(np.linalg.norm(difference)/max(np.linalg.norm(expected),1e-30))
    rows.append(dict(seed=seed,layer=l,max_abs=mx,relative_l2=rel,byte_equal=bool(np.array_equal(h[ix],expected))))
    assert mx<=pre['max_abs_tolerance'] and rel<=pre['relative_l2_tolerance'],rows[-1]
   proof.append(dict(seed=seed,prediction_sha256=sha(p0),weights_sha256=sha(wp),full_capture=ap,H_capture=bp,replay=info))
   print(json.dumps(dict(utc=utc(),seed=seed,complete=True,max_abs=max(x['max_abs'] for x in rows if x['seed']==seed))),flush=True)
 result=dict(utc=utc(),successful=True,status='SAVED_H49_SOURCE_UNITS_AND_REPLAY_PASS',source_sha256=SOURCE_SHA,script_sha256=sha(__file__),prereg_sha256=sha(HERE/'REPLAY_PREREG_v1.json'),replay_sha256=sha(HERE/'replay_message49_v1.py'),max_abs=max(r['max_abs'] for r in rows),max_relative_l2=max(r['relative_l2'] for r in rows),byte_equal_blocks=sum(r['byte_equal'] for r in rows),rows=rows,provenance=proof,columns=pre['columns'],no_truth_arrays=True,no_fit=True,no_estimator_predictions_run=True,no_missing_physical_features=True)
 with (out/'AUDIT_v1.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['rows','provenance','columns']}),flush=True)
if __name__=='__main__':main()
