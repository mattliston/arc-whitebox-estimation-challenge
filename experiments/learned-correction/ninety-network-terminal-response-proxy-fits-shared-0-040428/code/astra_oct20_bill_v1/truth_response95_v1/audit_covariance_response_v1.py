"""Target-free TRAIN4 audit of explicit head-to-covariance feedback.
Uses one already frozen perturbation and existing predictions; no truth or fit.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import ast,gzip,hashlib,json,math,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
V=HERE.parents[1]/'astra_oct20_vm_frontier_v1';BASECAP=RUNS/'full_state_response_train4_v1'
INNOV=RUNS/'current_innovation_exploratory_markline32_blas4_v1'
CONTROL=RUNS/'msg8_loaderfix337422_control32_markline_blas4_v1'
OUT=RUNS/'truth_response_covariance_audit_v1';TRAIN=[35,150,200,250]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
NEW='fb4bbca40f592959b244d510e236a13a7c8d913c9383655204d769322156a3ff'

def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def arr(r,k):
 a=np.asarray(r[k]['data'],dtype='f8');assert list(a.shape)==r[k]['shape'] and np.isfinite(a).all();return a
def compare(p,y):
 a=float(np.linalg.norm(p));b=float(np.linalg.norm(y));assert b>0 and a>0
 return dict(predicted_rms=float(np.sqrt(np.mean(p*p))),observed_rms=float(np.sqrt(np.mean(y*y))),
  relative_L2_error=float(np.linalg.norm(p-y)/b),cosine=float(p@y/(a*b)),
  descriptive_regression_scale=float(p@y/(p@p)))
def selftest():
 rng=np.random.default_rng(20261021);n=13
 m=rng.uniform(.2,1,n);dm=rng.normal(size=n)*.01;K=rng.normal(size=n)*.02;D=rng.normal(size=n)*.01
 W=rng.normal(size=(n,n))/np.sqrt(n);beta=.346
 def cov(x):
  h=m+x*K;d=dm+x*D
  return -beta*(np.outer(h,d)+np.outer(d,h)+np.outer(d,d))
 eps=1e-5;fd=(cov(eps)-cov(-eps))/(2*eps)
 V=np.column_stack([m+dm,dm]);Q=-beta*np.column_stack([D,K])
 dense=V@Q.T+Q@V.T
 np.testing.assert_allclose(fd,dense,rtol=1e-8,atol=1e-11)
 A=W.T@V;B=W.T@Q;diag=2*np.sum(A*B,axis=1)
 np.testing.assert_allclose(np.diag(W.T@fd@W),diag,rtol=1e-8,atol=1e-11)
 gate=rng.uniform(0,1,n);v=gate[:,None]*A;q=gate[:,None]*B
 np.testing.assert_allclose(v@q.T+q@v.T,gate[:,None]*(W.T@dense@W)*gate[None,:],rtol=1e-12,atol=1e-12)
 return dict(rank_two_covariance_update_derivative=True,low_rank_sandwich_diagonal=True,frozen_gate_transport=True)

def main():
 sys.path.insert(0,str(V));import fit_frontier_v1 as f
 assert sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
 checks=selftest()
 if '--self-test' in sys.argv:print(json.dumps(checks));return
 assert sys.platform=='linux'
 prereg=read(HERE/'COV_RESPONSE_AUDIT_PREREG_v1.json');assert prereg['train']==TRAIN and prereg['no_truth_arrays']
 original=V/'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py'
 altered=HERE.parent/'shared_innovation_head_v1/native_exploratory_v1/candidate_current_innovation_exploratory_science_v1.py'
 assert sha(original)==SCI and sha(altered)==NEW
 fit=RUNS/'current_shared_innovation_head_fit_v1/FITTED_v1.json';assert sha(fit)=='8b16ef3239fbe0bb165c58f20a7e65da1fd7a20902a396b3dcb22cd4b34c0d94'
 # Read frozen source constants only. No truth labels or new optimization.
 p,_=f.literals_and_functions(original.read_text());q,_=f.literals_and_functions(altered.read_text())
 P=f.decode(p['_GNN_LITERAL']);Q=f.decode(q['_GNN_LITERAL'])
 assert all(np.array_equal(P[k],Q[k]) for k in P if k!='head')
 dh=Q['head'].astype('f8')-P['head'].astype('f8');assert np.array_equal(dh[0],np.zeros(64))
 constants={}
 for n in ast.parse(original.read_text()).body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['_GN_RC','_GN_KC']:
   constants[n.targets[0].id]=ast.literal_eval(n.value)
 beta={l:1+constants['_GN_RC'].get(l,0)-constants['_GN_KC'].get(l,0) for l in range(1,15)}
 ready=read(HERE.parent/'propagated_state_response_v1/CAPTURE_READY_v1.json')
 cp=read(ready['plan']);assert cp['groups']==[[35,150,200],[250]]
 OUT.mkdir(exist_ok=False);write(OUT/'PREREG_COPY_v1.json',prereg)
 rows=[];provenance=[];pernet=[]
 for gi,group in enumerate(cp['groups'],1):
  part='part%02d'%gi
  for run in (BASECAP,INNOV,CONTROL):
   g=read(run/(part+'_GUARD.json'));assert g['successful'] and g['child_exit']==0 and not g['cleanup_errors']
  cd=BASECAP/part/'gnn_full_state_v1';meta=read(cd/'METADATA_v1.json');done=read(cd/'CAPTURE_COMPLETE_v1.json')
  assert meta['source_sha256']==SCI and done['complete'] and done['target_free']
  for rank in group:
   base=BASECAP/part/f'diagnostic_pred_{rank}.bin';ref=CONTROL/part/f'pred_{rank}.bin';cand=INNOV/part/f'pred_{rank}.bin'
   assert sha(base)==sha(ref)
   y0=np.fromfile(base,dtype='<f4').reshape(16,1024).astype('f8')
   y1=np.fromfile(cand,dtype='<f4').reshape(16,1024).astype('f8');observed=y1-y0
   wr=cp['rows'][str(rank)]['weights'];wp=ROOT/wr['path'];assert sha(wp)==wr['sha256'];W=np.load(wp,mmap_mode='r',allow_pickle=False)
   Jmean=np.zeros(1024);Jcov=np.zeros(1024);carV=np.empty((1024,0));carQ=np.empty((1024,0));results=[]
   for l in range(1,16):
    rr=next(r for r in done['files'] if r['rank']==rank and r['phase']=='scored' and r['layer']==l)
    path=cd/rr['path'];assert sha(path)==rr['sha256']
    with gzip.open(path,'rt',encoding='utf-8') as stream:r=json.load(stream)
    S=arr(r,'state');sigma=arr(r,'sigma');ng=arr(r,'gate_feature');delta=arr(r,'delta')
    gate=ng*float(P['scales'][l,5])+float(P['centers'][l,5]);gate=np.clip(gate,0,1)
    w=np.asarray(W[l],dtype='f8');mu=w.T@y0[l-1];alpha=mu/sigma;pdf=np.exp(-.5*alpha*alpha)/np.sqrt(2*np.pi)
    cdf=.5*(1+np.array([math.erf(float(a)/np.sqrt(2)) for a in alpha]))
    gate_error=float(np.max(np.abs(gate-cdf)));assert gate_error<3e-5,gate_error
    scale=sigma*float(P['rms'][l])*(1+.25*np.tanh(float(P['gain'][l])))
    direct=scale*(S@dh[l]);mask=(y0[l]>1e-12).astype('f8') if l<15 else np.ones(1024)
    Jmean=mask*(gate*(w.T@Jmean)+direct)
    av=w.T@carV;aq=w.T@carQ;dv=2*np.sum(av*aq,axis=1)
    K=gate*(w.T@Jcov)+.5*pdf/sigma*dv
    D=direct+.5*delta/(sigma*sigma)*dv
    Jcov=mask*(K+D);deff=Jcov-K
    carV=gate[:,None]*av;carQ=gate[:,None]*aq
    clip_bound=0.;dominant_norm=0.
    if l<15:
     # Actual effective delta differs from raw delta only on clipped rows.
     # Its uncertain dm*K pair has an explicit conservative Frobenius bound.
     clip_error=np.where(mask==0,np.abs(delta),0.)
     clip_bound=float(2*abs(beta[l])*np.linalg.norm(clip_error)*np.linalg.norm(K))
     dominant_norm=float(2*abs(beta[l])*np.linalg.norm(y0[l])*np.linalg.norm(deff))
     carV=np.column_stack([carV,y0[l],delta]);carQ=np.column_stack([carQ,-beta[l]*deff,-beta[l]*K])
    row=dict(rank=rank,layer=l,mean_only=compare(Jmean,observed[l]),mean_and_rank_two_covariance=compare(Jcov,observed[l]),
      gate_replay_max_abs=gate_error,clipped=int(np.sum(mask==0)),uncertain_clipped_birth_frobenius_bound=clip_bound,
      dominant_birth_frobenius_upper=dominant_norm,variance_response_rms=float(np.sqrt(np.mean(dv*dv))))
    rows.append(row);results.append(row)
   pernet.append(results[-1]);provenance.append(dict(rank=rank,base_sha256=sha(base),changed_prediction_sha256=sha(cand),weights_sha256=sha(wp)))
   print(json.dumps(dict(rank=rank,terminal=results[-1])),flush=True)
 mean_error=float(np.mean([r['mean_only']['relative_L2_error'] for r in pernet]));cov_error=float(np.mean([r['mean_and_rank_two_covariance']['relative_L2_error'] for r in pernet]))
 mean_cos=float(np.mean([r['mean_only']['cosine'] for r in pernet]));cov_cos=float(np.mean([r['mean_and_rank_two_covariance']['cosine'] for r in pernet]))
 result=dict(utc=utc(),status='PHYSICS_RESPONSE_SUPPORTED' if cov_error<mean_error and cov_cos>mean_cos else 'PHYSICS_RESPONSE_NOT_SUPPORTED',
  no_truth_or_fit=True,source_sha256=SCI,altered_source_sha256=NEW,coefficient_source_sha256=sha(fit),
  selftests=checks,beta=beta,terminal_mean_only_relative_L2=mean_error,terminal_covariance_relative_L2=cov_error,
  terminal_mean_only_cosine=mean_cos,terminal_covariance_cosine=cov_cos,per_network_terminal=pernet,
  all_layers=rows,provenance=provenance,prereg_sha256=sha(HERE/'COV_RESPONSE_AUDIT_PREREG_v1.json'),script_sha256=sha(__file__),
  limitation=prereg['clip_limitation'],omitted_feedback='pair-kernel gate derivatives, state/input/edge/higher-cumulant feedback',no_adjusted_gain_claim=True)
 write(OUT/'AUDIT_v1.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ['all_layers','provenance','per_network_terminal']}),flush=True)
if __name__=='__main__':main()
