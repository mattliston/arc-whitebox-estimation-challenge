"""One fixed TRAIN16 deployable weighted-prefix rank2 proxy; no models, truths, or grid."""
from pathlib import Path
import hashlib,json,time
import numpy as np
D=Path(__file__).resolve().parent;C=D.parent/'astra_native_terminal_linear_refit_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ret=C/'ROOT_CAPTURE_RETIREMENT_v1.json';assert sha(ret)=='94dbc42175d4e7349217f6a3c37134ad83f454654fed4a17e1f58e27b6d6fde6'
assert json.loads(ret.read_text())['all_owned_PID_PPID_PGID_absent']
files=sorted((C/'run_capture_s0_v1/train16_read_observer_v1').glob('L*.npz'));assert [p.stem for p in files]==['L%02d'%l for l in range(2,15)]
def orth(x,y):
 a=np.column_stack((x,y));q,r=np.linalg.qr(a,mode='reduced');m=len(x);r0,r01,r1=abs(r[0,0]),abs(r[0,1]),abs(r[1,1]);eps=32*np.finfo(float).eps*max(m,2)
 if r0!=0 and r1>eps*(r01+r1):return q
 out=[]
 for z in (x,y):
  z=z.copy();original=np.linalg.norm(z)
  for _ in range(2):
   for b in out:z-=b*np.dot(b,z)
  norm=np.linalg.norm(z)
  if norm>np.finfo(float).eps*max(m,2)*original:out.append(z/norm)
 return np.column_stack(out)if out else np.zeros((m,0))
begin=time.monotonic();results=[];pooled={name:dict(raw=0.,directed=0.,symmetric=0.)for name in ('incumbent_QR','weighted_prefix_rank2')};pins={}
for p in files:
 pins[str(p)]=sha(p)
 with np.load(p,allow_pickle=False)as z:
  A=z['left_young'].astype('d');c=int(z['c15']);c2=int(z['c2']);B=z['right_young'][c:].astype('d')*z['lambda_young'].astype('d')
  rows=z['rows'].astype(int);cols=z['cols'][c:].astype(int);mu=z['mu'].astype('d');vc=z['vc'].astype('d');g=z['gate'][cols].astype('d');Q=z['Q'].astype('d');prefix=z['prefix'].astype('d');gp=z['gate'][z['cols'][:c]].astype('d');I=z['old_inner'].astype('d')if 'old_inner'in z else None;Bp=z['old_right'].astype('d')if 'old_right'in z else None
 sigma=np.sqrt(vc[rows]);alpha=mu[rows]/sigma;d=np.exp(-.5*alpha*alpha)/(2*np.sqrt(2*np.pi)*sigma)
 assert np.all(d>0)and np.all(g>0)and np.isfinite(d).all()
 Y=A@B.T;del A,B
 # Fixed deployable single weighted PREFIX power; exact old contribution excluded.
 Q0=orth(d*vc[rows],d*mu[rows]);R=d[:,None]*Q0
 V=prefix.T@R
 if I is not None:V-=Bp@(I.T@R)
 V*=gp[:,None]**2
 U=prefix@V
 if I is not None:U-=I@(Bp.T@V)
 U*=d[:,None]
 Qw=orth(U[:,0],U[:,1]if U.shape[1]==2 else np.zeros(len(rows)))
 if Qw.shape[1]<Q0.shape[1]:Qw=Q0
 oracle=Y-(Qw@(Qw.T@(d[:,None]*Y)))/d[:,None]
 parent=Y-Q@(Q.T@Y)
 _,ri,ci=np.intersect1d(rows,cols,return_indices=True)
 def score(E):
  E=E.copy();E[ri,ci]=0;W=d[:,None]*E*g[None,:]
  S=np.zeros((len(mu),len(mu)),dtype='d');S[np.ix_(rows,cols)]=W;S+=S.T.copy();np.fill_diagonal(S,0)
  return dict(raw=float(np.sum(E*E)),directed=float(np.sum(W*W)),symmetric=float(np.sum(S*S)))
 cases=dict(incumbent_QR=score(parent),weighted_prefix_rank2=score(oracle))
 for name,values in cases.items():
  for k,v in values.items():pooled[name][k]+=v
 ratios={k:cases['weighted_prefix_rank2'][k]/max(cases['incumbent_QR'][k],1e-300)for k in cases['incumbent_QR']}
 results.append(dict(layer=int(p.stem[1:]),rows=len(rows),tail_columns=len(cols),c15=c,c2=c2,row_weight_min=float(d.min()),row_weight_max=float(d.max()),cases=cases,ratio_to_incumbent=ratios))
 print(json.dumps(dict(layer=p.stem,ratio=ratios)),flush=True)
out=dict(status='ONE_FIXED_DEPLOYABLE_WEIGHTED_PREFIX_RANK2_PROXY',formula='D=phi(alpha)/(2sigma); Q0=QR(Dvc,Dmu); Qw=QR(D Yprefix Gprefix^2 Yprefix.T D Q0), exact old term subtracted via captured inner/right. Yhat=D^-1 Qw Qw.T D Ytail; fixed one step/rank2, rank-loss retainsQ0.',parent='Exact captured QR4d basis; original exact prefix and old term identical/cancel; tail beyondc2 unchanged.',per_layer=results,pooled=pooled,pooled_ratio_to_incumbent={k:pooled['weighted_prefix_rank2'][k]/max(pooled['incumbent_QR'][k],1e-300)for k in pooled['incumbent_QR']},evidence_sha256=pins,capture_retirement_sha256=sha(ret),script_sha256=sha(__file__),seconds=time.monotonic()-begin,limits=['TRAIN16 factors only; no truth, terminal-error or model inference.','Weighted-prefix power in float64 ideal arithmetic; production float32 matmuls still require source/tiny and native validation.','No tail information enters the proposed prefix basis; tail factors used only for fixed offline scoring.','Leading Gaussian covariance sensitivity excludes higher mixed terms, learned head/birth/NP and subsequent propagation.'],no_model_calls=True,no_new_captures=True,no_truths=True,no_grid=True)
q=D/'WEIGHTED_PREFIX_PROXY_v1.json';q.open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(pooled_ratio=out['pooled_ratio_to_incumbent'],seconds=out['seconds'],sha256=sha(q))))
