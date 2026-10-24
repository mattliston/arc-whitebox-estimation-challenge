"""One fixed TRAIN16 weighted-tail rank2 oracle; no models, truths, or grid."""
from pathlib import Path
import hashlib,json,time
import numpy as np
D=Path(__file__).resolve().parent;C=D.parent/'astra_native_terminal_linear_refit_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ret=C/'ROOT_CAPTURE_RETIREMENT_v1.json';assert sha(ret)=='94dbc42175d4e7349217f6a3c37134ad83f454654fed4a17e1f58e27b6d6fde6'
assert json.loads(ret.read_text())['all_owned_PID_PPID_PGID_absent']
files=sorted((C/'run_capture_s0_v1/train16_read_observer_v1').glob('L*.npz'));assert [p.stem for p in files]==['L%02d'%l for l in range(2,15)]
begin=time.monotonic();results=[];pooled={name:dict(raw=0.,directed=0.,symmetric=0.)for name in ('incumbent_QR','weighted_tail_rank2')};pins={}
for p in files:
 pins[str(p)]=sha(p)
 with np.load(p,allow_pickle=False)as z:
  A=z['left_young'].astype('d');c=int(z['c15']);c2=int(z['c2']);B=z['right_young'][c:].astype('d')*z['lambda_young'].astype('d')
  rows=z['rows'].astype(int);cols=z['cols'][c:].astype(int);mu=z['mu'].astype('d');vc=z['vc'].astype('d');g=z['gate'][cols].astype('d');Q=z['Q'].astype('d')
 sigma=np.sqrt(vc[rows]);alpha=mu[rows]/sigma;d=np.exp(-.5*alpha*alpha)/(2*np.sqrt(2*np.pi)*sigma)
 assert np.all(d>0)and np.all(g>0)and np.isfinite(d).all()
 Y=A@B.T;del A,B
 # Eckart-Young optimum for DIRECTED, diagonally weighted tail error.
 # Symmetrized/offdiagonal covariance is evaluated separately, not claimed optimized.
 M=(d[:,None]*Y)*g[None,:];U,s,_=np.linalg.svd(M,full_matrices=False);Qw=U[:,:min(2,len(s))]
 oracle=Y-(Qw@(Qw.T@(d[:,None]*Y)))/d[:,None]
 parent=Y-Q@(Q.T@Y)
 _,ri,ci=np.intersect1d(rows,cols,return_indices=True)
 def score(E):
  E=E.copy();E[ri,ci]=0;W=d[:,None]*E*g[None,:]
  S=np.zeros((len(mu),len(mu)),dtype='d');S[np.ix_(rows,cols)]=W;S+=S.T.copy();np.fill_diagonal(S,0)
  return dict(raw=float(np.sum(E*E)),directed=float(np.sum(W*W)),symmetric=float(np.sum(S*S)))
 cases=dict(incumbent_QR=score(parent),weighted_tail_rank2=score(oracle))
 for name,values in cases.items():
  for k,v in values.items():pooled[name][k]+=v
 ratios={k:cases['weighted_tail_rank2'][k]/max(cases['incumbent_QR'][k],1e-300)for k in cases['incumbent_QR']}
 results.append(dict(layer=int(p.stem[1:]),rows=len(rows),tail_columns=len(cols),c15=c,c2=c2,rank2_weighted_tail_energy_fraction=float(np.sum(s[:2]**2)/np.sum(s*s)),row_weight_min=float(d.min()),row_weight_max=float(d.max()),cases=cases,ratio_to_incumbent=ratios))
 print(json.dumps(dict(layer=p.stem,ratio=ratios)),flush=True)
out=dict(status='ONE_FIXED_WEIGHTED_TAIL_RANK2_ORACLE',formula='D=phi(alpha)/(2sigma), G=gate. Qw=top2 left singular vectors of D*youngTail*G. Yhat=D^-1 Qw Qw.T D Y. Compare actual leading covariance offdiagonal sym(D*(Y-Yhat)*G).',parent='Exact captured QR4d basis; original exact prefix and old term identical/cancel; tail beyondc2 unchanged.',per_layer=results,pooled=pooled,pooled_ratio_to_incumbent={k:pooled['weighted_tail_rank2'][k]/max(pooled['incumbent_QR'][k],1e-300)for k in pooled['incumbent_QR']},evidence_sha256=pins,capture_retirement_sha256=sha(ret),script_sha256=sha(__file__),seconds=time.monotonic()-begin,limits=['TRAIN16 factors only; no truth, terminal-error or model inference.','SVD minimizes directed weighted Frobenius error including diagonal; reported covariance error removes overwritten global diagonal and symmetrizes. It is not a global optimum for that nonseparable metric.','Tail-aware SVD is an oracle. A weighted PREFIX power iteration is a separate deployable approximation, whose quality is not established here.','Leading Gaussian covariance sensitivity excludes higher mixed terms, learned head/birth/NP and subsequent propagation.'],no_model_calls=True,no_new_captures=True,no_truths=True,no_grid=True)
q=D/'WEIGHTED_TAIL_ORACLE_v1.json';q.open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(pooled_ratio=out['pooled_ratio_to_incumbent'],seconds=out['seconds'],sha256=sha(q))))
