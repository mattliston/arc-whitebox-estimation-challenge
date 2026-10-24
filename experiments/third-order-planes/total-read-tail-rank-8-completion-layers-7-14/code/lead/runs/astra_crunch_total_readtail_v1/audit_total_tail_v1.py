"""Fixed rank8 total-prefix completion; saved-only known-band error audit."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
CAP=O/'lead/runs/astra_native_terminal_linear_refit_v1/run_capture_s0_v1/train16_read_observer_v1'
assert all(os.environ.get(x)=='1' for x in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
def maskdiag(E,rows,cols):
 _,ri,ci=np.intersect1d(rows,cols,return_indices=True);E[ri,ci]=0
def energy(E,d,g,rows,cols):
 E=E.copy();maskdiag(E,rows,cols)
 W=.5*d[:,None]*E*g[None,:]
 C=np.zeros((1024,1024));C[np.ix_(rows,cols)]=W;C=C+C.T
 return dict(matrix=float(np.sum(E*E)),weighted_directed=float(np.sum(W*W)),symmetric_covariance=float(np.sum(C*C)))
results=[];pooled={};start=time.monotonic()
for layer in range(7,15):
 path=CAP/f'L{layer:02d}.npz'
 with np.load(path,allow_pickle=False) as z:
  A=z['left_young'].astype(float);B=z['right_young'].astype(float);lam=z['lambda_young'].astype(float)
  rows=z['rows'].astype(int);cols=z['cols'].astype(int);mu=z['mu'].astype(float);v=z['vc'].astype(float);g=z['gate'].astype(float)
  Q=z['Q'].astype(float);prefix=z['prefix'].astype(float);c15=int(z['c15']);c2=int(z['c2'])
 c1=min(c2,((int(np.sum(g>=.15865525393145707))+31)//32)*32)
 oldtail=A@(B[c15:]*lam).T;del A,B,lam
 incumbent=oldtail-Q@(Q.T@oldtail)
 d=np.exp(-.5*mu[rows]**2/v[rows])/np.sqrt(2*np.pi*v[rows])
 # One fixed attainable basis: existing vc/mu plus six deterministic cosine
 # probes, one power step on the TOTAL exact retained prefix, reduced QR.
 m=len(rows);probes=np.cos(np.pi*(np.arange(m)[:,None]+.5)*np.arange(1,7)[None,:]/m)
 seed=np.column_stack((v[rows],mu[rows],probes));Q0=np.linalg.qr(seed,mode='reduced')[0]
 retained=prefix[:,:c1];Q8=np.linalg.qr(retained@(retained.T@Q0),mode='reduced')[0]
 band=prefix[:,c1:c15];actual=band-Q8@(Q8.T@band)
 # Best rank8 masked weighted fit to the already-computed NEW band is an
 # optimistic diagnostic. It is unavailable from the retained prefix.
 weighted=d[:,None]*band*g[cols[c1:c15]][None,:]
 _,ri,ci=np.intersect1d(rows,cols[c1:c15],return_indices=True)
 observed=weighted.copy();observed[ri,ci]=0;working=observed.copy();trace=[]
 for it in range(13):
  u,s,vt=np.linalg.svd(working,full_matrices=False)
  approx=(u[:,:8]*s[:8])@vt[:8]
  err=observed-approx;err[ri,ci]=0;trace.append(float(np.sum(err*err)))
  working=observed.copy();working[ri,ci]=approx[ri,ci]
 oracle=band-approx/(d[:,None]*g[cols[c1:c15]][None,:])
 cases=dict(incumbent_complete_tail=energy(incumbent,d,g[cols[c15:]],rows,cols[c15:]),fixed_total_prefix_rank8_new_band_only=energy(actual,d,g[cols[c1:c15]],rows,cols[c1:c15]),optimistic_masked_rank8_new_band_only=energy(oracle,d,g[cols[c1:c15]],rows,cols[c1:c15]))
 for key,vals in cases.items():
  for metric,val in vals.items():pooled.setdefault(key,{}).setdefault(metric,0.);pooled[key][metric]+=val
 ratios={key:{metric:val/max(cases['incumbent_complete_tail'][metric],1e-300) for metric,val in vals.items()} for key,vals in cases.items() if key!='incumbent_complete_tail'}
 result=dict(layer=layer,m=m,c1=c1,c15=c15,c2=c2,cases=cases,ratios=ratios,masked_error_trace=trace,capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 results.append(result);print(layer,json.dumps(ratios),flush=True)
ratios={key:{metric:val/max(pooled['incumbent_complete_tail'][metric],1e-300) for metric,val in vals.items()} for key,vals in pooled.items() if key!='incumbent_complete_tail'}
doc=dict(status='FIXED_TOTAL_PREFIX_RANK8_SAVED_KNOWN_BAND_AUDIT',rank=8,cutoff=1.0,layers=[7,14],pooled=pooled,pooled_ratios=ratios,rows=results,elapsed_s=time.monotonic()-start,models=0,new_truths=0,source_capture='Historical QR4d native TRAIN16; current S2tr differs upstream and old rank at L10-12.',limits=['Captured old_right ends at c15, so candidate error on original c15:c2 band is unavailable and excluded.','Incumbent error on entire original tail is known because its old part is exact.','New-band errors alone are not mathematical lower bounds on total symmetric covariance error: symmetric cross-band errors can cancel. Directed disjoint-column errors do add.','Masked SVD13 is an optimistic local matrix approximation, not certified optimum and not final MSE.','All global diagonals excluded, preserving exact cpdiag; NPforms and bank transport untouched.','No fitting to truths, public outcomes, model call or candidate deployment.'])
(D/'SAVED_ERROR_v1.json').write_text(json.dumps(doc,indent=2)+'\n')
print('POOLED',json.dumps(ratios),flush=True)
