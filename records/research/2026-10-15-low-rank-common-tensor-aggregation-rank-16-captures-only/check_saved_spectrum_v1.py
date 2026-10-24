from pathlib import Path
import numpy as np,json,hashlib
D=Path(__file__).resolve().parent;O=D.parents[2];p=O/'engineer/a35_join/state/s39_rank0651.npz';out=[]
with np.load(p)as z:
 for l in (9,12):
  P=z[f'old_{l:02d}'].astype(np.float64);lam=z[f'lamold_{l:02d}'].astype(np.float64);sg=np.sign(lam);ix=np.r_[0,np.flatnonzero(sg[1:]!=sg[:-1])+1,len(lam)];cs=[];hs=[];ww=[]
  for j in range(0,len(ix)-1,2):
   a,b,c=map(int,ix[j:j+3]);assert b-a==c-b and np.array_equal(lam[a:b],-lam[b:c]);cs.append((P[:,a:b]+P[:,b:c])*.5);hs.append((P[:,a:b]-P[:,b:c])*.5);ww.append(lam[a:b])
  C=np.concatenate(cs,axis=1);H=np.concatenate(hs,axis=1);w=np.concatenate(ww);Cn=np.sum(C*C,axis=0);Hn=np.sum(H*H,axis=0)
  # Weight common-direction capture by the first-order cch loss sensitivity lambda^2 * ||c||^2 * ||h||^2.
  weights=w*w*Cn*Hn;CW=C*np.sqrt(weights)[None,:];G=CW@CW.T;ev=np.linalg.eigvalsh(G)[::-1];tot=float(np.trace(G))
  # Terms below are sums of each pair's tensor-norm upper scales, not norm of their signed sum.
  cch_scale=float(np.sum(36*w*w*Cn*Cn*Hn));hhh_scale=float(np.sum(4*w*w*Hn**3))
  row=dict(layer=l,rows=P.shape[0],pairs=C.shape[1],first_order_common_capture={str(r):float(ev[:r].sum()/tot)for r in [1,2,4,8,16]},median_h_over_c=float(np.median(np.sqrt(Hn/np.maximum(Cn,1e-30)))),hhh_to_cch_pair_scale=hhh_scale/cch_scale)
  out.append(row);print(json.dumps(row),flush=True)
f=D/'SAVED_COMMON_SPECTRUM_v1.json';assert not f.exists();f.write_text(json.dumps(dict(status='HISTORICAL_S39_NATIVE651_SAVED_FACTORS_ONLY',archive=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=out,limitations=['No model, truth, fitted coefficient, or candidate.','Top-rank spectrum is oracle common subspace, not deployable basis or current output accuracy.','First-order cch weighting; cross-pair tensor cancellation and nonlinear pair-law NP terms not included.','hhh pair-scale is not signed aggregate tensor energy.']),indent=2)+'\n');print(hashlib.sha256(f.read_bytes()).hexdigest())
