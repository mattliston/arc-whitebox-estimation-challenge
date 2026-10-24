from pathlib import Path
import hashlib,json,numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2];P=O/'engineer/a35_join/state/s39_rank0651.npz'
with np.load(P)as z:
 J=z['J_10'].astype(np.float64);lam=z['lam_10'].astype(np.float64);Qp=z['Qp_10'].astype(np.float64);core=z['core_10'].astype(np.float64);old=z['old_09'][z['prev_10']].astype(np.float64);lo=z['lamold_09'].astype(np.float64)
full=np.concatenate([old,J],axis=1);coeff=np.r_[lo,lam];sign=np.sign(coeff);ix=np.r_[0,np.flatnonzero(sign[1:]!=sign[:-1])+1,len(coeff)];blocks=[]
for j in range(0,len(ix)-1,2):
 a,b,c=map(int,ix[j:j+3]);assert b-a==c-b and np.array_equal(coeff[a:b],-coeff[b:c]);blocks.append((a,b,c))
def split(F):return np.concatenate([.5*(F[:,a:b]+F[:,b:c])for a,b,c in blocks],axis=1),np.concatenate([F[:,a:b]-F[:,b:c]for a,b,c in blocks],axis=1)
u,v=split(full);w=np.concatenate([coeff[a:b]**2 for a,b,c in blocks]);denu=float(np.sum(u*u*w));denv=float(np.sum(v*v*w));diag=(J*J)@(lam*lam)+np.sum((Qp@core)*Qp,axis=1);rows=[]
for rank,ru in [(320,256),(384,256),(384,192)]:
 selected=np.sort(np.argsort(-diag,kind='stable')[:rank]);Y=J@((lam*lam)[:,None]*J[selected].T)+Qp@(core@Qp[selected].T);Q=np.linalg.qr(Y,mode='reduced')[0];F=Q.T@full;fu,fv=split(F);uh=Q[:,:ru]@fu[:ru];vh=Q@fv
 eu=float(np.sum((u-uh)**2*w));ev=float(np.sum((v-vh)**2*w));rows.append(dict(rank=rank,ru=ru,common_weighted_residual=eu,common_weighted_relative=eu/denu,difference_weighted_residual=ev,difference_weighted_relative=ev/denv,sum_leg_weighted_residual=eu+ev))
out=dict(status='HISTORICAL_JOIN_OPERAND_PROXY_NOT_OUTPUT_OR_CURRENT_MODEL',archive=str(P),archive_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),rows=rows,common_total_energy=denu,difference_total_energy=denv,limitations=['S39 rank651 saved operands; not current physics, no truths or model calls.','Reconstruct fixed original join selector and basis in float64; no FNP numerical parity claim.','Weighted sum of common/difference factor errors is not cubic tensor/read error and not an accuracy prediction.','Only L10 incoming frame. Future W/gate transport and L13 not assessed.'])
f=D/'SAVED_COMMON192_PROXY_v1.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));print(hashlib.sha256(f.read_bytes()).hexdigest())
