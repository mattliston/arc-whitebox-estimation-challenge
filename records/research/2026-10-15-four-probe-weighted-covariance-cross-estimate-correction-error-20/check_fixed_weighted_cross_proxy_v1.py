from pathlib import Path
import hashlib,json,time
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
C=O/'lead/runs/astra_native_terminal_linear_refit_v1/run_capture_s0_v1/train16_read_observer_v1'
out=D/'FIXED_WEIGHTED_CROSS_PROXY_v1.json';assert not out.exists()
Rfull=(2*np.random.default_rng(20261015).integers(0,2,size=(1024,4),dtype=np.int8)-1).astype(np.float64)
def norm(x):return float(np.linalg.norm(x))
rr=[];start=time.monotonic()
for l in range(7,15):
 with np.load(C/f'L{l:02d}.npz',allow_pickle=False)as z:
  A=z['left_young'].astype(float);B=z['right_young'].astype(float)*z['lambda_young'].astype(float)
  Q=z['Q'].astype(float);rows=z['rows'].astype(int);cols=z['cols'].astype(int);v=z['vc'].astype(float);cut=int(z['c15'])
  inner=z['old_inner'].astype(float);oldright=z['old_right'].astype(float)
  assert oldright.shape[0]==cut
 B=B[:cut];cols=cols[:cut]
 S=Q.T@A;Y0=Q@(S@B.T);E=A@B.T-Y0;old=inner@oldright.T
 K=Y0+old
 _,ri,ci=np.intersect1d(rows,cols,return_indices=True)
 E[ri,ci]=0.;K[ri,ci]=0.
 Ew=E/v[rows,None];Kw=K/v[rows,None]
 U=np.linalg.qr(Q/v[rows,None],mode='reduced')[0]
 Elow=U.T@Ew;Klow=U.T@Kw
 R=Rfull[rows];RU=R.T@U
 Es=R.T@Ew-RU@Elow;Ks=R.T@Kw-RU@Klow
 exactcross=np.sum(Ew*Kw,axis=0)/v[cols]
 lowcross=np.sum(Elow*Klow,axis=0)/v[cols]
 estimate=lowcross+np.sum(Es*Ks,axis=0)/4/v[cols]
 energy=np.sum(Ew*Ew,axis=0)/v[cols]
 delta=energy+2*exactcross;dhat=energy+2*estimate
 base=np.sum(Kw*Kw,axis=0)/v[cols]
 # Real norm correction is the actual reported feature displacement.
 true_norm=np.sqrt(np.maximum(base+delta,0))-np.sqrt(base)
 estimated_norm=np.sqrt(np.maximum(base+dhat,0))-np.sqrt(base)
 record=dict(layer=l,m=len(rows),prefix=len(cols),k=A.shape[1],old_rank=inner.shape[1],
  true_cross_norm=norm(exactcross),lowrank_cross_norm=norm(lowcross),
  cross_estimator_relative_rmse=norm(estimate-exactcross)/max(norm(exactcross),1e-30),
  correction_relative_rmse=norm(dhat-delta)/max(norm(delta),1e-30),
  correction_true_rms=float(np.sqrt(np.mean(delta**2))),
  correction_error_rms=float(np.sqrt(np.mean((dhat-delta)**2))),
  correction_sign_agreement=float(np.mean(np.sign(dhat)==np.sign(delta))),
  norm_displacement_true_rms=float(np.sqrt(np.mean(true_norm**2))),
  norm_displacement_error_rms=float(np.sqrt(np.mean((estimated_norm-true_norm)**2))),
  norm_displacement_relative_rmse=norm(estimated_norm-true_norm)/max(norm(true_norm),1e-30),
  cross_vs_energy_norm_ratio=2*norm(exactcross)/max(norm(energy),1e-30),
  orthogonal_current_fraction=norm(Kw-U@Klow)**2/max(norm(Kw)**2,1e-30),
  negative_corrected_energy_count=int(np.sum(base+dhat<0)),
  mean_energy=float(np.mean(energy)),mean_true_cross=float(np.mean(exactcross)),
  mean_estimated_cross=float(np.mean(estimate)),source_sha256=hashlib.sha256((C/f'L{l:02d}.npz').read_bytes()).hexdigest())
 rr.append(record);print(json.dumps(record),flush=True)
doc=dict(status='SAVED_QR_PREFIX_PROXY_ONLY',probe_seed=20261015,probe_count=4,
 probe_sha256=hashlib.sha256(Rfull.astype(np.int8).tobytes()).hexdigest(),
 fixed_design='Weighted U=qr(diag(1/v)Q), exact lowrank cross, 4 fixed Rademacher response-row probes of both orthogonal remainders; current and residual diagonals zeroed.',
 rows=rr,seconds=time.monotonic()-start,models=0,captures=0,truth_reads=0,
 limits=['Counterfactual rank2 approximation on saved exact prefix with exact old contribution, not production tail.',
 'Current proxy K=rank2 young+exact old only; current STAR/NP restored forms/mean terms unavailable.',
 'Exact residual energy is supplied to isolate cross-estimator noise; approximate pair energy could add further error.',
 'No ranks, seeds, bands, coefficients or outcomes selected.'],
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out.open('x').write(json.dumps(doc,indent=2)+'\n')
print('OUTPUT_SHA256',hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
