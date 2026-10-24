from pathlib import Path
import hashlib,json,time
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
C=O/'lead/runs/astra_native_terminal_linear_refit_v1/run_capture_s0_v1/train16_read_observer_v1'
F=O/'lead/runs/astra_native_intermediate_mean_refit_v1/run_capture_s0_v1/native_16_capture_capture.npz'
out=D/'TRAIN16_ENERGY_BIAS_v1.json';assert not out.exists()
res=[];t0=time.monotonic()
with np.load(F,allow_pickle=False)as f:
 for path in sorted(C.glob('L*.npz')):
  l=int(path.stem[1:])
  if l<7:continue
  with np.load(path,allow_pickle=False)as z:
   A=z['left_young'].astype(float);B=z['right_young'].astype(float)*z['lambda_young'].astype(float);Q=z['Q'].astype(float)
   rows=z['rows'].astype(int);cols=z['cols'].astype(int);v=z['vc'].astype(float);cut=int(z['c15']);cols=cols[cut:];B=B[cut:]
  S=Q.T@A;Y=A@B.T;Y0=Q@(S@B.T);E=Y-Y0
  _,ri,ci=np.intersect1d(rows,cols,return_indices=True);E[ri,ci]=0
  Z=E/v[rows,None]/np.sqrt(v[cols])[None,:]
  x=f[f'L{l}_features30'].astype(float)
  er=np.sum(Z*Z,axis=1);ec=np.sum(Z*Z,axis=0)
  # Exact residual norms bound the displacement of full K21 row/column norm features.
  rowupper=np.sqrt(er)/np.maximum(x[rows,22],1e-30);colupper=np.sqrt(ec)/np.maximum(x[cols,23],1e-30)
  # Energy-only repair's change to existing norm; this is NOT exact norm repair,
  # because 2<K_current,E> is unavailable in the saved captures.
  rowadd=np.sqrt(x[rows,22]**2+er)-x[rows,22];coladd=np.sqrt(x[cols,23]**2+ec)-x[cols,23]
  rowfrac=rowadd/np.maximum(x[rows,22],1e-30);colfrac=coladd/np.maximum(x[cols,23],1e-30)
  res.append(dict(layer=l,m=len(rows),tail=len(cols),k=A.shape[1],residual_standardized_energy=float(np.sum(Z*Z)),full_row_norm_energy=float(np.sum(x[:,22]**2)),full_col_norm_energy=float(np.sum(x[:,23]**2)),residual_energy_fraction=float(np.sum(Z*Z)/np.sum(x[:,22]**2)),norm_change_upper_median=[float(np.median(rowupper)),float(np.median(colupper))],norm_change_upper_max=[float(np.max(rowupper)),float(np.max(colupper))],energy_only_fraction_median=[float(np.median(rowfrac)),float(np.median(colfrac))],energy_only_fraction_max=[float(np.max(rowfrac)),float(np.max(colfrac))],input=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  print(l,res[-1]['residual_energy_fraction'],res[-1]['energy_only_fraction_max'],flush=True)
doc=dict(status='TRAIN16_READ_RESIDUAL_NORM_BIAS_BOUND',rows=res,seconds=time.monotonic()-t0,feature_file=str(F),feature_sha256=hashlib.sha256(F.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits=['Exact saved QR parent only, same TRAIN16 network; HF1c913 reassociates pair arithmetic so this is a physics-proxy diagnostic, not current-source exact identity.','No truth targets, learned fit or model/capture call.','Norm energy correction omits unavailable full-K21 cross term; it is a magnitude diagnostic, not an unbiased estimator or proposed patch.','Residual diagonal is zero because production restores exact k3 diagonals.'])
out.write_text(json.dumps(doc,indent=2)+'\n');print('sha',hashlib.sha256(out.read_bytes()).hexdigest())
