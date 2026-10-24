"""Fixed spectral128+exact residual diagonal feasibility; saved TRAIN16 only."""
from pathlib import Path
import hashlib, json, os
import numpy as np

D=Path(__file__).resolve().parent
O=D.parents[2]
assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
out=D/'TRAIN16_SPECTRAL128_DIAGONAL_v1.json'
assert not out.exists()
protocol=json.loads((O/'lead/runs/astra_current_variance_capture_v1/CAPTURE_PROTOCOL_HELD_v1.json').read_text())
wp=Path(protocol['weights_path'])
assert hashlib.sha256(wp.read_bytes()).hexdigest()==protocol['weights_raw_sha256']
W=np.load(wp,allow_pickle=False)
assert W.shape==(16,1024,1024)
evidence={str(wp):protocol['weights_raw_sha256']}
rows=[]
for layer in (13,14,15):
    p=O/f'physicist/t128_cdump/t128_C_16_L{layer}.npz'
    evidence[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    with np.load(p,allow_pickle=False) as z:
        C=z['C'].astype(float);m=z['mu'].astype(float)
    C=(C+C.T)*.5
    lam,V=np.linalg.eigh(C)
    idx=np.argsort(-np.abs(lam),kind='stable')[:128]
    U=V[:,idx];vals=lam[idx]
    low=(U*vals)@U.T
    residual_diag=np.diag(C)-np.diag(low)
    approx=low.copy();approx[np.diag_indices_from(C)]+=residual_diag
    E=approx-C
    w=W[layer].astype(float)
    pushed=w.T@C@w
    error=w.T@E@w
    v=np.diag(pushed);dv=np.diag(error);mu=m@w
    sigma=np.sqrt(np.maximum(v,1e-20));a=mu/sigma
    dmean=np.exp(-.5*a*a)/(2*np.sqrt(2*np.pi)*sigma)*dv
    off=C.copy();np.fill_diagonal(off,0)
    rec=dict(next_layer=layer,rank=128,
        trace=float(np.trace(C)),minimum_eigenvalue=float(lam[0]),
        residual_diagonal_min=float(residual_diag.min()),
        exact_diagonal_max_abs=float(np.max(abs(np.diag(E)))),
        offdiag_relative_Frobenius=float(np.linalg.norm(E)/np.linalg.norm(off)),
        next_covariance_relative_Frobenius=float(np.linalg.norm(error)/np.linalg.norm(pushed)),
        next_variance_relative_rms=float(np.sqrt(np.mean((dv/v)**2))),
        next_variance_relative_mean=float(np.mean(dv/v)),
        next_variance_relative_max_abs=float(np.max(abs(dv/v))),
        gaussian_mean_linear_disturbance_mse=float(np.mean(dmean*dmean)),
        gaussian_mean_linear_disturbance_rms=float(np.sqrt(np.mean(dmean*dmean))),
        post_mu_rms=float(np.sqrt(np.mean(m*m))),
        predicted_next_mean_rms=float(np.sqrt(np.mean(mu*mu))))
    rows.append(rec)
    print(json.dumps(rec),flush=True)
doc=dict(status='OFFLINE_FIXED_RANK128_ORACLE',rows=rows,evidence_sha256=evidence,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    formula='C128 = V128 diag(lambda128) V128.T + diag(diag(C)-diag(V128 diag(lambda128) V128.T)); largest absolute eigenvalues, fixed128; exact real-arithmetic old C diagonal',
    limits=['Historical T128 S47e plain Torch carried covariance, not current QR physics.',
            'Exact spectral oracle is not deployable at its full eigendecomposition price.',
            'Derivative-weighted mean disturbance is a local Gaussian sensitivity, not measured final MSE or a fit.',
            'No truth arrays, model calls, captures, parameter sweep or candidate selection.'])
out.write_text(json.dumps(doc,indent=2)+'\n')
print('sha256',hashlib.sha256(out.read_bytes()).hexdigest())
