"""Full saved-covariance spectrum oracle; no SDK or runtime eig proposal."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

D=Path(__file__).resolve().parent
A=D.parent
capture=A/'PRICE_CAPTURE_RANK16_v1.npz'
assert hashlib.sha256(capture.read_bytes()).hexdigest()=='563744cd09dcaeb68243c986eaf8f4b05082c0b914c5acb50791d6e78806fbea'
z=np.load(capture)
started=time.monotonic()
rows=[]; factors={}
for l in (13,14):
    m=z[f'L{l}_host'].astype(float)
    dm=np.maximum(z[f'L{l}_delta'].astype(float),-m)
    C0=z[f'L{l}_cov_after_edge'].astype(float)
    rc=-.08-.5438712708150119
    diag0=np.diag(C0).copy(); md=m*dm
    diag=((diag0-md)-md)-dm*dm
    diag=diag-rc*(diag0-diag)
    m32,dm32,c32=m.astype(np.float32),dm.astype(np.float32),C0.astype(np.float32)
    md32=np.outer(m32,dm32)
    C=((c32-md32)-md32.T)-np.outer(dm32,dm32)
    C=C-np.float32(rc)*(c32-C)
    C=C.astype(float);np.fill_diagonal(C,diag)
    C=(C+C.T)/2
    lam,V=np.linalg.eigh(C)
    neg=lam<0
    Vn=V[:,neg]; e=-lam[neg]
    factors[f'L{l}_vectors']=Vn
    factors[f'L{l}_magnitudes']=e
    rows.append(dict(layer=l, min_eigenvalue=float(lam[0]), negative_modes=int(neg.sum()),
        modes_below_minus1e8=int((lam<-1e-8).sum()), negative_mass=float(e.sum()),
        negative_mass_over_trace=float(e.sum()/np.trace(C)),
        negative_squared_mass=float(e@e), covariance_squared_mass=float(np.sum(C*C)),
        first20_eigenvalues=lam[:20].tolist(),
        top1_negative_mass_share=float(e[0]/e.sum()) if e.sum() else 0,
        top4_negative_mass_share=float(e[:4].sum()/e.sum()) if e.sum() else 0,
        top8_negative_mass_share=float(e[:8].sum()/e.sum()) if e.sum() else 0,
        negative_mode_absmean_projection=(np.abs(Vn.T@m)/np.linalg.norm(m)).tolist()))
np.savez(D/'PSD_NEGATIVE_FACTORS_v1.npz',**factors)
result=dict(status='SPECTRUM_ORACLE_NOT_DEPLOYABLE',rank=16,capture_sha256='563744cd09dcaeb68243c986eaf8f4b05082c0b914c5acb50791d6e78806fbea',
    current_best=1.4468565569762618e-9,rows=rows,elapsed_s=time.monotonic()-started,
    maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    no_model=True,no_sdk=True,no_gain_banked=True)
with(D/'PSD_SPECTRUM_RESULT_v1.json').open('x')as f:json.dump(result,f,indent=2)
for row in rows:row.pop('negative_mode_absmean_projection')
print(json.dumps(result,indent=2))
