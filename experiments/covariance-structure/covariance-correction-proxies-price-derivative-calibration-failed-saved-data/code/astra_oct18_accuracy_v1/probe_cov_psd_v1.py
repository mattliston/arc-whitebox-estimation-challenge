"""Saved native covariance feasibility, no candidate or model execution."""
import json
from pathlib import Path
import numpy as np

O=Path(__file__).resolve().parents[3]
D=O/'lead/runs/opus_k4unit_v1/cap'
rows=[]
for rank in (16,35,150,200,250,300):
    d=np.load(D/f'cap_rank{rank}.npz')
    row=dict(rank=rank)
    for name in ('L14_cov','L15_C'):
        C=d[name].astype(np.float64)
        eig,U=np.linalg.eigh(0.5*(C+C.T))
        neg=eig < -1e-12*np.max(np.abs(eig))
        diag=np.diag(C)
        item=dict(min_eigenvalue=float(eig[0]),max_eigenvalue=float(eig[-1]),negative_count=int(neg.sum()),negative_trace=float(-eig[neg].sum()),trace=float(diag.sum()),asym_relative=float(np.linalg.norm(C-C.T)/np.linalg.norm(C)),negative_diagonals=int((diag<0).sum()),cauchy_excess_max=float(np.max(np.abs(C)-np.sqrt(np.maximum(0,diag[:,None]*diag[None,:])))))
        if name=='L15_C':
            W=d['L15_W']
            vc=np.einsum('ij,ij->j',W,C@W)
            item['captured_v_relative_discrepancy']=float(np.linalg.norm(vc-d['L15_v'])/np.linalg.norm(vc))
            vfix=((U[:,neg].T@W)**2).T@(-eig[neg])
            item['psd_clip_var_change_rms_relative']=float(np.sqrt(np.mean((vfix/d['L15_v'])**2)))
            item['psd_clip_var_change_max_relative']=float(np.max(vfix/d['L15_v']))
        row[name]=item
    rows.append(row)
res=dict(scope='Saved native S2tr snapshots; diagnostic only, no current score claim',rows=rows)
out=Path(__file__).resolve().parent/'COV_PSD_AUDIT_v1.json'; assert not out.exists();out.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
