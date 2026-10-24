"""Saved-data ceiling only: fixed six coherent dial directions, LORO, no model run.
Not a release estimate: source is historical S2tr, current lineage must be tested.
"""
import json
from pathlib import Path
import numpy as np

O = Path(__file__).resolve().parents[3]
K = O / 'lead/runs/opus_kinkoracle_v1'
OUT = Path(__file__).resolve().parent
RANKS = (35, 150, 200, 250, 259, 277, 287)
ARMS = ('gk4_08', 'ssb_085', 'gsk_09', 'bfix119', 'k4l34_04', 'gne_12_1012')
XX, AA, BB, WW, GG = [], [], [], [], []
for r in RANKS:
    d = np.load(K / f'res_dump_{r}_dump.npz')
    a = np.load(O / f'analyst/fresh/pu/mps_truth_rank{r:04d}_A32M.npz')
    b = np.load(O / f'analyst/fresh/pu/mps_truth_rank{r:04d}_B32M.npz')
    v, mu = d['pre_v14'], d['mu14']
    z = mu / np.sqrt(v)
    keep = np.abs(z) < 1
    x = np.stack([(np.load(K / f'res_d_{arm}_{r}_dump.npz')['pre_v14'] - v) / v for arm in ARMS], axis=1)
    XX.append(x[keep]); AA.append(((a['mc_pre_v'][14] - v) / v)[keep]); BB.append(((b['mc_pre_v'][14] - v) / v)[keep])
    WW.append((np.exp(-z*z) * v)[keep]); GG.append(np.full(keep.sum(), r))
X, A, B, W, G = map(np.concatenate, (XX, AA, BB, WW, GG))
pred = np.zeros_like(A); rows=[]
for r in RANKS:
    tr, te = G != r, G == r
    scale = np.sqrt(np.sum(W[tr,None]*X[tr]**2,axis=0)/W[tr].sum())
    Xt=X[tr]/scale; Xe=X[te]/scale
    gram=Xt.T@(W[tr,None]*Xt)
    # Fixed 1% ridge; no intercept and no hyperparameter selection.
    coef=np.linalg.solve(gram+0.01*np.trace(gram)/len(ARMS)*np.eye(len(ARMS)),Xt.T@(W[tr]*0.5*(A[tr]+B[tr]))) / scale
    pred[te]=X[te]@coef
    base=float(np.sum(W[te]*A[te]*B[te])); new=float(np.sum(W[te]*(A[te]-pred[te])*(B[te]-pred[te])))
    rows.append(dict(rank=r,units=int(te.sum()),share_recovered_pct=100*(1-new/base),coef=dict(zip(ARMS,map(float,coef)))))
base=float(np.sum(W*A*B)); new=float(np.sum(W*(A-pred)*(B-pred)))
res=dict(scope='Historical S2tr saved-data ceiling; not banked vs v53j/v61j',arms=ARMS,ranks=RANKS,n_units=len(A),ridge=0.01,cv='Leave-one-rank-out',share_recovered_pct=100*(1-new/base),optimistic_final_adjusted_pct=-9.3*(1-new/base),rows=rows)
out=OUT/'COHERENT_COV_PROBE_v1.json'
assert not out.exists();out.write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
