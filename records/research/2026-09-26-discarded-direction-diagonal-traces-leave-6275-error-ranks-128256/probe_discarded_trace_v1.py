"""TRAIN-only virtual read-rank cut and first-chaos compensation diagnostic.

Uses existing unchanged-S30 captures, never Monte Carlo targets. This is not a
candidate, native endpoint score, or metered cost claim. In particular it does
not simulate future propagation or the current L15 signed-pair split.
"""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
folder=HERE/'old_diagonal_capture_train2_v1'
records=[]
for path in sorted(folder.glob('old_diag_rank*_layer*.npz')):
    with np.load(path) as f:
        U=f['U'].astype(np.float64);F=f['F'].astype(np.float64)
        lam=f['lam'].astype(np.float64);alpha=f['alpha'];observed=f['diagonal']
    P=U@F
    full=np.sum((P*P*P)*lam,axis=1)
    del P
    rho=np.mean(U*U)
    variances=np.mean(U*U,axis=0)
    covariance=U.T@U/len(U)
    full_traces={
        'isotropic':3*rho*(U@(F@(lam*np.sum(F*F,axis=0)))),
        'coordinate_variance':3*(U@(F@(lam*((F*F).T@variances)))),
        'full_covariance_expensive':3*(U@(F@(lam*np.sum(F*(covariance@F),axis=0))))}
    entry=dict(path=path.name,shape=list(F.shape),captured_diagonal_rounding_relative_rms=float(
        np.linalg.norm(full-observed)/np.linalg.norm(observed)),cuts=[])
    for r in (128,192,256):
        if r>=U.shape[1]:continue
        Us,Fs=U[:,:r],F[:r,:]
        P=Us@Fs
        short=np.sum((P*P*P)*lam,axis=1)
        del P
        short_traces={
            'isotropic':3*rho*(Us@(Fs@(lam*np.sum(Fs*Fs,axis=0)))),
            'coordinate_variance':3*(Us@(Fs@(lam*((Fs*Fs).T@variances[:r])))),
            'full_covariance_expensive':3*(Us@(Fs@(lam*np.sum(Fs*(covariance[:r,:r]@Fs),axis=0))))}
        difference=full-short
        masks={'all':np.ones(len(U),dtype=bool),'response_abs_alpha_below_2p33':np.abs(alpha)<2.33,
               'positive_tail':alpha>2.5}
        metrics={}
        for name,trace in full_traces.items():
            repair=trace-short_traces[name]
            metrics[name]={}
            for label,mask in masks.items():
                error=difference[mask]
                metrics[name][label]=dict(
                    residual_squared_fraction=float(np.sum((error-repair[mask])**2)/np.sum(error**2)),
                    error_correlation=float(np.corrcoef(error,repair[mask])[0,1]))
        entry['cuts'].append(dict(rank=r,unrepaired_relative_squared_error=float(np.sum(difference**2)/np.sum(full**2)),metrics=metrics))
    records.append(entry)
    print(json.dumps(entry),flush=True)
    (HERE/'discarded_trace_probe_train2_v1.json').write_text(json.dumps(dict(
        scope=__doc__,records=records),indent=2)+'\n')
