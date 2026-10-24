"""Natural conditional isotropic-weight quadratic action; no regression."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
OUT=HERE/'conditional_weight_value_attribution_r35_v1.json'
ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
current=np.load(HERE/'current_pair_oracle_attribution_v2/inputs_rank_0035.npz')
old=np.load(HERE/'rows_covariance_innovation_probe_v2/row_0035/data.npz')
start=time.time();records=[];saves={};baseline_energy=0.;error_energy=0.;mean_energy=0.
with threadpool_limits(limits=3):
    for index,l in enumerate(range(8,15)):
        E=old['target'][index].astype(np.float64)
        W=old['weight'][index].astype(np.float64)
        response=old['response'][index].astype(np.float64)
        m=current['host'][index].astype(np.float64)
        n=len(m);u=m/np.linalg.norm(m)
        z=W.T@u
        perpendicular=np.sum(W*W,axis=0)-z*z
        eu=float(u@E@u);trace=float(np.trace(E))
        isotropic=response*trace*np.sum(W*W,axis=0)/n
        meanray=response*(eu-trace/n)*(z*z-perpendicular/(n-1))
        projected=isotropic+meanray
        direct=response*np.sum(W*(E@W),axis=0)
        signal=float(direct@direct)
        baseline_energy+=signal
        error_energy+=float(np.sum((direct-projected)**2))
        mean_energy+=float(meanray@meanray)
        raw_expected=eu*z*z+(trace-eu)*perpendicular/(n-1)
        assert np.allclose(projected,response*raw_expected,rtol=1e-12,atol=1e-18)
        arms={}
        for name,delta in (('two_statistic',projected),('trace_only',isotropic),('mean_ray_only',meanray)):
            arms[name]=dict(error_ratio=float(np.sum((direct-delta)**2)/signal),
                            relative_energy=float(delta@delta/signal),
                            cosine=float(direct@delta/np.sqrt(signal*(delta@delta))))
        row=dict(layer=l,baseline_next_mean_mse=signal/n,trace_error=trace,
                 mean_ray_covariance_error=eu,arms=arms)
        records.append(row);saves[f'value_{l}']=direct;saves[f'trace_{l}']=isotropic;saves[f'meanray_{l}']=meanray
        print(json.dumps(row),flush=True)
ratio=error_energy/baseline_energy
relative_mean=mean_energy/baseline_energy
np.savez(ART,**saves)
record=dict(rank=35,layers=list(range(8,15)),records=records,
            pooled_two_statistic_error_ratio=ratio,pooled_mean_ray_relative_energy=relative_mean,
            consumer_geometry_gate=ratio<=.5 and relative_mean>=.25,
            wall_s=time.time()-start,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='Exact conditional unweighted quadratic form for independent isotropic Gaussian weights; multiplication by actual w-dependent saved response is an approximation. Original8M coherent-source target and pre-target host mean. Natural no-fit oracle attribution across all7 fixed cells, not a deployable covariance estimate or score.')
OUT.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='records'},indent=2),flush=True)
