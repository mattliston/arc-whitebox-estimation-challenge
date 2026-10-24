"""Admit only the fixed first training file's third moments and identity checks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from public_third_centering_v1 import central_plane, raw_f32_rounding_bound, gaussian_third_variance

HERE=Path(__file__).resolve().parent
OUT=HERE/'public_p2_first_moment_qualification_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
cert=json.loads((HERE/'public_third_centering_certificate_v1.json').read_text())
assert cert['passed'] and cert['source_sha256']==hashlib.sha256((HERE/'public_third_centering_v1.py').read_bytes()).hexdigest()
path=HERE/'public_p2_higher_moments_pilot_v1/mlp_00057.npz'
download=json.loads(path.with_suffix('.download.json').read_text())
assert download['passed'] and hashlib.sha256(path.read_bytes()).hexdigest()==download['sha256']
frozen=HERE/'public_p2_third_cell_mini0057_v1.npz'
fr=json.loads(frozen.with_suffix('.json').read_text())
assert fr['passed'] and fr['output_sha256']==hashlib.sha256(frozen.read_bytes()).hexdigest()
f=dict(np.load(frozen));W=np.load(HERE/'public_p2_third_cell_mini0057_weights_v1.npy')
assert hashlib.sha256(W.tobytes()).hexdigest()==fr['weights_array_sha256']
start=time.time()
with threadpool_limits(limits=3), np.load(path,allow_pickle=False) as z:
    allowed=('global_index','mlp_name','n_samples','sample_seed','pre_mean','pre_m2','pre_m3','pre_M11','pre_M21','mean','M11')
    data={key:z[key] for key in allowed}
    identity={key:data[key].item() for key in allowed[:4]}
    assert identity['global_index']==57 and identity['mlp_name']==fr['identity']['mlp_name']
    count=int(identity['n_samples']);assert count==100000000
    assert int(identity['sample_seed'])==(20260824 ^ (57*2654435761 % 2**63))
    for key in ('pre_mean','pre_m2','pre_m3','mean'):
        assert data[key].shape==(16,1024) and data[key].dtype==np.float64
    for key in ('pre_M11','pre_M21','M11'):
        assert data[key].shape==(16,1024,1024) and data[key].dtype==np.float32
    checks={}
    for matrix,marginal in (('pre_M11','pre_m2'),('pre_M21','pre_m3')):
        diag=np.diagonal(data[matrix],axis1=1,axis2=2)
        units=np.spacing(np.abs(diag)).astype(np.float64)
        max_units=float(np.max(np.abs(diag.astype(np.float64)-data[marginal])/np.maximum(units,1e-45)))
        checks[matrix+'_diagonal']=dict(max_float32_ulps=max_units,passed=max_units<=2)
    w0=W[0].astype(np.float64);C0=w0.T@w0;v0=np.diag(C0)
    mu0=data['pre_mean'][0]
    empiricalC0=data['pre_M11'][0].astype(np.float64)-np.outer(mu0,mu0)
    standardized_mean=mu0/np.sqrt(v0/count)
    covariance_se=np.sqrt((np.outer(v0,v0)+C0*C0)/count)
    cov_rms=float(np.sqrt(np.mean(((empiricalC0-C0)/covariance_se)**2)))
    checks['input_gaussian_covariance']=dict(standardized_rms=cov_rms,passed=cov_rms<=2)
    checks['input_gaussian_mean']=dict(standardized_rms=float(np.sqrt(np.mean(standardized_mean**2))),standardized_max=float(np.max(np.abs(standardized_mean))))
    checks['input_gaussian_mean']['passed']=checks['input_gaussian_mean']['standardized_rms']<=2 and checks['input_gaussian_mean']['standardized_max']<=12
    k0=central_plane(mu0,data['pre_m2'][0],data['pre_M11'][0],data['pre_M21'][0])
    normalized=k0/np.sqrt(gaussian_third_variance(C0,count))
    krms=float(np.sqrt(np.mean(normalized*normalized)));kmax=float(np.max(np.abs(normalized)))
    checks['input_gaussian_third']=dict(standardized_rms=krms,standardized_max=kmax,passed=krms<=1.5 and kmax<=12)
    w13=W[13].astype(np.float64)
    mu=data['pre_mean'][13];pm=data['mean'][12]
    trueC=data['pre_M11'][13].astype(np.float64)-np.outer(mu,mu)
    postC=data['M11'][12].astype(np.float64)-np.outer(pm,pm)
    mean_error=float(np.sqrt(np.mean((mu-w13.T@pm)**2)))
    cov_error=float(np.linalg.norm(trueC-w13.T@postC@w13)/np.linalg.norm(trueC))
    checks['affine_pre13']=dict(mean_rms=mean_error,covariance_relative=cov_error,passed=mean_error<=1e-6 and cov_error<=1e-5)
    K=central_plane(mu,data['pre_m2'][13],data['pre_M11'][13],data['pre_M21'][13])
    bound=raw_f32_rounding_bound(mu,data['pre_M11'][13],data['pre_M21'][13])
    residual=K-f['K21'];np.fill_diagonal(residual,0);np.fill_diagonal(bound,0)
    ratio=float(np.linalg.norm(bound)/np.linalg.norm(residual))
    checks['pre13_stored_rounding']=dict(bound_residual_rms_ratio=ratio,residual_rms=float(np.sqrt(np.mean(residual*residual))),bound_rms=float(np.sqrt(np.mean(bound*bound))),passed=ratio<=.05)
    passed=all(row['passed'] for row in checks.values())
    result=dict(passed=passed,identity=identity,checks=checks,decoded_keys=allowed,
                mean_score_targets_decoded=False,heldout_files_opened=0,source_sha256=download['sha256'],
                frozen_input_sha256=fr['output_sha256'],centering_helper_sha256=cert['source_sha256'],
                wall_s=time.time()-start,utc_finished=datetime.now(timezone.utc).isoformat(),
                qualification='One public mini training file numerical/identity qualification. Gaussian input checks have analytic sampling scales; late-layer storage bound excludes Monte Carlo error and shared forward/accumulation bias. No predictor fit or score.')
    np.savez(OUT.with_suffix('.npz'),pre13_K21=K,pre13_C=trueC,pre13_mu=mu,pre13_rounding_bound=bound)
    result['output_sha256']=hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
