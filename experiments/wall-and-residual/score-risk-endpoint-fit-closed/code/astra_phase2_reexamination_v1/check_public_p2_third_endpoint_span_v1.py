"""Fixed in-sample third-plane endpoint-span oracle, one admitted training cell."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from current_pair_oracle_math_v1 import exact_change
from third_endpoint_span_v1 import WeightedPlaneOperator,input_basis,third_covariance_kernel

HERE=Path(__file__).resolve().parent
OUT=HERE/'public_p2_third_endpoint_span_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
cert=json.loads((HERE/'third_endpoint_span_small_v1.json').read_text())
assert cert['passed'] and cert['source_sha256']==hashlib.sha256((HERE/'third_endpoint_span_v1.py').read_bytes()).hexdigest()
paths={name:HERE/file for name,file in (
    ('inputs','public_p2_third_cell_mini0057_v1.npz'),
    ('truth','public_p2_first_moment_qualification_v1.npz'),
    ('next','public_p2_third_next_response_v1.npz'))}
data={};provenance={}
for name,path in paths.items():
    receipt=json.loads(path.with_suffix('.json').read_text())
    assert receipt['passed']
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==receipt['output_sha256']
    data[name]=dict(np.load(path));provenance[name]=digest
f=data['inputs'];W=np.load(HERE/'public_p2_third_cell_mini0057_weights_v1.npy')[14].astype(np.float64)
start=time.time()
with threadpool_limits(limits=3):
    s=f['sigma'];R=f['Cpre']/np.outer(s,s)
    H,powers,basis=input_basis(f['state'],R)
    kernel=third_covariance_kernel(R,s,f['alpha'],f['pdf'],f['gate'],f['gaussian_mean']/s)
    weights2=kernel*kernel+kernel.T*kernel.T
    weight_mean=float(weights2.mean());assert weight_mean>0
    weights2/=weight_mean
    common=(R,s,f['alpha'],f['pdf'],f['gate'],f['gaussian_mean']/s,f['host']/s)
    target=(data['truth']['pre13_K21']-f['K21'])/(s[:,None]**2*s[None,:])
    np.fill_diagonal(target,0)
    _,target_cov=exact_change(common,dS=target)
    direct=kernel*target;direct=direct+direct.T
    target_kernel_error=float(np.max(np.abs(direct-target_cov)))
    assert target_kernel_error<=1e-12
    print(json.dumps(dict(stage='input_qualified',basis_rank=basis['numerical_rank'],
                          target_weighted_plane_rms=float(np.sqrt(np.mean(weights2*target*target))),
                          target_covariance_rms=float(np.sqrt(np.mean(target_cov*target_cov))),
                          kernel_parity_max=target_kernel_error)),flush=True)
    op=WeightedPlaneOperator(H,powers,weights2,ridge=.001)
    theta,solve=op.fit(target,tolerance=1e-7,max_iterations=300)
    prediction=op.forward(theta)
    assert np.max(np.abs(np.diag(prediction)))==0
    _,pred_cov=exact_change(common,dS=prediction)
    direct=kernel*prediction;direct=direct+direct.T
    pred_kernel_error=float(np.max(np.abs(direct-pred_cov)))
    assert pred_kernel_error<=1e-12
    response=data['next']['response14']
    target_value=np.sum(W*(target_cov@W),axis=0)*response
    pred_value=np.sum(W*(pred_cov@W),axis=0)*response
    cov_ratio=float(np.sum((pred_cov-target_cov)**2)/np.sum(target_cov*target_cov))
    value_ratio=float(np.sum((pred_value-target_value)**2)/np.sum(target_value*target_value))
    weighted_ratio=float(np.sum(weights2*(prediction-target)**2)/np.sum(weights2*target*target))
    passed=solve['passed'] and cov_ratio<=.50 and value_ratio<=.50
    result=dict(passed=passed,ordinal=57,network='gregory-wilson',role='train',layer=13,next_mean_layer=14,
                samples=100000000,basis=basis,solve=solve,
                covariance_response_error_ratio=cov_ratio,next_mean_response_error_ratio=value_ratio,
                weighted_plane_error_ratio=weighted_ratio,
                target_covariance_energy=float(np.sum(target_cov*target_cov)),
                target_next_mean_energy=float(np.sum(target_value*target_value)),
                covariance_kernel_max_error=max(target_kernel_error,pred_kernel_error),
                marginal_diagonal_write_max=float(np.max(np.abs(np.diag(prediction)))),
                provenance=provenance,helper_sha256=cert['source_sha256'],
                wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                utc_finished=datetime.now(timezone.utc).isoformat(),
                qualification='One fixed regularized in-sample coefficient-span probe on a preselected training net, not cross-net prediction, an unregularized ceiling, a noise-corrected target, total covariance error, final raw/adjusted score or estimator. Actual bank remains unchanged.')
    np.savez(OUT.with_suffix('.npz'),theta=theta,H=H,target=target,prediction=prediction,
             target_covariance=target_cov,predicted_covariance=pred_cov,
             target_next_mean=target_value,predicted_next_mean=pred_value)
    result['output_sha256']=hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
