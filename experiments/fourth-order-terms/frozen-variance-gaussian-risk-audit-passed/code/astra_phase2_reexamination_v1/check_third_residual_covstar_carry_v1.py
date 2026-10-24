"""Fixed natural oracle residual carry, admitted mini57 only, no fitting."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from current_pair_oracle_math_v1 import exact_change
from third_endpoint_span_v1 import third_covariance_kernel
from public_third_centering_v1 import central_plane,raw_f32_rounding_bound
from third_residual_covstar_v1 import factors,plane,carry

HERE=Path(__file__).resolve().parent
OUT=HERE/'third_residual_covstar_carry_v1.json'
assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
cert=json.loads((HERE/'third_residual_covstar_small_v1.json').read_text())
assert cert['passed'] and cert['source_sha256']==hashlib.sha256((HERE/'third_residual_covstar_v1.py').read_bytes()).hexdigest()
files={key:HERE/name for key,name in (
    ('pre13','public_p2_third_cell_mini0057_v1.npz'),
    ('consumer','public_p2_third_carry_inputs_v1.npz'),
    ('truth13','public_p2_first_moment_qualification_v1.npz'))}
data={};provenance={}
for key,path in files.items():
    receipt=json.loads(path.with_suffix('.json').read_text());assert receipt['passed']
    digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==receipt['output_sha256']
    data[key]=dict(np.load(path));provenance[key]=digest
f=data['pre13'];b=data['consumer']
W=np.load(HERE/'public_p2_third_cell_mini0057_weights_v1.npy').astype(np.float64)
moment=HERE/'public_p2_higher_moments_pilot_v1/mlp_00057.npz'
download=json.loads(moment.with_suffix('.download.json').read_text())
assert download['passed'] and hashlib.sha256(moment.read_bytes()).hexdigest()==download['sha256']
start=time.time()
with threadpool_limits(limits=3),np.load(moment,allow_pickle=False) as z:
    # Only fields admitted by this design; no official1e9 gt_mean or fourth data.
    mu=z['pre_mean'][14].astype(np.float64);m2=z['pre_m2'][14].astype(np.float64)
    raw11=z['pre_M11'][14];raw21=z['pre_M21'][14]
    pm=z['mean'][13].astype(np.float64);rawpost=z['M11'][13]
    trueC=raw11.astype(np.float64)-np.outer(mu,mu)
    postC=rawpost.astype(np.float64)-np.outer(pm,pm)
    mean_difference=float(np.sqrt(np.mean((mu-W[14].T@pm)**2)))
    covariance_difference=float(np.linalg.norm(trueC-W[14].T@postC@W[14])/np.linalg.norm(trueC))
    trueK=central_plane(mu,m2,raw11,raw21)
    error14=trueK-b['K21_14'];np.fill_diagonal(error14,0)
    bound=raw_f32_rounding_bound(mu,raw11,raw21);np.fill_diagonal(bound,0)
    rounding_ratio=float(np.linalg.norm(bound)/np.linalg.norm(error14))
    numerical=dict(affine_mean_rms=mean_difference,affine_covariance_relative=covariance_difference,
                   stored_rounding_bound_residual_ratio=rounding_ratio,
                   passed=mean_difference<=1e-6 and covariance_difference<=1e-5 and rounding_ratio<=.05)
    assert numerical['passed'],numerical
    source=data['truth13']['pre13_K21']-f['K21'];np.fill_diagonal(source,0)
    P,E,B=factors(f['Cpre'],source)
    identity=float(np.linalg.norm(plane(P,E,B)-source)/np.linalg.norm(source))
    assert identity<=1e-10
    prediction=carry(f['Cpre'],source,f['gate'],W[14])
    discarded_diagonal=prediction.diagonal().copy();np.fill_diagonal(prediction,0)
    s=b['sigma14'];R=b['Cpre14']/np.outer(s,s)
    common=(R,s,b['alpha14'],b['pdf14'],b['gate14'],b['gaussian_mean14']/s,b['host14']/s)
    denominator=s[:,None]**2*s[None,:]
    targetS=error14/denominator;predS=prediction/denominator
    target_mean,target_cov=exact_change(common,dS=targetS)
    pred_mean,pred_cov=exact_change(common,dS=predS)
    assert np.max(np.abs(target_mean))==0 and np.max(np.abs(pred_mean))==0
    target_value=np.sum(W[15]*(target_cov@W[15]),axis=0)*b['response15']
    pred_value=np.sum(W[15]*(pred_cov@W[15]),axis=0)*b['response15']
    L=third_covariance_kernel(R,s,b['alpha14'],b['pdf14'],b['gate14'],b['gaussian_mean14']/s)
    weights2=L*L+L.T*L.T
    consumers={}
    for name,target,pred in (('covariance',target_cov,pred_cov),('next_mean',target_value,pred_value)):
        baseline=float(np.sum(target*target));residual=float(np.sum((pred-target)**2))
        ratio=residual/baseline
        consumers[name]=dict(baseline_energy=baseline,prediction_energy=float(np.sum(pred*pred)),
                             residual_ratio=ratio,cosine=float(np.sum(target*pred)/np.linalg.norm(target)/np.linalg.norm(pred)),
                             passed=ratio<=.75)
    result=dict(passed=all(row['passed'] for row in consumers.values()),ordinal=57,network='gregory-wilson',
                source_pre_layer=13,consumer_pre_layer=14,next_mean_layer=15,samples=100000000,
                source_plane_identity_relative=identity,numerical_qualification=numerical,consumers=consumers,
                weighted_plane_error_ratio=float(np.sum(weights2*(predS-targetS)**2)/np.sum(weights2*targetS*targetS)),
                source_error_norm=float(np.linalg.norm(source)),discarded_arrival_diagonal_norm=float(np.linalg.norm(discarded_diagonal)),
                provenance=provenance,moment_sha256=download['sha256'],helper_sha256=cert['source_sha256'],
                wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                utc_finished=datetime.now(timezone.utc).isoformat(),
                qualification='Natural-strength oracle carry of ONLY off-diagonal K21 residual through a fixed coordinate/covariance completion and Gaussian gates. Entire bank retained. No deployable source, independent replica, noise correction, full-carrier write, final raw/adjusted score, fit or submission.')
    np.savez(OUT.with_suffix('.npz'),source_K21=source,prediction_K21=prediction,target_K21=error14,
             prediction_covariance=pred_cov,target_covariance=target_cov,
             prediction_next_mean=pred_value,target_next_mean=target_value)
    result['output_sha256']=hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
