"""A posteriori ridge optimization uncertainty; no additional fitting iterations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from third_endpoint_span_v1 import WeightedPlaneOperator,input_basis,third_covariance_kernel

HERE=Path(__file__).resolve().parent
OUT=HERE/'third_endpoint_solve_bound_v1.json'
assert not OUT.exists()
rng=np.random.default_rng(930082)


def bounds(op,theta,target,L,W,response):
    residual=op.normal(theta)-op.adjoint(op.weights2*target)/op.scale
    rnorm=float(np.linalg.norm(residual))
    plane_bound=np.sqrt(op.scale)*rnorm/np.sqrt(op.ridge)
    wmean=float(np.mean(L*L+L.T*L.T))
    covariance_bound=2*np.sqrt(wmean)*plane_bound
    gram=(W.T@W)**2*np.outer(response,response)
    value_operator_bound=float(np.sqrt(np.max(np.sum(np.abs(gram),axis=1))))
    return dict(normal_residual_norm=rnorm,weighted_plane_frobenius_bound=plane_bound,
                covariance_frobenius_bound=covariance_bound,value_operator_norm_bound=value_operator_bound,
                next_mean_l2_bound=covariance_bound*value_operator_bound)


with threadpool_limits(limits=3):
    n,k=9,3
    H=np.linalg.qr(rng.normal(size=(n,k)))[0]*np.sqrt(n)
    R=rng.normal(size=(n,n))*.1
    powers=np.stack((np.ones_like(R),R,R*R))
    for p in powers:np.fill_diagonal(p,0)
    L=rng.normal(size=(n,n))*.01;np.fill_diagonal(L,0)
    weights2=L*L+L.T*L.T;weights2/=weights2.mean()
    op=WeightedPlaneOperator(H,powers,weights2)
    target=rng.normal(size=(n,n))*.02;theta=rng.normal(size=op.shape)*.003
    count=int(np.prod(op.shape))
    D=np.stack([op.forward(v.reshape(op.shape)).ravel() for v in np.eye(count)],axis=1)
    A=D.T@(weights2.ravel()[:,None]*D)/(n*n)+op.ridge*np.eye(count)
    exact=np.linalg.solve(A,D.T@(weights2*target).ravel()/(n*n)).reshape(op.shape)
    W=rng.normal(size=(n,n))/np.sqrt(n);response=rng.uniform(.1,1,size=n)
    error=op.forward(theta-exact);dc=L*error;dc=dc+dc.T
    dv=np.sum(W*(dc@W),axis=0)*response
    small=bounds(op,theta,target,L,W,response)
    actual=dict(weighted_plane=float(np.linalg.norm(np.sqrt(weights2)*error)),covariance=float(np.linalg.norm(dc)),next_mean=float(np.linalg.norm(dv)))
    small_pass=actual['weighted_plane']<=small['weighted_plane_frobenius_bound']*(1+1e-10) and actual['covariance']<=small['covariance_frobenius_bound']*(1+1e-10) and actual['next_mean']<=small['next_mean_l2_bound']*(1+1e-10)
    assert small_pass
    fit_path=HERE/'public_p2_third_endpoint_span_v1.npz'
    fr=json.loads(fit_path.with_suffix('.json').read_text())
    assert fr['output_sha256']==hashlib.sha256(fit_path.read_bytes()).hexdigest()
    fitted=dict(np.load(fit_path));f=dict(np.load(HERE/'public_p2_third_cell_mini0057_v1.npz'))
    nextrow=dict(np.load(HERE/'public_p2_third_next_response_v1.npz'))
    W=np.load(HERE/'public_p2_third_cell_mini0057_weights_v1.npy')[14].astype(np.float64)
    s=f['sigma'];R=f['Cpre']/np.outer(s,s)
    H,powers,_=input_basis(f['state'],R);assert np.array_equal(H,fitted['H'])
    L=third_covariance_kernel(R,s,f['alpha'],f['pdf'],f['gate'],f['gaussian_mean']/s)
    weights2=L*L+L.T*L.T;weights2/=weights2.mean()
    op=WeightedPlaneOperator(H,powers,weights2)
    b=bounds(op,fitted['theta'],fitted['target'],L,W,nextrow['response14'])
    observed=float(np.linalg.norm(fitted['predicted_next_mean']-fitted['target_next_mean']))
    target_norm=float(np.linalg.norm(fitted['target_next_mean']))
    uncertainty=1.001*b['next_mean_l2_bound']
    lower=max(0,observed-uncertainty)**2/(target_norm*target_norm)
    record=dict(small_certificate_passed=small_pass,small_bounds=small,small_actual=actual,
                wide=b,observed_next_mean_residual_norm=observed,target_next_mean_norm=target_norm,
                uncertainty_safety_factor=1.001,exact_ridge_optimum_value_error_ratio_lower_bound=lower,
                cannot_reach_declared_value_gate=lower>.50,additional_fit_iterations=0,
                unchanged_fit_sha256=fr['output_sha256'],utc_finished=datetime.now(timezone.utc).isoformat(),
                qualification='Numerical algebraic bound for the unique optimum of the same fixed weighted K21 ridge objective. Not an interval arithmetic certificate, statistical confidence bound, unregularized representation ceiling, cross-net result or raw/adjusted score.')
    OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
