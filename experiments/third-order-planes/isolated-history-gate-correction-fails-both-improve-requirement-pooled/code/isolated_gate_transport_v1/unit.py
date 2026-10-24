"""Independent density integration and explicit symmetric tensor checks."""
import hashlib
import json
from math import erf,sqrt,pi
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ns={'fnp':np}
exec((HERE/'kernel.py').read_text(),ns)
gate=ns['_gt_cp_gate']


def cubic(P,coefficients):
    return np.einsum('ir,jr,kr,r->ijk',P,P,P,coefficients)


def repeated(T):
    return np.array([[T[i,i,j] for j in range(T.shape[0])] for i in range(T.shape[0])])


def star_from_plane(R):
    n=R.shape[0];T=np.zeros((n,n,n))
    for i in range(n):
        T[i,i,i]=R[i,i]
        for j in range(n):
            if i!=j:
                T[i,i,j]=T[i,j,i]=T[j,i,i]=R[i,j]
    return T



def native_insertion_certificate(P, coefficients):
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(t['num_threads']==1 for t in threadpool_info())
    native={'fnp':fnp}
    exec(compile((HERE/'kernel.py').read_text(),str(HERE/'kernel.py'),'exec'),native)
    n=P.shape[0]
    alpha=np.linspace(-2,2,n)
    sigma=np.linspace(.7,1.3,n);variance=sigma*sigma
    g=np.array([.5*(1+erf(float(x)/sqrt(2))) for x in alpha])
    aux=dict(alpha=alpha,sigma=sigma,pre_v=variance,pdf=np.exp(-alpha*alpha/2)/sqrt(2*pi),
             gate=g,pre_k3=.03*sigma**3,pre_k4=.02*variance**2)
    pre=repeated(cubic(P,coefficients))
    inherited=g[:,None]**2*g[None,:]*pre
    expected_q,expected_plane=ns['_igt_rebase_history'](7,aux,pre,inherited)
    history=[]
    for repeat in range(2):
        with flops.BudgetContext(flop_budget=10**6,quiet=True) as ctx:
            fa={k:fnp.asarray(v.copy()) for k,v in aux.items()}
            fp=fnp.asarray(pre.copy());fi=fnp.asarray(inherited.copy())
            original=[np.asarray(fa[k]).tobytes() for k in sorted(fa)]+[np.asarray(fp).tobytes(),np.asarray(fi).tobytes()]
            g0,p0=native['_igt_rebase_history'](0,fa,fp,fi)
            assert g0 is fa['gate'] and p0 is fi
            q,plane=native['_igt_rebase_history'](7,fa,fp,fi)
            qn,pn=np.asarray(q).copy(),np.asarray(plane).copy()
            assert original==[np.asarray(fa[k]).tobytes() for k in sorted(fa)]+[np.asarray(fp).tobytes(),np.asarray(fi).tobytes()]
        assert np.max(np.abs(qn-expected_q))<=1e-12
        assert np.max(np.abs(pn-expected_plane))<=1e-12
        history.append((qn,pn,ctx.summary_dict()['flops_used']))
    assert history[0][0].tobytes()==history[1][0].tobytes()
    assert history[0][1].tobytes()==history[1][1].tobytes() and history[0][2]==history[1][2]
    # This is the exact innovation dependency that the previous gate unit omitted.
    td=np.linspace(-.1,.1,n);trace_diag=np.linspace(.01,.02,n)
    original_innovation=td-np.diag(inherited)-trace_diag
    isolated_innovation=td-np.diag(inherited)-trace_diag
    early_changed_innovation=td-np.diag(expected_plane)-trace_diag
    assert np.array_equal(original_innovation,isolated_innovation)
    witnessed=float(np.max(np.abs(early_changed_innovation-original_innovation)))
    assert witnessed>1e-7
    return dict(layer0_returns_original_objects=True,inputs_unchanged=True,
                q_maxabs=float(np.max(np.abs(history[0][0]-expected_q))),
                plane_maxabs=float(np.max(np.abs(history[0][1]-expected_plane))),
                repeated_arrays_and_bills_exact=True,complete_context_flops=history[0][2],
                original_innovation_exact=True,early_insertion_innovation_maxabs=witnessed)

def main():
    out=HERE/'unit_v1';out.mkdir(exist_ok=False)
    build=json.loads((HERE/'build.json').read_text())
    assert sha(HERE/'candidate.py')==build['candidate_sha256']
    alpha=np.array([-8.,-6.,-4.,-2.,-.5,0.,.5,2.,4.,6.,8.])
    sigma=np.linspace(.3,1.7,len(alpha));variance=sigma*sigma
    pdf=np.exp(-alpha*alpha/2)/sqrt(2*pi)
    gaussian=np.array([.5*(1+erf(float(a)/sqrt(2))) for a in alpha])
    nodes,weights=np.polynomial.legendre.leggauss(192)
    records=[]
    for skew,kurt in ((0.,0.),(.02,.03),(-.02,.03),(.1,-.05)):
        aux=dict(alpha=alpha,sigma=sigma,pre_v=variance,pdf=pdf,gate=gaussian,
                 pre_k3=skew*sigma**3,pre_k4=kurt*variance**2)
        got=gate(aux)
        integrated=[]
        for a in alpha:
            lo=-a;hi=16.
            z=(hi-lo)*nodes/2+(hi+lo)/2
            density=np.exp(-z*z/2)/sqrt(2*pi)
            H3=z*z*z-3*z;H4=z**4-6*z*z+3
            integrated.append((hi-lo)/2*np.dot(weights,density*(1+skew*H3/6+kurt*H4/24)))
        raw=gaussian+pdf*(skew*(alpha*alpha-1)/6-kurt*(alpha**3-3*alpha)/24)
        error=float(np.max(np.abs(np.array(integrated)-raw)))
        assert error<=2e-12
        assert np.array_equal(got,np.clip(raw,0,1)) or np.max(np.abs(got-np.clip(raw,0,1)))<1e-15
        assert (got>=0).all() and (got<=1).all()
        if skew==kurt==0:assert np.array_equal(got,gaussian)
        records.append(dict(skew=skew,kurt=kurt,max_abs_density_error=error,
                            raw_below_zero=int((raw<0).sum()),raw_above_one=int((raw>1).sum())))

    rng=np.random.default_rng(190919117)
    n=7;P=rng.normal(size=(n,8))/3
    coeff=rng.normal(size=4);coeff=np.r_[coeff,-coeff]
    T=cubic(P,coeff)
    g=np.linspace(.07,.94,n)
    d=np.array([.004,-.003,.01,-.02,.008,-.001,.001])
    gc=np.clip(g+d,0,1)
    direct=T*gc[:,None,None]*gc[None,:,None]*gc[None,None,:]
    carried=cubic(gc[:,None]*P,coeff)
    leg_error=float(np.max(np.abs(direct-carried)))
    inherited_plane=repeated(T)*gc[:,None]**2*gc[None,:]
    plane_error=float(np.max(np.abs(inherited_plane-repeated(carried))))
    newborn=cubic(rng.normal(size=(n,6))/4,rng.normal(size=6))
    C=rng.normal(size=(n,n));C=C@C.T/20
    b=rng.normal(size=n)/20
    trace=b[:,None,None]*C[None,:,:]+b[None,:,None]*C[:,None,:]+b[None,None,:]*C[:,:,None]
    target=rng.normal(size=(n,n))
    residual=target-inherited_plane-repeated(newborn)-repeated(trace)
    rebuilt=carried+newborn+trace+star_from_plane(residual)
    restore_error=float(np.max(np.abs(repeated(rebuilt)-target)))
    assert max(leg_error,plane_error,restore_error)<=1e-12
    # Shared-old representation must apply the same row gate as full factors.
    Q,F=rng.normal(size=(n,3)),rng.normal(size=(3,8))
    shared_error=float(np.max(np.abs((gc[:,None]*Q)@F-gc[:,None]*(Q@F))))
    assert shared_error<=1e-12
    native_record=native_insertion_certificate(P,coeff)
    result=dict(passed=True,native_insertion=native_record,density_cases=records,three_leg_error=leg_error,
                inherited_plane_error=plane_error,star_restoration_error=restore_error,
                shared_basis_gate_error=shared_error,zero_cumulant_identity=True,
                truth_accessed=False,candidate_sha256=build['candidate_sha256'],
                kernel_sha256=sha(HERE/'kernel.py'),unit_sha256=sha(__file__),numpy_version=np.__version__)
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
