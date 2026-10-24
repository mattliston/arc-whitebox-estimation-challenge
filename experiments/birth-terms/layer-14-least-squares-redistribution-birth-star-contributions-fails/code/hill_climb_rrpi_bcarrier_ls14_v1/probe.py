"""Root-only paid original-versus-held LS14 fixture; no estimator imports."""
from pathlib import Path
import ast, hashlib, importlib.util, json, os, sys

D = Path(__file__).resolve().parent


def main():
    assert len(sys.argv) == 2
    spec=importlib.util.spec_from_file_location('held_ls14_source',D/'prepare.py')
    p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
    parent,candidate,proof=p.source(); fragment=proof['appended_fragment']
    assert fragment['sha256']==sys.argv[1] and not (D/'primitive_result.json').exists()
    shared=p.common(); assert str(Path(sys.executable).absolute())==shared['runtime']
    assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    tree=ast.parse(p.checked(parent)); classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_E2Hold']
    assert len(classes)==1
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    assert np.__version__=='2.4.6'
    original=dict(fnp=fnp); held=dict(fnp=fnp)
    exec(compile(p.checked(proof['reference_helper']),proof['reference_helper']['path'],'exec'),original)
    exec(compile(ast.Module(body=classes,type_ignores=[]),'actual_E2Hold','exec'),held)
    exec(compile(p.checked(fragment),fragment['path'],'exec'),held)
    n=8
    C4=np.array([[2.,.25,-.125,.0625],[.25,1.5,.375,-.25],[-.125,.375,1.,.125],[.0625,-.25,.125,.75]])
    C=np.kron(np.eye(2),C4)
    b=np.tile(np.array([.0625,-.125,.1875,-.25]),2)
    R32=((np.arange(n*n).reshape(n,n)%7)-3).astype(np.float32)/16
    rd=np.tile(np.array([.0312500001,-.0625000002,.0937500003,-.1250000004]),2)
    cases=[]

    def plane(covariance,vector):
        v=np.diag(covariance)
        return v[:,None]*vector[None,:]+2*covariance*vector[:,None]

    def paid(env,inputs):
        with flops.BudgetContext(flop_budget=1000000,quiet=True) as budget:
            values=tuple(fnp.asarray(x) for x in inputs)
            snapshots=[np.asarray(x).tobytes() for x in values]
            result=env['_bcls14'](*values)
            arrays=[np.asarray(x).copy() for x in result]
            assert snapshots==[np.asarray(x).tobytes() for x in values]
            bill=int(budget.summary_dict()['flops_used']); assert 0<bill<1000000
        return arrays,bill

    def run(name,covariance,residual,residual_diag):
        inputs=(covariance,b,residual,residual_diag); snapshots=[x.tobytes() for x in inputs]
        R=residual.astype(np.float64); np.fill_diagonal(R,residual_diag)
        design=np.column_stack([plane(covariance,np.eye(n)[j]).reshape(-1) for j in range(n)])
        rhs=design.T@R.reshape(-1); normal=design.T@design
        direction=rhs/np.maximum(np.diag(normal),1e-30)
        step=(rhs@direction)/max(float(direction@normal@direction),1e-30)
        expected_delta=1.0*step*direction
        reference,bill0=paid(original,inputs); answer,bill1=paid(held,inputs)
        assert [x.tobytes() for x in reference]==[x.tobytes() for x in answer] and bill0==bill1
        assert snapshots==[x.tobytes() for x in inputs]
        bb,rr,dd=answer
        assert all(np.isfinite(x).all() for x in answer)
        assert np.allclose(bb-b,expected_delta,rtol=2e-12,atol=2e-12)
        assert np.allclose(dd,residual_diag-3*np.diag(covariance)*expected_delta,rtol=2e-12,atol=2e-12)
        corrected=rr.astype(np.float64); np.fill_diagonal(corrected,dd)
        before=plane(covariance,b)+R; after=plane(covariance,bb)+corrected
        assert np.allclose(before,after,rtol=2e-7,atol=2e-8)
        assert np.linalg.norm(R-plane(covariance,expected_delta))<=np.linalg.norm(R)+1e-12
        star=3*rr.T; np.fill_diagonal(star,dd.astype(np.float32))
        represented=star.T.astype(np.float64)/3; np.fill_diagonal(represented,np.diag(star))
        assert np.allclose(plane(covariance,bb)+represented,before,rtol=4e-7,atol=5e-8)
        if name in ('zero_residual','zero_covariance'): assert bb.tobytes()==b.tobytes()
        cases.append(dict(case=name,n=n,passed=True,original_bill=bill0,held_bill=bill1,
                          outputs_bytes_exact=True,max_plane_error=float(np.max(np.abs(after-before)))))

    run('signed_symmetric',C,R32,rd)
    skew=C.copy(); skew[0,1]+=1e-7
    run('rounded_nonsymmetric',skew,R32,rd)
    run('zero_residual',C,np.zeros_like(R32),np.zeros(n))
    run('zero_covariance',np.zeros_like(C),R32,rd)
    rng=np.random.default_rng(20260930)
    random_cov=rng.normal(size=(n,n))*.2; random_cov+=np.eye(n)
    run('fixed_random_nonsymmetric',random_cov,rng.normal(size=(n,n)).astype(np.float32)*.05,rng.normal(size=n)*.03)
    result=dict(passed=True,parent=parent,candidate=candidate,fragment=fragment,reference=proof['reference_helper'],
         probe=p.pin(__file__),cases=cases,all_outputs_bytes_exact=True,full_bills_exact=True,
         all_inputs_unchanged=True,held_helper_ast_inverse_exact=True,no_estimator_import=True,
         rounding_matches_reference=True,server_free_trip_saving_not_claimed=True,model_executed=False)
    p.once(D/'primitive_result.json',result)
    print(json.dumps(dict(passed=True,cases=cases),allow_nan=False))


if __name__=='__main__': main()
