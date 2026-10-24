"""Tiny actual FNP port versus independent explicit 16-tree FP64 tensors."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE = Path(__file__).resolve().parent


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def trees():
    for word in itertools.product(range(4), repeat=2):
        degree = [1 + word.count(i) for i in range(4)]; edges = []
        for j in word:
            i = next(i for i in range(4) if degree[i] == 1)
            edges.append((i,j)); degree[i] -= 1; degree[j] -= 1
        left = [i for i in range(4) if degree[i] == 1]
        edges.append(tuple(left)); yield edges


def tensor(C,g,d,e):
    n = len(g); out = np.zeros((n,)*4, dtype=np.float64)
    for edges in trees():
        degrees = [sum(i in edge for edge in edges) for i in range(4)]
        for ix in itertools.product(range(n), repeat=4):
            value = math.prod((g,d,e)[degrees[i]-1][ix[i]] for i in range(4))
            value *= math.prod(C[ix[i],ix[j]] for i,j in edges)
            out[ix] += value
    return out


def error(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    absolute = float(np.linalg.norm(actual-reference)); norm = float(np.linalg.norm(reference))
    relative = absolute / max(norm,1e-30)
    assert absolute <= 1e-8*max(1.,norm), (absolute,norm)
    assert relative <= 5e-5 or norm == 0 and absolute == 0, (relative,absolute)
    return dict(absolute_l2=absolute,relative_l2=relative)


def main():
    assert not (HERE/'unit_result.json').exists()
    build = json.loads((HERE/'build.json').read_text())
    for key in ('parent','candidate','fragment','protocol','override'):
        assert build[key] == pin(Path(build[key]['path']))
    env = dict(fnp=fnp, _mm=lambda a,b:a@b, _k31_row_scale=lambda mu,k3,v:.85+.5*mu*k3/(v*v))
    exec(compile((HERE/'fragment.py').read_bytes(), str(HERE/'fragment.py'), 'exec'),env)
    M = np.asarray([[1,.25,0,-.125],[0,.875,.125,0],[.25,0,1,.125],[0,.25,-.125,.75]],np.float64)
    C = M@M.T; Cpost = .5*C+.25*np.eye(4)
    W = np.asarray([[.5,.125,-.25,.25],[-.125,.5,.25,0],[.25,-.25,.5,.125],[.125,.25,-.125,.5]],np.float32)
    alpha = np.asarray([-.5,.25,.75,-.125]); sigma=np.sqrt(np.diag(C))
    pdf=np.exp(-alpha*alpha/2)/np.sqrt(2*np.pi); g=np.asarray([.5*math.erfc(-x/np.sqrt(2)) for x in alpha])
    rows=[]
    for name,c,p in [('full_covariance',C,pdf),('diagonal_covariance',np.diag(np.diag(C)),pdf),('zero_nonlinear_derivatives',C,np.zeros(4))]:
        d=p/sigma; e=-alpha*p/(sigma*sigma); Q=tensor(c,g,d,e)
        tree31=np.einsum('abcd,ai,bi,ci,dj->ij',Q,W.astype(np.float64),W.astype(np.float64),W.astype(np.float64),W.astype(np.float64),optimize=False)
        trace=float(np.einsum('iijj->',Q)); den=np.trace(Cpost)**2+2*np.sum(Cpost*Cpost); beta=3*trace/den
        metric=W.astype(np.float64).T@Cpost@W.astype(np.float64); v=np.diag(metric)
        delta31=tree31-beta*v[:,None]*metric; delta4=np.diag(tree31)-beta*v*v
        mu=np.asarray([.125,-.25,.375,-.125]); K=np.diag(np.asarray([.01,-.02,.03,.01])); r=.85+.5*mu*np.diag(K)/(v*v)
        with flops.BudgetContext(flop_budget=2**29,quiet=True) as budget:
            aux={k:fnp.asarray(x) for k,x in dict(gate=g,pdf=p,sigma=sigma,alpha=alpha).items()}
            source=env['_gt13_prepare'](fnp.asarray(c),aux,fnp.asarray(Cpost))
            db,ex=env['_gt13_faces'](source,fnp.asarray(W),fnp.asarray(W.astype(np.float64)),fnp.asarray(metric),fnp.asarray(mu),fnp.asarray(K))
            metrics=dict(beta=error(source[4],np.asarray(beta)),k4=error(db,delta4))
            supplied=np.asarray(ex)+(np.asarray(db)*r/v)[:,None]*metric
            metrics['q31']=error(supplied,delta31)
            old4=np.asarray([.01,.02,.03,.04]); old31=(old4*r/v)[:,None]*metric
            new31=((old4+np.asarray(db))*r/v)[:,None]*metric+np.asarray(ex)
            np.testing.assert_allclose(new31,old31+supplied,rtol=2e-13,atol=2e-13)
            assert source[0].shape==(4,4) and str(source[0].dtype)=='float32'
        assert abs(trace-beta*(den/3)) <= 1e-12*max(1.,abs(trace))
        rows.append(dict(name=name,passed=True,errors=metrics,flops=int(budget.summary_dict()['flops_used'])))
    assert env['_GT13_STATS']==dict(captures=3,consumptions=3,products=12)
    result=dict(passed=True,candidate=build['candidate'],protocol=build['protocol'],fragment=build['fragment'],unit=pin(Path(__file__).resolve()),
                cases=rows,explicit_labeled_trees=16,source_trace_identity=True,modeled_q31_compensation=True,
                small_ordinary_F32_contractions_only=True,fullwidth_D7_fidelity_not_measured=True,estimator_imported=False,no_targets=True)
    with (HERE/'unit_result.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__': main()
