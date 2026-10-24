"""Root-only tiny source-bound E1 fixture: NATIVE -B probe.py BUILD_SHA RESULT.
No estimator/source-module import; guard60s/1GB/zeroSwap with BLAS1.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

D = Path(__file__).resolve().parent
PARENT_SHA = '88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6'


def pin(path):
    path = Path(path).resolve(); b = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def extract(source, n):
    nodes = ast.parse(source).body
    fn = {node.name:node for node in nodes if isinstance(node, ast.FunctionDef)}
    env = dict(fnp=fnp, _E2_PAIR_STATE={}, _E2_PAIR_ARENA={},
               _SCRATCH=[fnp.empty(72*n*n, dtype=fnp.float32)])
    module = ast.Module(body=[fn['_e2_pair_state'], fn['_e2_pair_arena']], type_ignores=[])
    exec(compile(module, '<pinned helper definitions>', 'exec'), env)
    body = fn['_nonlinear_pair_batch'].body
    start = next(i for i,node in enumerate(body)
                 if ast.get_source_segment(source,node) == "X = _e2_A['X2']")
    code = '\n'.join(ast.unparse(node) for node in body[start:start+4])
    text = 'def actual(st, _e2_A):\n    _e2_H = lambda x:x\n    _e2_hold = []\n'
    text += '\n'.join('    '+line for line in code.splitlines())+'\n    return (T1,T2,T3,T4,T5)\n'
    exec(compile(text, '<pinned X2 statements>', 'exec'), env)
    selected = [node for node in fn['_gn_dcov'].body if isinstance(node,ast.Assign) and
                any(isinstance(t,ast.Name) and t.id in ('rho2','t21a','t31a') for t in node.targets)]
    code = '\n'.join(ast.unparse(node) for node in selected)
    text = 'def actual_gn(A, rho):\n    _e2_H = lambda x:x\n    A1,A2,A3,A4,A5 = fnp.unstack(A[1:6])\n'
    text += '\n'.join('    '+line for line in code.splitlines())+'\n    return t21a,t31a\n'
    exec(compile(text, '<pinned GN reads>', 'exec'), env)
    return env


def reference(F, G, rho, order):
    out = np.zeros((5,3,rho.shape[0],rho.shape[1]),dtype=np.float64)
    for j,(a,b) in enumerate(((2,1),(1,2),(2,2),(3,1),(1,3))):
        for r in range(order+1):
            weight = (1 if r==0 else rho if r==1 else rho*rho/2)
            for c in range(3):
                out[j,c] += weight*F[a+r,c,:,None]*G[b+r,c,None,:]
    return out


def main():
    assert len(sys.argv)==3
    target=Path(sys.argv[2]).resolve(); assert not target.exists()
    assert Path(sys.executable).absolute()==D.parent.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
    assert np.__version__=='2.4.6'
    assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    bp=pin(D/'build.json'); assert bp['sha256']==sys.argv[1]
    build=json.loads((D/'build.json').read_text())
    for key in ('parent','parent_build','candidate','builder','protocol','primitive','pricing','policy'):
        assert pin(build[key]['path'])==build[key]
    assert build['parent']['sha256']==PARENT_SHA and build['source_inverse_exact'] is True
    source=Path(build['candidate']['path']).read_text(); parent=Path(build['parent']['path']).read_text()
    inverse=source
    for edit in reversed(build['edits']):
        assert inverse.count(edit['after'])==1
        inverse=inverse.replace(edit['after'],edit['before'],1)
    assert inverse==parent
    n=4; results=[]
    F=((np.arange(7*3*n).reshape(7,3,n)%7)-3).astype(np.float32)
    G=((np.arange(7*3*n).reshape(7,3,n)*3%5)-2).astype(np.float32); G[:,0]=F[:,0]
    base=((np.arange(n*n).reshape(n,n)%5)-2).astype(np.float32)/4
    base=(base+base.T)/2; np.fill_diagonal(base,0)
    for name,rho in (('zero',np.zeros_like(base)),('signed',base),('opposite',-base)):
        arms={}
        for label,text,order in (('parent',parent,2),('candidate',source,1)):
            with flops.BudgetContext(flop_budget=1000000,quiet=True) as ctx:
                env=extract(text,n); st=env['_e2_pair_state'](n); arena=env['_e2_pair_arena'](n)
                fnp.copyto(st['FT'],fnp.asarray(F)); fnp.copyto(st['GT'],fnp.asarray(G))
                fnp.copyto(st['RR'],fnp.asarray(np.stack((np.ones_like(rho),rho,rho*rho/2))[:order+1,None]))
                out=np.stack([np.asarray(x).copy() for x in env['actual'](st,arena)])
                gn=[np.asarray(x).copy() for x in env['actual_gn'](st['FT'][:,0],fnp.asarray(rho))]
                expected=reference(F.astype(np.float64),G.astype(np.float64),rho.astype(np.float64),order)
                assert out.dtype==np.float32 and np.array_equal(out.astype(np.float64),expected)
                assert np.array_equal(gn[0],out[0,0]) and np.array_equal(gn[1],out[3,0])
                assert st['P16'].shape==(3,n,n) and st['P12'].shape==(2,1,n,n)
                assert arena['X3'].shape==(4,2,3,n,n) and st['FT'].shape==(7,3,n)
                d21=base.astype(np.float64); d31=d21.T/2
                X=d21*gn[0]/2+d31*gn[1]/6
                direct=(d21*out[0,0]+d21.T*out[1,0])/2+(d31*out[3,0]+d31.T*out[4,0])/6
                assert np.allclose(X+X.T,direct,rtol=1e-14,atol=1e-14)
                arms[label]=dict(flops=int(ctx.summary_dict()['flops_used']),out=out)
        omitted=reference(F.astype(np.float64),G.astype(np.float64),rho.astype(np.float64),2)-reference(F.astype(np.float64),G.astype(np.float64),rho.astype(np.float64),1)
        assert np.array_equal(arms['parent']['out']-arms['candidate']['out'],omitted)
        assert arms['candidate']['flops'] < arms['parent']['flops']
        results.append(dict(case=name,all_channels_exact=True,gn_read_equivalent=True,
            removed_term_exact=True,parent_flops=arms['parent']['flops'],candidate_flops=arms['candidate']['flops']))
    result=dict(passed=True,candidate=build['candidate'],parent=build['parent'],build=bp,probe=pin(__file__),
        cases=results,source_inverse_exact=True,model_executed=False,release_admitted=False)
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
