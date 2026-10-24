"""Actual public-FNP mixed-precision test of the literal converter v2."""
import ast, hashlib, json, math, resource
from pathlib import Path
import numpy as np
import flopscope as fs
import flopscope.numpy as fn
D=Path(__file__).resolve().parent;R=D.parents[1]
fragment=R/'astra_crunch_physics_v1/global_radius_v1/fragment_converter_v2.py.txt'
env={'fnp':fn,'_rc_math':math,'_ps32':lambda x:x if str(x.dtype)=='float32' else x.astype(fn.float32)}
tree=ast.parse(fragment.read_text());nodes=[]
for x in tree.body:
    if isinstance(x,ast.Assign) and all(isinstance(t,ast.Name) and t.id.startswith('_GR_') and t.id!='_GR_PARENT_PREDICT' for t in x.targets):nodes.append(x)
    if isinstance(x,ast.FunctionDef) and x.name=='_gr_convert_first':nodes.append(x)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(fragment),'exec'),env)
helpers={};tree=ast.parse((D/'check_analytic_v1.py').read_text());nodes=[]
for x in tree.body:
    if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='results' for t in x.targets):break
    nodes.append(x)
helpers['__file__']=str(D/'check_analytic_v1.py')
exec(compile(ast.Module(body=nodes,type_ignores=[]),'analytic helpers','exec'),helpers)
# Use the exact marginal formula and deterministic correlated Gaussian moments.
uni=helpers['uni_raw'];pair=helpers['pair_raw']
checks=[]
for rho in (-1.,-.9,0.,.3,.95,1.):
    scale=np.array([.3,2.7]); n=1024
    g=[np.array([uni(None,k)*s**k for s in scale]) for k in range(1,5)]
    s=[np.array([uni(n,k)*q**k for q in scale]) for k in range(1,5)]
    def rawmat(dim,p,q):
        return np.array([[(uni(dim,p+q) if i==j else pair(dim,p,q,rho))*scale[i]**p*scale[j]**q for j in range(2)] for i in range(2)])
    g11,g21,g22=[rawmat(None,*pq) for pq in ((1,1),(2,1),(2,2))]
    s11,s21,s22=[rawmat(n,*pq) for pq in ((1,1),(2,1),(2,2))]
    mg,r2g,r3g,r4g=g;ms,r2s,r3s,r4s=s
    planes=(g11-np.outer(mg,mg),g21-np.outer(r2g,mg),g22-np.outer(r2g,r2g))
    with fs.BudgetContext(flop_budget=2**60,quiet=True):
        pp=[fn.asarray(x.astype(np.float32)) for x in planes];vv=[fn.asarray(x) for x in g]
        m,cov,p21,p22,r2,r3,r4=env['_gr_convert_first'](*pp,*vv)
        m32=env['_ps32'](m)
        k21=p21-2*m32[:,None]*cov
        k22=p22-2*m32[:,None]*p21.T-2*m32[None,:]*p21+4*fn.outer(m32,m32)*cov-2*cov*cov
        v=r2-m*m;k3=r3-3*m*r2+2*m**3;k4=r4-4*m*r3+6*m*m*r2-3*m**4-3*v*v
        fn.fill_diagonal(cov,v);fn.fill_diagonal(k21,k3);fn.fill_diagonal(k22,k4)
    cs=s11-np.outer(ms,ms);vs=r2s-ms*ms
    ref21=s21-2*ms[:,None]*s11-ms[None,:]*r2s[:,None]+2*ms[:,None]**2*ms[None,:]
    ref22=s22-2*ms[None,:]*s21-2*ms[:,None]*s21.T+4*np.outer(ms,ms)*s11+np.outer(ms*ms,r2s)+np.outer(r2s,ms*ms)-3*np.outer(ms*ms,ms*ms)-np.outer(vs,vs)-2*cs*cs
    arrays=[np.asarray(x) for x in (cov,k21,k22)]
    errs=[float(np.max(np.abs(x-y))) for x,y in zip(arrays,(cs,ref21,ref22))]
    assert max(errs)<2e-5,(rho,errs)
    assert all(np.isfinite(x).all() for x in arrays)
    symmetry=float(max(np.max(np.abs(arrays[0]-arrays[0].T)),np.max(np.abs(arrays[2]-arrays[2].T))))
    assert symmetry<1e-6
    checks.append(dict(rho=rho,max_abs_errors=errs,symmetry=symmetry,dtypes=[str(x.dtype) for x in arrays]))
old=(fragment.parent/'fragment_converter_v1.py.txt').read_text()
assert fragment.read_text()==old.replace('math.exp','_rc_math.exp').replace('math.log','_rc_math.log').replace('math.lgamma','_rc_math.lgamma')
out=dict(status='PASS',fragment_sha256=hashlib.sha256(fragment.read_bytes()).hexdigest(),cases=checks,v2_changes_only_math_alias=True,tariff_v1_applies_unchanged=True,max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(D/'LITERAL_v2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
