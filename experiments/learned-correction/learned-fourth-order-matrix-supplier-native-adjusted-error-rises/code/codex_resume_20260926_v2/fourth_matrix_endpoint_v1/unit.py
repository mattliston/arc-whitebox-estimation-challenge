from pathlib import Path
import ast,hashlib,json,math,sys,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent;SUP=H2/'fourth_matrix_learned_supplier_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((HERE/'build.json').read_text());assert sha(HERE/'candidate.py')==b['candidate_sha256']
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
m=types.ModuleType('matrix_native_unit');m.__file__=str(HERE/'candidate.py');sys.modules[m.__name__]=m
exec(compile((HERE/'candidate.py').read_text(),m.__file__,'exec'),m.__dict__)
env={'np':np};source=(SUP/'model.py').read_text();tree=ast.parse(source)
for name in ('numpy_pair','numpy_diag'):
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.get_source_segment(source,node),str(SUP/'model.py'),'exec'),env)
hp=P2/'codex_resume_20260919_v1/joint_incoming_moment_head_v1/head.py';ref={}
exec(compile(hp.read_text(),str(hp),'exec'),ref)
params={};norm={}
for key,dest in [('model',params),('normalizers',norm)]:
    path=Path(b[key]['path']);assert sha(path)==b[key]['sha256']
    with np.load(path) as z:dest.update({k:z[k] for k in z.files})
pd={k:v.astype(np.float64) for k,v in params.items()};rng=np.random.default_rng(39331);n=9
A=rng.normal(size=(n,n));C=A@A.T/n+np.eye(n);C[0,1]+=1e-5
mu=rng.normal(size=n);K=rng.normal(size=(n,n))*.02;W=rng.normal(size=(n,n)).astype(np.float32)/np.sqrt(n)
H=rng.normal(size=(n,64)).astype(np.float32)*.2;radial=.015
cdf=np.vectorize(lambda x:.5*math.erfc(-float(x)/math.sqrt(2)),otypes=[float])
rows=[]
for layer in (8,11,14):
    F,R,S,s=ref['features'](np,cdf,layer,mu,C,K,radial,W,H)
    F=np.clip(F/norm['feature_scale'],-8,8);R=R/norm['rho_scale'];S=S/norm['third_scale']
    i,j=np.indices((n,n));i=i.ravel();j=j.ravel()
    e=np.stack((R[i,j],R[j,i],S[i,j],S[j,i]),axis=1).astype(np.float64)
    D=(env['numpy_pair'](F[i].astype(np.float64),F[j].astype(np.float64),e,pd)*float(norm['off_scale'])).reshape(n,n)
    np.fill_diagonal(D,env['numpy_diag'](F.astype(np.float64),pd)*float(norm['diag_scale']))
    a=np.diag(D);v=np.diag(C);rawR=C/np.outer(s,s);Rs=(rawR+rawR.T)/2;np.fill_diagonal(Rs,1.)
    d4=a*s**4;q31=(D+a[:,None]*Rs)/2;q22=(a[:,None]+a[None,:]+4*Rs*D)/6
    coefficients=m._K31R_TABLE.get(layer)
    rscale=np.ones(n) if coefficients is None else coefficients[0]+coefficients[1]*mu*np.diag(K)/(v*v)
    expected31=(q31-(d4/(v*v)*rscale)[:,None]*rawR)*((v*s)[:,None]*s[None,:])
    expected22=q22*np.outer(v,v);np.fill_diagonal(expected31,0);np.fill_diagonal(expected22,0)
    expected=(d4,expected31,expected22);m._PHYS_LAYER[0]=layer
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        args=[fnp.asarray(x) for x in (mu,C,K,W,H)]
        actual=[np.array(x) for x in m._fm_moments(layer,args[0],args[1],args[2],radial,args[3],args[4])]
    errors=[]
    for x,y in zip(actual,expected):
        err=float(np.max(np.abs(x-y)));tol=1e-9+2e-5*float(np.max(np.abs(y)))
        assert np.isfinite(x).all() and err<=tol,(layer,err,tol);errors.append(dict(max_abs=err,tolerance=tol))
    assert np.array_equal(np.diag(actual[1]),np.zeros(n)) and np.array_equal(np.diag(actual[2]),np.zeros(n))
    # Explicit effective31 increment includes the automatic shift from k4.
    effective=actual[1]/((v*s)[:,None]*s[None,:])+(actual[0]/(v*v)*rscale)[:,None]*rawR
    np.fill_diagonal(effective,0);np.fill_diagonal(q31,0)
    overlap=float(np.max(np.abs(effective-q31)));assert overlap<1e-9+2e-5*float(np.max(np.abs(q31)))
    rows.append(dict(layer=layer,errors=errors,radial31_overlap_error=overlap,flops=int(ctx.summary_dict()['flops_used'])))
result=dict(passed=True,rows=rows,all_parameter_bytes_exact=b['all_parameter_bytes_exact'],source_inverse=b['exact_source_inverse'],
            candidate_sha256=sha(HERE/'candidate.py'),model_sha256=b['model']['sha256'],truth_accessed=False)
with (HERE/'unit_result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
