"""Root-only passive native2 probe; prepare is stdlib, diagnostics follow predict."""
from pathlib import Path
import ast, gc, hashlib, json, os, sys, time, types

D = Path(__file__).resolve().parent
H = D.parent
PARENT = H/'hill_climb_pair_rr_inverse_v1/candidate.py'
PARENT_SHA = 'a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
EDITS = [
    ("post_v=var, post_k3=post_k3, post_k4=post_k4)\n    L['pair'] = dict(cov=cov, den=den, aux=aux, d=d, ss=ss, ss32=_ps_ss32, mgs=mgs)",
     "post_v=var, post_k3=post_k3, post_k4=post_k4, post22=post22)\n    L['pair'] = dict(cov=cov, den=den, aux=aux, d=d, ss=ss, ss32=_ps_ss32, mgs=mgs)"),
    ("            radial = radial_new\n        return fnp.asarray",
     "            radial = radial_new\n            if 1 <= l <= 14:\n                _k4_information_observe(l, C, radial, aux)\n        return fnp.asarray")]


def pin(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def checked(p):
    assert pin(p['path'])==p
    return Path(p['path']).read_bytes()


def once(p,v):
    with Path(p).open('x') as f: json.dump(v,f,indent=2,allow_nan=False); f.write('\n')


def observer_source():
    assert pin(PARENT)['sha256']==PARENT_SHA
    old=PARENT.read_text(); source=old
    for before,after in EDITS:
        assert source.count(before)==1 and after not in source
        source=source.replace(before,after,1)
    inverse=source
    for before,after in reversed(EDITS): inverse=inverse.replace(after,before,1)
    assert inverse==old
    compile(source,'passive_observer','exec')
    return source


def prepare():
    G=H/'hill_climb_rrpi_native_joint3_recovery_v1'
    pp=pin(G/'native_fit4_plan.json'); ap=pin(G/'fit4_assessment.json')
    assert pp['sha256']=='43fc8ac2cd8a82835366b4144bdedb5c91780d7758f369a10353c7fb226bf10d'
    assert ap['sha256']=='57d52b3155bb09ab7add2952f2be234109e3e7b025b6b2616952307ed4e44338'
    old=json.loads(checked(pp)); receipt=json.loads(checked(ap)); art=receipt['artifacts']['base']
    assert receipt['passed_integrity'] and old['arms']['base']['source']==pin(PARENT)
    controls={x['rank']:x for x in map(json.loads,checked(art['rows']).decode().splitlines())}
    data=[dict(rank=x['rank'],weights=x['weights'],prediction=art['predictions'][str(x['rank'])],
               flops=controls[x['rank']]['flops']) for x in old['data'][:2]]
    assert [x['rank'] for x in data]==[35,100]
    source=observer_source()
    plan=dict(parent=pin(PARENT),probe=pin(__file__),fragment=pin(D/'fragment.py'),algebra=pin(D/'algebra.py'),
        baseline_plan=pp,baseline_assessment=ap,baseline_rows=art['rows'],baseline_guard=art['guard'],
        data=data,ranks=[35,100],layers=list(range(2,16)),observer_edits=EDITS,
        observer_source_sha256=hashlib.sha256(source.encode()).hexdigest(),inverse_exact=True,
        gate=dict(each_network_relative_squared_error_at_most=.25,each_network_centered_pearson_above=.85),
        diagnostics_only=True,targets_accessed=False,parameters_fitted=False,no_candidate_export=True,
        retained_plane_bytes=14*2*1024*1024*8,delayed_math_after_budget_context=True,
        runtime=old['runtime'],light=old['light'],guard_source=old['guard_source'],seconds=150,rss_bytes=7500000000)
    once(D/'information_plan.json',plan); pp=pin(D/'information_plan.json')
    command=[old['runtime'],'-B',str(Path(__file__).resolve()),'--run',pp['path'],pp['sha256']]
    once(D/'information_commands.json',dict(plan=pp,root_only=True,
        command=[old['light'],'-B',old['guard_source']['path'],'--record',str(D/'information_guard.json'),
                 '--seconds','150','--rss-limit','7500000000',*command],
        log=str(D/'information.log'),thread_env=dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'),'1')))
    print(json.dumps(pp))


def run(path,digest):
    pp=pin(path); assert pp['sha256']==digest; plan=json.loads(checked(pp))
    for key in ('parent','probe','fragment','algebra','baseline_plan','baseline_assessment','baseline_rows','baseline_guard','guard_source'): checked(plan[key])
    assert plan['probe']==pin(__file__) and plan['ranks']==[35,100] and plan['layers']==list(range(2,16))
    assert plan['gate']==dict(each_network_relative_squared_error_at_most=.25,each_network_centered_pearson_above=.85)
    assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    import numpy as np
    import flopscope as flops
    import ctypes
    from threadpoolctl import threadpool_info
    assert str(Path(sys.executable).absolute())==plan['runtime'] and np.__version__=='2.4.6'
    assert all(x['num_threads']==1 for x in threadpool_info())
    relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
    relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t]; relief.restype=ctypes.c_size_t
    nodes=[x for x in ast.parse(checked(plan['fragment'])).body if isinstance(x,ast.FunctionDef)]
    env={'fnp':np}; exec(compile(ast.Module(body=nodes,type_ignores=[]),'offline_actual_helpers','exec'),env)
    source=observer_source(); assert hashlib.sha256(source.encode()).hexdigest()==plan['observer_source_sha256']
    stub=types.ModuleType('whestbench'); stub.BaseEstimator=type('BaseEstimator',(),{}); sys.modules['whestbench']=stub
    mod=types.ModuleType('passive_k4_information'); mod.__file__=str(PARENT); sys.modules[mod.__name__]=mod
    exec(compile(source,str(PARENT),'exec'),mod.__dict__)
    captures={}; incoming={}
    def moments(l,aux):
        assert l not in incoming
        incoming[l]={k:np.asarray(aux[k]).copy() for k in ('sigma','pdf','alpha')}
    def observe(l,C,radial,aux):
        assert l not in captures
        captures[l]=(np.asarray(aux['post_k4']).copy(),np.asarray(aux['post22']).copy(),np.asarray(C).copy(),float(np.asarray(radial)))
        moments(l,aux)
    mod._k4_information_observe=observe
    final=mod.final_scalar_features
    def final_observe(l,m,aux,w):
        moments(l,aux); return final(l,m,aux,w)
    mod.final_scalar_features=final_observe
    est=mod.Estimator(); est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
    def metrics(full,approx):
        full=np.asarray(full).ravel(); approx=np.asarray(approx).ravel(); error=approx-full
        energy=float(full@full); ae=float(approx@approx); f=full-full.mean(); a=approx-approx.mean()
        fv=float(f@f); av=float(a@a); good=energy>0 and fv>0 and av>0
        return dict(relative_squared_error=float(error@error)/energy if energy>0 else None,
                    centered_pearson=float(f@a)/(fv*av)**.5 if fv>0 and av>0 else None,
                    cosine=float(full@approx)/(energy*ae)**.5 if energy>0 and ae>0 else None,
                    full_energy=energy,informative=good)
    records=[]; began=time.monotonic()
    for item in plan['data']:
        checked(item['weights']); checked(item['prediction']); assert not captures and not incoming
        W=np.load(item['weights']['path'],allow_pickle=False); assert W.shape==(16,1024,1024)
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=[W[l] for l in range(16)],seed=0,name=f"phase2-{item['rank']}")
        t=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as budget:
            prediction=np.asarray(est.predict(mlp,2**41),dtype=np.float64).copy()
        bill=int(budget.summary_dict()['flops_used']); expected=np.load(item['prediction']['path'],allow_pickle=False)
        assert prediction.tobytes()==expected.tobytes() and prediction.shape==(16,1024) and np.isfinite(prediction).all()
        assert bill==item['flops'] and set(captures)==set(range(1,15)) and set(incoming)==set(range(1,16))
        assert getattr(mod,'_W5_PORT_SCOPE',None) is None
        last=getattr(mod,'_W5_PORT_LAST',None)
        if last is not None: assert last['completed'] and last['numeric_cache_retired']
        layers=[]; fs=[]; rs=[]; ds=[]; weights=[]
        for l in range(2,16):
            packet=captures.pop(l-1); M=env['_k4r_residual'](*packet); state=env['_k4r_project'](M)
            w=W[l].astype(np.float32); Q=w.astype(np.float64)**2
            full=np.sum(Q*(M@Q),axis=0); rowdiag=env['_k4r_delta'](w,state); diagonal=np.diag(M)@(Q*Q)
            assert all(np.isfinite(x).all() for x in (M,full,rowdiag,diagonal))
            a=incoming[l]; weight=(a['alpha']**2-1)*a['pdf']/(24*a['sigma']**3)
            layers.append(dict(layer=l,rowdiag=metrics(full,rowdiag),diagonal=metrics(full,diagonal)))
            fs.append(full); rs.append(rowdiag); ds.append(diagonal); weights.append(weight)
            del packet,M,state,Q,w
        full,rowdiag,diagonal,weight=map(np.concatenate,(fs,rs,ds,weights))
        quality=metrics(full,rowdiag)
        passed=quality['informative'] and quality['relative_squared_error']<=.25 and quality['centered_pearson']>.85
        row=dict(rank=item['rank'],full_bill_exact=True,all16_bytes_exact=True,flops=bill,layers=layers,
                 rowdiag=quality,diagonal=metrics(full,diagonal),quality_passed=bool(passed),
                 scalar_mean_read_descriptive=dict(rowdiag=metrics(full*weight,rowdiag*weight),diagonal=metrics(full*weight,diagonal*weight)),
                 elapsed_s=time.monotonic()-t)
        records.append(row); once(D/f"information_rank{item['rank']}.json",row); print(json.dumps(row),flush=True)
        incoming.clear(); del W,mlp,prediction,expected,fs,rs,ds,weights,full,rowdiag,diagonal,weight,a
        gc.collect(); relief(None,0); assert not captures and not incoming
    once(D/'information_result.json',dict(completed=True,quality_passed=all(x['quality_passed'] for x in records),rows=records,
         plan=pp,elapsed_s=time.monotonic()-began,targets_accessed=False,arrays_persisted=False,
         diagnostic_only=True,accuracy_gain_measured=False,candidate_bill_gain_measured=False))


if __name__=='__main__':
    if sys.argv[1:]==['--prepare']: prepare()
    else:
        assert len(sys.argv)==4 and sys.argv[1]=='--run'
        run(sys.argv[2],sys.argv[3])
