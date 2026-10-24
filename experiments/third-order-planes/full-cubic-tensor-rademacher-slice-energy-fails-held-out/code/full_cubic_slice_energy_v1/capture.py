"""Metered passive full-tensor feature capture; reference arithmetic is research only."""
import argparse,ast,importlib.util,json,sys,types,time,resource,gc,hashlib
from common import HERE,D,PARENT,PARENT_SHA,TRAIN,read,write,pin,sha,verify

def instrument():
    source=PARENT.read_text()
    node=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_predict_binary'][-1]
    original=ast.get_source_segment(source,node);changed=original
    edits=[
      ('        W = mlp.weights','        W = mlp.weights\n        _fcse_probes = _FCSE_MAKE()'),
      ('                m = m + final_hook(l, m, aux, W[l])',
       '                _FCSE_CAPTURE(l, pp, lam, w32, C, previous_alive, star, bpre, aux, shared_old["cohorts"], _fcse_probes)\n                m = m + final_hook(l, m, aux, W[l])'),
      ('            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)',
       '            _fcse_previous_C = C\n            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)\n            if l >= 8:\n                _FCSE_CAPTURE(l, pp, lam, w32, _fcse_previous_C, previous_alive, star, bpre, aux, shared_old["cohorts"], _fcse_probes)\n            del _fcse_previous_C')]
    for a,b in edits:assert changed.count(a)==1;changed=changed.replace(a,b,1)
    restored=changed
    for a,b in reversed(edits):restored=restored.replace(b,a,1)
    assert restored==original;ast.parse(changed)
    return changed,dict(inverse_byte_exact=True,original_sha256=hashlib.sha256(original.encode()).hexdigest(),instrumented_sha256=hashlib.sha256(changed.encode()).hexdigest())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qualification',action='store_true');parser.add_argument('--rank',type=int);args=parser.parse_args()
    assert args.qualification != (args.rank is not None)
    verify();assert sha(PARENT)==PARENT_SHA
    assert read(HERE/'unit.json')['passed']
    if not args.qualification:
        assert args.rank in TRAIN and read(HERE/'qualification/result.json')['qualified']
        g=read(HERE/'qualification_watch.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    from feature_fnp import slices,probes
    from reference import reference
    assert all(x['num_threads']==1 for x in threadpool_info())
    inputs=read(HERE/'inputs.json');rows=inputs['qualification'] if args.qualification else [inputs['train'][str(args.rank)]]
    out=HERE/('qualification' if args.qualification else f'captures/rank{args.rank:04d}');out.mkdir(parents=True,exist_ok=False)
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    spec=importlib.util.spec_from_file_location('fcse_unchanged_lazy',PARENT);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    observed,proof=instrument();tap={};records=[];first=None;bill=0;setup_bill=0;details=[]
    def make():
        nonlocal bill,setup_bill
        before=ctx.summary_dict()['flops_used'];answer=probes()
        setup_bill=ctx.summary_dict()['flops_used']-before;bill+=setup_bill
        return answer
    def capture(l,pp,lam,w,C,alive,A,b,aux,cohorts,gh):
        nonlocal bill
        before=ctx.summary_dict()['flops_used'];t0=time.monotonic()
        z,energy=slices(pp,lam,w,C,alive,A,b,aux['sigma'],cohorts,*gh)
        elapsed=time.monotonic()-t0;charged=ctx.summary_dict()['flops_used']-before;bill+=charged
        zn=np.asarray(z);en=np.asarray(energy)
        t1=time.monotonic()
        ref=reference(*[np.asarray(v) for v in [pp,lam,w,C,alive,A,b,aux['sigma']]],cohorts,*[np.asarray(v) for v in gh])
        er=float(np.linalg.norm(zn-ref)/max(np.linalg.norm(ref),1e-30))
        ee=float(np.linalg.norm(en-np.mean(ref*ref,axis=1))/max(np.linalg.norm(np.mean(ref*ref,axis=1)),1e-30))
        assert ctx.summary_dict()['flops_used']==before+charged
        assert er<=.001 and ee<=.002,(l,er,ee)
        tap[l]=dict(z=z,energy=energy,sigma=aux['sigma'],alpha=aux['alpha'],pdf=aux['pdf'])
        details.append(dict(layer=l,flops=charged,producer_wall_s=elapsed,reference_wall_s=time.monotonic()-t1,
                            z_relative=er,energy_relative=ee,columns=int(pp.shape[1]),cohorts=len(cohorts)))
    for call,row in enumerate(rows):
        rank=row['rank'];wp=row['weights']['path'];weights=np.load(wp,allow_pickle=False)
        assert sha(wp)==row['weights']['sha256'] and weights.shape==(16,1024,1024) and weights.dtype==np.float32
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(weights[l]) for l in range(16)]
        assert ingress.summary_dict()['flops_used']==0
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
        tap.clear();details.clear();bill=0;setup_bill=0;started=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            if call==0:
                spec.loader.exec_module(mod);mod._FCSE_MAKE=make;mod._FCSE_CAPTURE=capture
                exec(compile(observed,str(HERE/'instrumented.py'),'exec'),mod.__dict__)
                original_step=mod.gnn_step
                def step(l,x,sigma,w,state,P):
                    answer=original_step(l,x,sigma,w,state,P)
                    if l>=8:tap[l]['features']=x
                    return answer
                mod.gnn_step=step;estimator=mod.Estimator()
            prediction=estimator.predict(mlp,2**41)
        F=ctx.summary_dict()['flops_used'];wall=time.monotonic()-started
        assert F-bill==row['flops'] and bill==setup_bill+sum(d['flops'] for d in details)
        assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
        assert set(tap)==set(range(8,16))
        arrays={f'{k}_{l:02d}':np.asarray(v) for l,fields in tap.items() for k,v in fields.items()}
        arrays['prediction']=np.asarray(prediction)
        assert arrays['prediction'].tobytes()==np.load(row['prediction']['path'],allow_pickle=False).tobytes()
        assert all(np.isfinite(v).all() for v in arrays.values())
        # Exact original feature and scalar inputs, using prior independently captured parent.
        with np.load(row['parent_capture']['path'],allow_pickle=False) as zold:
            for l in range(8,16):
                for k in ['features','sigma','alpha','pdf']:
                    assert arrays[f'{k}_{l:02d}'].tobytes()==zold[f'{k}_{l:02d}'].tobytes(),(k,l)
        digests={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
        if args.qualification:
            if call==0:first=digests
            if call==2:assert digests==first
        path=out/f'call_{call:02d}_rank{rank:04d}.npz';np.savez(path,**arrays)
        overhead=bill/row['flops'];qualified=overhead<=.075
        record=dict(call=call,rank=rank,capture=pin(path),digests=digests,whole_flops=F,parent_flops=row['flops'],extra_flops=bill,
            extra_flop_ratio=overhead,cost_gate_passed=qualified,probe_setup_flops=setup_bill,wall_includes_research_reference_s=wall,
            includes_initialization=call==0,all16_parent_bytes_and_bill_exact=True,numeric_cache_retired=True,
            old_parent_feature_bytes_exact=True,details=list(details),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        records.append(record);print(json.dumps({k:v for k,v in record.items() if k not in ['digests','details']}),flush=True)
        tap.clear();del arrays,prediction,weights,ww,mlp;gc.collect()
    result=dict(completed=True,qualified=all(x['cost_gate_passed'] for x in records),records=records,source_proof=proof,parent_sha256=PARENT_SHA,
        truth_accessed=False,repeat_all_capture_bytes_exact=bool(args.qualification),full_estimator_calls=len(rows),
        capture_bytes=sum(x['capture']['bytes'] for x in records),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out/'result.json',result)
if __name__=='__main__':main()
