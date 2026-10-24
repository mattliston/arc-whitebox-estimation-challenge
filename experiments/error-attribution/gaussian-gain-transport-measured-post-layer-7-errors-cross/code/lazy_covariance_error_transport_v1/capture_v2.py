"""Passive capture; only reference bindings are added inside the parent."""
import ast,importlib.util,json,resource,sys,time,types,gc,hashlib
from common import HERE,D,SOURCE,PARENT_SHA,sha,write,verify_seal,weight_path,pin
def observed_source():
    s=SOURCE.read_text();assert sha(SOURCE)==PARENT_SHA
    n=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='_predict_binary'][-1]
    original=ast.get_source_segment(s,n);changed=original
    edits=[('                m = m + final_hook(l, m, aux, W[l])',
        '                m = m + final_hook(l, m, aux, W[l])\n                _CET_TAP[l] = dict(m=m,mu=mu,sigma=aux["sigma"],alpha=aux["alpha"],pdf=aux["pdf"],gate=aux["gate"],mg=aux["gaussian_mean"],v=v)'),
        ('            previous, previous_alive = (live_indices, alive)',
        '            if l >= 7:\n                _CET_TAP[l] = dict(m=m,C_post=C,C_pre=Cpre,mu=mu,sigma=aux["sigma"],alpha=aux["alpha"],pdf=aux["pdf"],gate=aux["gate"],mg=aux["gaussian_mean"])\n            previous, previous_alive = (live_indices, alive)')]
    for old,new in edits:assert changed.count(old)==1;changed=changed.replace(old,new,1)
    back=changed
    for old,new in reversed(edits):back=back.replace(new,old,1)
    assert back==original;ast.parse(changed)
    return changed,dict(function='_predict_binary',original_sha256=hashlib.sha256(original.encode()).hexdigest(),observed_sha256=hashlib.sha256(changed.encode()).hexdigest(),inverse_byte_exact=True)
def main():
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    verify_seal();verify_seal('dependency_fix_sealed.json');assert json.loads((HERE/'unit.json').read_text())['passed']
    guard=json.loads((HERE/'unit_v2_watch.json').read_text());assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
    out=HERE/'capture';out.mkdir(exist_ok=False);source,proof=observed_source()
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    spec=importlib.util.spec_from_file_location('lcet_lazy_parent',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    baselines=json.loads((D/'lazy_rotation_v1/dev/result.json').read_text())['records']
    held={};records=[];first_digests=None
    for call,rank in enumerate((35,100,35)):
        wp=weight_path(rank);weights=np.load(wp,allow_pickle=False)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(weights[l]) for l in range(16)]
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
        started=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            if call==0:
                spec.loader.exec_module(mod);mod._CET_TAP=held;exec(compile(source,str(HERE/'observed.py'),'exec'),mod.__dict__);estimator=mod.Estimator()
            prediction=estimator.predict(mlp,2**41)
        elapsed=time.monotonic()-started;flop=ctx.summary_dict()['flops_used']
        assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
        assert set(held)==set(range(7,16))
        arrays={f'{key}_{l:02d}':np.asarray(value) for l,row in held.items() for key,value in row.items() if not(l==7 and key=='C_pre')}
        arrays['prediction']=np.asarray(prediction)
        assert all(np.isfinite(v).all() for v in arrays.values())
        baseline=baselines[call];assert baseline['rank']==rank and baseline['weights_sha256']==sha(wp)
        assert flop==baseline['flops'] and np.array_equal(arrays['prediction'],np.load(baseline['output'],allow_pickle=False))
        digests={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes(order='C')).hexdigest()) for k,v in arrays.items()}
        if call==0:first_digests=digests
        if call==2:assert digests==first_digests
        dest=out/f'call_{call:02d}.npz';np.savez(dest,**arrays)
        record=dict(call=call,rank=rank,capture=pin(dest),array_digests=digests,parent_flops=flop,parent_wall_s=elapsed,input_flops=ingress.summary_dict()['flops_used'],weights=pin(wp),parent_all16_and_bill_exact=True,numeric_cache_retired=True)
        records.append(record);print(json.dumps({k:v for k,v in record.items() if k!='array_digests'}),flush=True)
        held.clear();del arrays,prediction,weights,ww,mlp;gc.collect()
    total=sum(r['capture']['bytes'] for r in records);assert total<500_000_000
    result=dict(completed=True,records=records,source_proof=proof,parent_sha256=PARENT_SHA,repeat_all_captured_arrays_exact=True,truth_accessed=False,extra_parent_operations=0,capture_total_bytes=total,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out/'result.json',result)
if __name__=='__main__':main()
