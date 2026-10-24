"""Passive combined feature and physical-state capture of exact accepted Lazy."""
import argparse,ast,importlib.util,json,sys,types,time,resource,gc,hashlib
from common import HERE,D,PARENT,PARENT_SHA,TRAIN,read,write,pin,sha,verify
def instrument():
    source=PARENT.read_text();node=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_predict_binary'][-1]
    original=ast.get_source_segment(source,node);changed=original
    edits=[('                m = m + final_hook(l, m, aux, W[l])',
      '                m = m + final_hook(l, m, aux, W[l])\n                _MIO_TAP[l] = dict(m=m,mu=mu,sigma=aux["sigma"],alpha=aux["alpha"],pdf=aux["pdf"],gate=aux["gate"],mg=aux["gaussian_mean"],v=v)'),
      ('            previous, previous_alive = (live_indices, alive)',
      '            if l >= 7:\n                _MIO_TAP[l] = dict(m=m,C_post=C,C_pre=Cpre,mu=mu,sigma=aux["sigma"],alpha=aux["alpha"],pdf=aux["pdf"],gate=aux["gate"],mg=aux["gaussian_mean"])\n            previous, previous_alive = (live_indices, alive)')]
    for a,b in edits:assert changed.count(a)==1;changed=changed.replace(a,b,1)
    restored=changed
    for a,b in reversed(edits):restored=restored.replace(b,a,1)
    assert restored==original;ast.parse(changed)
    return changed,dict(inverse_byte_exact=True,original_sha256=hashlib.sha256(original.encode()).hexdigest(),observed_sha256=hashlib.sha256(changed.encode()).hexdigest())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qualification',action='store_true');parser.add_argument('--rank',type=int);args=parser.parse_args()
    assert args.qualification != (args.rank is not None);verify();assert sha(PARENT)==PARENT_SHA
    if not args.qualification:
        assert args.rank in TRAIN and read(HERE/'capture_qualification/result.json')['completed']
        g=read(HERE/'capture_qualification_watch.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    inputs=read(HERE/'inputs.json');rows=inputs['qualification'] if args.qualification else [inputs['train'][str(args.rank)]]
    out=HERE/('capture_qualification' if args.qualification else f'captures/rank{args.rank:04d}');out.mkdir(parents=True,exist_ok=False)
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    spec=importlib.util.spec_from_file_location('lmio_unchanged_lazy',PARENT);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    observed,proof=instrument();tap={};neural={};records=[];first=None
    for call,row in enumerate(rows):
        rank=row['rank'];wp=row['weights']['path'];weights=np.load(wp,allow_pickle=False)
        assert sha(wp)==row['weights']['sha256'] and weights.shape==(16,1024,1024) and weights.dtype==np.float32
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(weights[l]) for l in range(16)]
        assert ingress.summary_dict()['flops_used']==0
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
        tap.clear();neural.clear();started=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            if call==0:
                spec.loader.exec_module(mod);mod._MIO_TAP=tap;exec(compile(observed,str(HERE/'observed.py'),'exec'),mod.__dict__)
                original_step=mod.gnn_step
                def step(l,x,sigma,w,state,P):
                    answer=original_step(l,x,sigma,w,state,P)
                    if l>=8:neural[l]=dict(features=x,hidden=answer[1])
                    return answer
                mod.gnn_step=step;estimator=mod.Estimator()
            prediction=estimator.predict(mlp,2**41)
        F=ctx.summary_dict()['flops_used'];wall=time.monotonic()-started
        assert F==row['flops'] and mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
        assert set(tap)==set(range(7,16)) and set(neural)==set(range(8,16))
        arrays={f'{k}_{l:02d}':np.asarray(v) for l,fields in tap.items() for k,v in fields.items() if not(l==7 and k=='C_pre')}
        arrays.update({f'{k}_{l:02d}':np.asarray(v) for l,fields in neural.items() for k,v in fields.items()})
        arrays['prediction']=np.asarray(prediction)
        assert arrays['prediction'].tobytes()==np.load(row['prediction']['path'],allow_pickle=False).tobytes()
        assert all(np.isfinite(v).all() for v in arrays.values())
        for l in range(8,16):
            assert arrays[f'features_{l:02d}'].shape==(1024,30) and arrays[f'hidden_{l:02d}'].shape==(1024,64)
            assert arrays[f'features_{l:02d}'].dtype==arrays[f'hidden_{l:02d}'].dtype==np.float32
        digests={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
        common_exact=False
        if args.qualification:
            with np.load(row['old_physical']['path'],allow_pickle=False) as z:
                assert all(arrays[k].dtype==z[k].dtype and arrays[k].tobytes()==z[k].tobytes() for k in z.files)
            with np.load(row['old_neural']['path'],allow_pickle=False) as z:
                for l in range(8,16):
                    assert arrays[f'features_{l:02d}'].tobytes()==z['features'][l].tobytes()
                    assert arrays[f'hidden_{l:02d}'].tobytes()==z['hidden'][l].tobytes()
            common_exact=True
            if call==0:first=digests
            if call==2:assert digests==first
        path=out/f'call_{call:02d}_rank{rank:04d}.npz';np.savez(path,**arrays)
        record=dict(call=call,rank=rank,capture=pin(path),digests=digests,flops=F,wall_s=wall,includes_initialization=call==0,
            all16_parent_bytes_and_bill_exact=True,numeric_cache_retired=True,old_physical_neural_common_fields_exact=common_exact,
            weights=row['weights'],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        records.append(record);print(json.dumps({k:v for k,v in record.items() if k!='digests'}),flush=True)
        tap.clear();neural.clear();del arrays,prediction,weights,ww,mlp;gc.collect()
    result=dict(completed=True,records=records,source_proof=proof,parent_sha256=PARENT_SHA,truth_accessed=False,
        repeat_all_capture_bytes_exact=bool(args.qualification),extra_parent_numerical_operations=0,full_estimator_calls=len(rows),
        capture_bytes=sum(x['capture']['bytes'] for x in records),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out/'result.json',result)
if __name__=='__main__':main()
