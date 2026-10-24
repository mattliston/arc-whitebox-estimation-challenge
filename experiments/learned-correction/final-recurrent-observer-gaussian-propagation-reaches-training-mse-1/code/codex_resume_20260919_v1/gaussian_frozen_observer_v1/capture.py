"""Native Gaussian host qualification and small, target-free feature capture."""
import argparse,importlib.util,sys,types,time,resource,gc
from common import HERE,D,GAUSSIAN,TRAIN,CHECK,wp,pin,sha,write,read,verify
def main():
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(t['num_threads']==1 for t in threadpool_info())
    parser=argparse.ArgumentParser();parser.add_argument('--qualification',action='store_true');parser.add_argument('--all',action='store_true');args=parser.parse_args()
    assert args.qualification != args.all
    verify('capture_sources_sealed.json')
    if args.all:
        assert read(HERE/'qualification/result.json')['passed']
        g=read(HERE/'qualification_watch.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    out=HERE/('qualification' if args.qualification else 'capture_all');out.mkdir(exist_ok=False)
    sys.path.insert(0,str(GAUSSIAN))
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    spec=importlib.util.spec_from_file_location('gfo_gaussian_host',GAUSSIAN/'estimator.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    spec.loader.exec_module(mod);original=mod.forecast
    exec(compile((HERE/'observed_forecast.py').read_text(),str(HERE/'observed_forecast.py'),'exec'),mod.__dict__);observed=mod.forecast
    fs=importlib.util.spec_from_file_location('gfo_features',HERE/'features.py');fm=importlib.util.module_from_spec(fs);fs.loader.exec_module(fm)
    held={}
    def callback(l,w,scalar,pre,post):
        assert l not in held
        x=fm.gaussian_features(l,w,scalar,pre,post)
        held[l]=(x,scalar['s'],scalar['m'])
    def call(weights,observe):
        held.clear()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(w) for w in weights]
        start=time.monotonic();mod.forecast=observed if observe else original;mod._OBSERVER=callback if observe else None
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:output,diag=mod._predict(ww,False)
        result=np.asarray(output).copy();F=ctx.summary_dict()['flops_used'];elapsed=time.monotonic()-start
        arr=None
        if observe:
            assert set(held)==set(range(len(weights)))
            arr=dict(features=np.stack([np.asarray(held[i][0]) for i in range(len(weights))]),
                     sigma=np.stack([np.asarray(held[i][1]) for i in range(len(weights))]),
                     final_host=np.asarray(held[len(weights)-1][2]).copy(),prediction=result)
            assert all(np.isfinite(v).all() for v in arr.values())
            assert np.array_equal(arr['final_host'].astype(np.float32),result[-1])
        held.clear();del ww,output;gc.collect()
        return result,diag,F,elapsed,arr,ingress.summary_dict()['flops_used']
    tiny=None
    if args.qualification:
        rng=np.random.default_rng(2026092053);tinyW=(rng.normal(size=(4,7,7))/np.sqrt(7)).astype(np.float32)
        bare=call(tinyW,False);cap=call(tinyW,True)
        assert np.array_equal(bare[0],cap[0]) and bare[1]==cap[1]
        tiny=dict(passed=True,control_flops=bare[2],observed_flops=cap[2],all_physical_outputs_and_ledger_exact=True)
        del tinyW,bare,cap
    ranks=[35,100,35] if args.qualification else TRAIN[1:]+CHECK
    records=[];controls=[];first=None
    for index,rank in enumerate(ranks):
        path=wp(rank);weights=np.load(path,allow_pickle=False);assert weights.shape==(16,1024,1024) and weights.dtype==np.float32
        bare=None
        if args.qualification and index<2:
            bare=call(weights,False);bp=out/f'control_rank{rank:04d}.npy';np.save(bp,bare[0])
            controls.append(dict(rank=rank,output=pin(bp),flops=bare[2],wall_s=bare[3],diagnostics=bare[1]))
        output,diag,F,elapsed,arrays,inputF=call(weights,True)
        assert F<=100_000_000_000 and F+8_000_000_000<2**41*.1
        assert output.dtype==np.float32 and output.shape==(16,1024) and (output>=0).all()
        assert arrays['features'].shape==(16,1024,30) and arrays['features'].dtype==np.float32
        assert arrays['sigma'].shape==(16,1024) and arrays['sigma'].dtype==np.float64
        if bare is not None:assert np.array_equal(output,bare[0]) and diag==bare[1]
        digests={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=__import__('hashlib').sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
        if args.qualification and index==0:first=digests
        if args.qualification and index==2:assert digests==first
        dest=out/f'call_{index:02d}_rank{rank:04d}.npz';np.savez(dest,**arrays)
        rec=dict(rank=rank,index=index,features=pin(dest),weights=pin(path),host_feature_flops=F,neural_allowance_flops=8_000_000_000,
                 planning_multiplier=.1,wall_s=elapsed,input_flops=inputF,array_digests=digests,diagnostics=diag,
                 same_runtime_unmodified_control_exact=bool(args.qualification),truth_accessed=False)
        records.append(rec);print(__import__('json').dumps({k:v for k,v in rec.items() if k not in ('array_digests','diagnostics')}),flush=True)
        del weights,output,arrays,bare;gc.collect()
    result=dict(passed=True,records=records,controls=controls,tiny=tiny,truth_accessed=False,repeat_exact=bool(args.qualification),
                host_feature_cap_passed=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                full_native_calls=len(ranks)+len(controls),tiny_native_calls=2 if args.qualification else 0)
    write(out/'result.json',result)
if __name__=='__main__':main()
