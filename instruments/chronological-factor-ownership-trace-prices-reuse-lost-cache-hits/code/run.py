"""Root-only unchanged native35->100 chronology. No targets or score computation."""
from pathlib import Path
import hashlib,json,os,sys,time,types
import numpy as np
import flopscope as flops
from flopscope import _budget
from observer import NativeTrace

HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
SOURCE=H2/'hill_climb_star_projection32_v1/candidate.py'
CONTROL=H2/'hill_climb_star_projection32_train2_v1'
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'


def pin(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=p.stat().st_size)


def once(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def main():
    assert not (HERE/'complete.json').exists()
    refs=json.loads((HERE/'data_refs.json').read_text())
    seal=json.loads((HERE/'seal.json').read_text())
    assert seal['passed'] and seal['source']==refs['source']
    for row in seal['sources']+refs['pins']:assert pin(Path(row['path']))==row
    native=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1'
    assert Path(sys.executable).parent==native/'bin' and np.__version__=='2.4.6'
    assert str(Path(flops.__file__).resolve()).startswith(str(native.resolve())+'/')
    thread={n:os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}
    assert all(v=='1' for v in thread.values())
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    flops.configure(symmetry_warnings=False)
    mod=types.ModuleType('s32_hk_chronology');mod.__file__=str(SOURCE);sys.modules[mod.__name__]=mod
    exec(compile(SOURCE.read_text(),str(SOURCE),'exec',dont_inherit=True),mod.__dict__)
    class MLP:
        def __init__(self,w,r):self.width,self.depth,self.weights,self.seed,self.name=1024,16,w,0,f'phase2-{r}'
    est=mod.Estimator();start=time.monotonic()
    est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
    setup=dict(seconds=time.monotonic()-start,pb_log=[list(x) for x in mod._PB_LOG])
    assert [x[1] for x in setup['pb_log'] if x[0]=='bound']==[624]
    assert [x[1] for x in setup['pb_log'] if x[0]=='lvfuse']==[600]
    once(HERE/'setup.json',dict(source=refs['source'],setup=setup,threads=thread,runtime=sys.executable))
    receipts=[]
    for row in refs['rows']:
        rank=row['rank'];weights=np.load(row['weights']['path'],allow_pickle=False)
        assert weights.shape==(16,1024,1024) and weights.dtype==np.dtype('float32')
        observer=NativeTrace(mod,_budget,HERE/f'trace_{rank}.jsonl');started=time.monotonic()
        with observer:
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                observer.ctx=ctx
                prediction=np.asarray(est.predict(MLP([weights[l] for l in range(16)],rank),2**41),dtype=np.float64)
        bill=int(ctx.summary_dict()['flops_used'])
        reference=np.load(row['prediction']['path'],allow_pickle=False)
        assert prediction.shape==reference.shape==(16,1024)
        assert prediction.dtype==reference.dtype==np.dtype('float64')
        assert prediction.tobytes()==reference.tobytes() and np.isfinite(prediction).all()
        assert bill==row['flops']
        summary=observer.summary();assert summary['patches_restored']
        trace=pin(HERE/f'trace_{rank}.jsonl');assert trace['bytes']==summary['bytes']
        receipt=dict(passed=True,rank=rank,flops=bill,all16_bytes_exact=True,full_bill_exact=True,
            source=refs['source'],control=row['prediction'],trace=trace,observer=summary,
            elapsed_s=time.monotonic()-started,targets_accessed=False,
            prediction_bytes_sha256=hashlib.sha256(prediction.tobytes()).hexdigest())
        once(HERE/f'row_{rank}.json',receipt);receipts.append(pin(HERE/f'row_{rank}.json'))
        print(json.dumps(receipt),flush=True)
        del weights,prediction,reference,observer
    for row in seal['sources']+refs['pins']:assert pin(Path(row['path']))==row
    once(HERE/'complete.json',dict(passed=True,ranks=[35,100],source=refs['source'],rows=receipts,
        seal=pin(HERE/'seal.json'),data_refs=pin(HERE/'data_refs.json'),targets_accessed=False,
        current_source_unchanged=True,setup=pin(HERE/'setup.json')))
    print('UNCHANGED_NATIVE2_COMPLETE',flush=True)


if __name__=='__main__':main()
