"""Fixed development or validation acquisition with complete native bills.

OFFPY only. No truth files are opened. Candidate module and output/bill
seals identify the actual jointly evaluated configuration.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
P2=PARENT.parent
OFFPY=P2/'data_gate/.venv'
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
BUDGET=2**41
BANK=PARENT/'unused_materializations_v1/estimator.py'
BANK_SHA='409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f'
VALID=(37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
       139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
       204,219,224,227,233,234)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--split',choices=('development','validation'))
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.source is None: parser.print_help();return
    assert args.split and args.out and not args.out.exists()
    assert Path(sys.prefix).resolve()==OFFPY.resolve() and 'anaconda3' not in sys.executable
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    assert importlib.metadata.version('flopscope')=='0.12.0'
    assert importlib.metadata.version('whestbench')=='0.16.0'
    candidate=args.source.resolve(); assert candidate.is_relative_to(PARENT.resolve())
    inputs={str(p):sha(p) for p in (candidate,HERE/'PROTOCOL.md',
        HERE/'NATIVE_SCOPE_CLARIFICATION.md',Path(__file__),STORE/'manifest.json')}
    if candidate==BANK:
        assert sha(candidate)==BANK_SHA
    else:
        build=candidate.with_name('build.json')
        receipt=json.loads(build.read_text())
        assert receipt['passed'] and receipt['inverse_source_byte_exact']
        assert receipt['parent_sha256']==BANK_SHA and receipt['candidate_sha256']==sha(candidate)
        for item in receipt['source_bindings'].values():
            assert sha(item['path'])==item['sha256']
            inputs[item['path']]=item['sha256']
        wrapper=HERE/'fnp/build_v2.py'
        assert sha(wrapper)=='cdbb295a2e598506a15778dbfeec9f6033c4ad23aaa442881da743c7a8b06365'
        inputs[str(wrapper)]=sha(wrapper)
        inputs[str(build)]=sha(build)
    ranks=(35,100) if args.split=='development' else VALID
    manifest=json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    rows={}
    for rank in ranks:
        item=next(x for x in manifest if x['role_rank']==rank)
        rp=STORE/item['path'];assert sha(rp)==item['sha256']
        row=json.loads(rp.read_text()); wp=rp.parent/'weights.npy'
        assert sha(wp)==row['files']['weights.npy']['sha256'],rank
        inputs[str(wp)]=sha(wp);inputs[str(rp)]=sha(rp)
        rows[rank]=dict(weights=str(wp),mlp_seed=int(row['mlp_seed']))
    args.out.mkdir()
    result=dict(passed=False,source=str(candidate),candidate_sha256=sha(candidate),
        source_bindings=inputs,split=args.split,ranks=list(ranks),truth_accessed=False,
        started_utc=datetime.now(timezone.utc).isoformat(),trials=[],artifacts={},
        initialization_once=True,repeat_excluded=True,production_residual_qualified=False)
    write(args.out/'started.json',result)
    try:
        import numpy as np
        import flopscope as flops
        from threadpoolctl import threadpool_limits,threadpool_info
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        spec=importlib.util.spec_from_file_location('birth_direction_candidate',candidate)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module;spec.loader.exec_module(module)
        estimator=None
        first_bytes=None
        with threadpool_limits(limits=1):
            result['threadpools']=threadpool_info()
            assert all(p['num_threads']==1 for p in result['threadpools'] if p.get('user_api')=='blas')
            for call,rank in enumerate(tuple(ranks)+(ranks[0],)):
                weights=np.load(rows[rank]['weights'],mmap_mode='r')
                assert weights.shape==(16,1024,1024) and weights.dtype==np.float32 and np.isfinite(weights).all()
                mlp=MLP(width=1024,depth=16,weights=list(weights),
                        seed=int(derive_estimator_seed(rows[rank]['mlp_seed'])),name=f'birth-dir-{rank}')
                started=time.monotonic()
                with flops.BudgetContext(flop_budget=BUDGET,wall_time_limit_s=120,quiet=True) as ctx:
                    if estimator is None: estimator=module.Estimator()
                    served=estimator.predict(mlp,BUDGET)
                    assert isinstance(served,np.ndarray) and served.dtype==np.float32 and served.shape==(16,1024)
                    materialized=materialise_predictions(served,depth=16,width=1024)
                wall=time.monotonic()-started
                assert served.tobytes()==materialized.tobytes()
                values=np.frombuffer(materialized.tobytes(),dtype=np.float32).reshape(16,1024)
                assert np.isfinite(values).all() and (values>=0).all()
                output=args.out/f'call_{call:02d}_rank{rank}.npy'
                with output.open('xb') as f: np.save(f,values)
                output_hash=sha(output)
                result['artifacts'][str(output)]=output_hash
                trial=dict(call=call,rank=rank,flops=int(ctx.flops_used),
                    multiplier=max(.1,float(ctx.flops_used)/BUDGET),wall_s=wall,
                    context_wall_s=float(ctx.wall_time_s),
                    flopscope_backend_s=float(ctx.flopscope_backend_time_s),
                    flopscope_overhead_s=float(ctx.flopscope_overhead_time_s),
                    native_residual_s=float(ctx.residual_wall_time_s),
                    peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    output=str(output),output_sha256=output_hash,
                    output_bytes_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                    includes_initialization=(call==0))
                assert 0<trial['flops']<=BUDGET and wall<120 and trial['peak_rss_bytes']<8e9
                assert all(math.isfinite(trial[k]) and trial[k]>=0 for k in
                           ('wall_s','context_wall_s','flopscope_backend_s',
                            'flopscope_overhead_s','native_residual_s'))
                if call==0: first_bytes=values.tobytes()
                if call==len(ranks): assert values.tobytes()==first_bytes,'Repeat identity'
                result['trials'].append(trial)
                write(args.out/f'call_{call:02d}_bill.json',trial)
                print(json.dumps(trial),flush=True)
                del weights,mlp,served,materialized,values
                gc.collect()
        assert all(sha(p)==v for p,v in inputs.items())
        assert all(sha(p)==v for p,v in result['artifacts'].items())
        result.update(passed=True,hashes_unchanged=True,
                      initialization_flops=result['trials'][0]['flops']-result['trials'][-1]['flops'])
    except BaseException:
        result['error']=traceback.format_exc()
    result['finished_utc']=datetime.now(timezone.utc).isoformat()
    write(args.out/'result.json',result)
    write(args.out/'completion.json',dict(passed=result['passed'],result_sha256=sha(args.out/'result.json')))
    print(json.dumps(dict(passed=result['passed'],trials=len(result['trials']),error=result.get('error'))),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
