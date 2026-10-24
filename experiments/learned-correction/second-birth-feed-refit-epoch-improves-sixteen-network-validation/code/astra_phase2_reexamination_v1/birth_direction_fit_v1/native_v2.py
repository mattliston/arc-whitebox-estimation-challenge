"""Fixed native acquisition with official weight types and bounded log lifetime.

Unexecuted harness successor to preserved native.py.  The estimator, sequence,
prediction budget, initialization ownership, and 120 s / 8 GB gates are unchanged.
Official float32 FNP weight copies are prepared in a separate input context, not
charged to the participant.  Only after a closed context's bill has been saved
does the harness reset accumulated diagnostics and release that context.  No
reset, counter modification, or candidate-state modification occurs in predict.
No truth files are opened.  A failed call's metrics are saved before its gates.
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
import weakref

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
P2=PARENT.parent
OFFPY=P2/'data_gate/.venv'
SITE=OFFPY/'lib/python3.11/site-packages'
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
BUDGET=2**41
BANK=PARENT/'unused_materializations_v1/estimator.py'
BANK_SHA='409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f'
PINS={
    HERE/'native.py':'7718591dd4324956fdcad52721009e7a9f7fb01c6969710b622376633ffbe075',
    HERE/'NATIVE_INPUT_AND_LOG_LIFECYCLE_V2.md':'c08620d45fbcd93385f26c882fbc6c56e988673a19fe89484bc66dcdd2ba5dfa',
    SITE/'flopscope/_budget.py':'ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01',
    SITE/'flopscope/_array_ops.py':'e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c',
    SITE/'flopscope/_ndarray.py':'fa5bbecb48b123ab77f31ced3da877d2e333aa5b403a324c33da591b967d3e74',
    SITE/'whestbench/domain.py':'db736fec3f983258e363750384d943db1b662a027c998dc494f7646586e3ade5',
}
VALID=(37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
       139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
       204,219,224,227,233,234)
TIMINGS=('wall_s','context_wall_s','flopscope_backend_s',
         'flopscope_overhead_s','native_residual_s')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def json_safe(value):
    # Preserve a diagnostic even if a timing gate fails on NaN or infinity.
    # Such a receipt is failed, never eligible for scoring.
    if isinstance(value,float) and not math.isfinite(value):
        return {'nonfinite_float':repr(value)}
    if isinstance(value,dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    return value


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(json_safe(value),f,indent=2,allow_nan=False);f.write('\n')


def metrics(ctx,started):
    return dict(flops=int(ctx.flops_used),wall_s=time.monotonic()-started,
                context_wall_s=float(ctx.wall_time_s),
                flopscope_backend_s=float(ctx.flopscope_backend_time_s),
                flopscope_overhead_s=float(ctx.flopscope_overhead_time_s),
                native_residual_s=float(ctx.residual_wall_time_s),
                peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                operation_count=len(ctx.op_log))


def guard(record):
    assert 0<record['flops']<=BUDGET and record['wall_s']<120 and record['peak_rss_bytes']<8e9
    assert all(math.isfinite(record[k]) and record[k]>=0 for k in TIMINGS)


def closed_metrics(ctx):
    return dict(flops=int(ctx.flops_used),operation_count=len(ctx.op_log),
                context_wall_s=float(ctx.wall_time_s),
                backend_s=float(ctx.flopscope_backend_time_s),
                overhead_s=float(ctx.flopscope_overhead_time_s),
                residual_s=float(ctx.residual_wall_time_s))


def reset_saved_history(flops,ctx,saved_bill):
    """Harness only; every caller is lexically after BudgetContext.__exit__."""
    before=closed_metrics(ctx)
    record=dict(saved_bill=str(saved_bill),saved_bill_sha256=sha(saved_bill),
                reset_api='flopscope.budget_reset',after_context_exit=True,
                candidate_state_reset=False,closed_metrics=before)
    flops.budget_reset()
    record['closed_context_bill_unchanged_by_reset']=before==closed_metrics(ctx)
    return record


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
    for path,digest in PINS.items():
        assert sha(path)==digest,path
        inputs[str(path)]=digest
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
        initialization_once=True,repeat_excluded=True,production_residual_qualified=False,
        input_preparation=[],input_representation='owned FNP float32 copies, official loader convention',
        input_preparation_in_participant_bill=False,
        reset_scope='harness only, after each closed context bill is saved',
        candidate_state_reset=False,peak_rss_scope='process lifetime, including input preparation')
    write(args.out/'started.json',result)
    try:
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
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
                input_started=time.monotonic()
                with flops.BudgetContext(flop_budget=BUDGET,wall_time_limit_s=120,quiet=True) as input_ctx:
                    weight_layers=[fnp.array(w,dtype=fnp.float32) for w in weights]
                input_record=dict(call=call,rank=rank,**metrics(input_ctx,input_started),
                                  charged_to_participant=False,exact_bytes=False,
                                  official_fnp_type=False,independent_weight_buffers=False,
                                  layers=16,context_closed=True,
                                  weight_file_sha256=inputs[rows[rank]['weights']])
                input_record['official_fnp_type']=all(
                    isinstance(w,fnp.ndarray) and w.dtype==np.float32 and w.shape==(1024,1024)
                    for w in weight_layers)
                input_record['exact_bytes']=all(
                    w.tobytes()==weights[l].tobytes() for l,w in enumerate(weight_layers))
                input_record['independent_weight_buffers']=all(
                    not np.may_share_memory(np.asarray(w),weights) for w in weight_layers)
                input_record['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                result['input_preparation'].append(input_record)
                input_bill=args.out/f'call_{call:02d}_input_bill.json'
                write(input_bill,input_record);result['artifacts'][str(input_bill)]=sha(input_bill)
                # Public session-history reset, strictly after context exit and receipt.
                input_ref=weakref.ref(input_ctx)
                input_reset=reset_saved_history(flops,input_ctx,input_bill)
                del input_ctx
                gc.collect()
                input_reset.update(call=call,rank=rank,context_released=input_ref() is None,
                                   complete=True,finished_utc=datetime.now(timezone.utc).isoformat())
                input_reset_path=args.out/f'call_{call:02d}_input_reset.json'
                write(input_reset_path,input_reset)
                result['artifacts'][str(input_reset_path)]=sha(input_reset_path)
                guard(input_record)
                assert input_record['official_fnp_type'] and input_record['exact_bytes'] and input_record['independent_weight_buffers']
                assert input_record['flops']==16*1024*1024
                assert input_reset['closed_context_bill_unchanged_by_reset'] and input_reset['context_released']
                mlp=MLP(width=1024,depth=16,weights=weight_layers,
                        seed=int(derive_estimator_seed(rows[rank]['mlp_seed'])),name=f'birth-dir-{rank}')
                del weights,weight_layers
                started=time.monotonic()
                with flops.BudgetContext(flop_budget=BUDGET,wall_time_limit_s=120,quiet=True) as ctx:
                    if estimator is None: estimator=module.Estimator()
                    served=estimator.predict(mlp,BUDGET)
                    assert isinstance(served,np.ndarray) and served.dtype==np.float32 and served.shape==(16,1024)
                    materialized=materialise_predictions(served,depth=16,width=1024)
                trial=dict(call=call,rank=rank,**metrics(ctx,started),
                           multiplier=max(.1,float(ctx.flops_used)/BUDGET),
                           includes_initialization=(call==0),
                           input_bill=str(input_bill),input_bill_sha256=sha(input_bill),
                           input_reset_receipt=str(input_reset_path),input_reset_sha256=sha(input_reset_path),
                           lifecycle_receipt=str(args.out/f'call_{call:02d}_lifecycle.json'))
                assert served.tobytes()==materialized.tobytes()
                values=np.frombuffer(materialized.tobytes(),dtype=np.float32).reshape(16,1024)
                assert np.isfinite(values).all() and (values>=0).all()
                output=args.out/f'call_{call:02d}_rank{rank}.npy'
                with output.open('xb') as f: np.save(f,values)
                output_hash=sha(output)
                result['artifacts'][str(output)]=output_hash
                trial.update(output=str(output),output_sha256=output_hash,
                             output_bytes_sha256=hashlib.sha256(values.tobytes()).hexdigest())
                trial['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                # Save this attempted call before any resource/timing/repeat gate.
                result['trials'].append(trial)
                bill=args.out/f'call_{call:02d}_bill.json'
                write(bill,trial);result['artifacts'][str(bill)]=sha(bill)
                print(json.dumps(json_safe(trial)),flush=True)
                context_ref=weakref.ref(ctx)
                lifecycle=reset_saved_history(flops,ctx,bill)
                del ctx
                gc.collect()
                lifecycle.update(call=call,rank=rank,context_released=context_ref() is None,
                                 complete=True,finished_utc=datetime.now(timezone.utc).isoformat())
                lifecycle_path=Path(trial['lifecycle_receipt'])
                write(lifecycle_path,lifecycle)
                result['artifacts'][str(lifecycle_path)]=sha(lifecycle_path)
                guard(trial)
                assert lifecycle['closed_context_bill_unchanged_by_reset'] and lifecycle['context_released']
                if call==0: first_bytes=values.tobytes()
                if call==len(ranks): assert values.tobytes()==first_bytes,'Repeat identity'
                del mlp,served,materialized,values
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
