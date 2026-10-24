"""Inert, fixed metadata-retention/release diagnostic; no MLP or model inputs.

Four contexts each create 16 float32 ones and add zero in place 4096 times.
The fixed bill is 65552 F/context.  Save all closed bills, drop context owners,
then inspect the installed accumulator's record COUNTS only.  Its strong logs
must persist until the PUBLIC budget_reset call and be absent afterward.
A fifth fixed context held across reset demonstrates that all saved complete
bill/timing fields are unchanged.  No private backend state is modified.

OFFPY, nice >= 10, BLAS=1; 55 s alarm, elapsed < 60 s, process peak < 1 GB.
Default/help imports no numeric library.  --execute requires a new --out path.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import weakref

HERE=Path(__file__).resolve().parent
P2=HERE.parent.parent
OFFPY=P2/'data_gate/.venv'
SITE=OFFPY/'lib/python3.11/site-packages'
PINS={
    HERE/'NATIVE_INPUT_AND_LOG_LIFECYCLE_V2.md':'c08620d45fbcd93385f26c882fbc6c56e988673a19fe89484bc66dcdd2ba5dfa',
    SITE/'flopscope/_budget.py':'ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01',
    SITE/'flopscope/_array_ops.py':'e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c',
    SITE/'flopscope/_ndarray.py':'fa5bbecb48b123ab77f31ced3da877d2e333aa5b403a324c33da591b967d3e74',
}
STEPS=4096
WIDTH=16
EXPECTED_F=WIDTH*(STEPS+1)
EXPECTED_OPS=STEPS+1


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def bill(ctx):
    return dict(flops=int(ctx.flops_used),flop_budget=int(ctx.flop_budget),
                flops_remaining=int(ctx.flops_remaining),operation_count=len(ctx.op_log),
                context_wall_s=float(ctx.wall_time_s),
                backend_s=float(ctx.flopscope_backend_time_s),
                overhead_s=float(ctx.flopscope_overhead_time_s),
                residual_s=float(ctx.residual_wall_time_s))


def limits(start):
    assert time.monotonic()-start<60,'Elapsed limit'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1_000_000_000,'RSS limit'


def timeout(signum,frame):
    raise TimeoutError('Fixed 55-second diagnostic alarm')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if not args.execute: parser.print_help();return
    assert args.out is not None and not args.out.exists()
    assert args.out.resolve().is_relative_to(HERE.resolve())
    assert Path(sys.prefix).resolve()==OFFPY.resolve() and 'anaconda3' not in sys.executable
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    assert importlib.metadata.version('numpy')=='2.4.6'
    assert importlib.metadata.version('flopscope')=='0.12.0'
    assert all(sha(p)==digest for p,digest in PINS.items())
    bindings={str(p):digest for p,digest in PINS.items()}
    bindings[str(Path(__file__).resolve())]=sha(__file__)
    args.out.mkdir()
    result=dict(passed=False,completed=False,source_bindings=bindings,
                started_utc=datetime.now(timezone.utc).isoformat(),bills=[],artifacts={},
                model_accessed=False,weights_accessed=False,truth_accessed=False,
                private_backend_mutated=False,reset_api='flopscope.budget_reset',
                private_read_scope='lengths of accumulator records and their op_log lists only',
                fixed_contexts_before_reset=4,fixed_post_reset_contexts=1,
                width=WIDTH,steps=STEPS,expected_context_flops=EXPECTED_F)
    write(args.out/'started.json',result)
    start=time.monotonic()
    previous_handler=signal.signal(signal.SIGALRM,timeout)
    signal.alarm(55)
    try:
        import flopscope as flops
        import flopscope.numpy as fnp
        from flopscope import _budget as budget_source
        from threadpoolctl import threadpool_info,threadpool_limits
        flops.budget_reset()
        assert len(budget_source._accumulator._records)==0
        with threadpool_limits(limits=1):
            result['threadpools']=threadpool_info()
            assert all(p['num_threads']==1 for p in result['threadpools'] if p.get('user_api')=='blas')
            context_refs=[]
            for index in range(4):
                with flops.BudgetContext(flop_budget=1_000_000,wall_time_limit_s=10,quiet=True) as ctx:
                    value=fnp.ones(WIDTH,dtype=fnp.float32)
                    for step in range(STEPS):
                        fnp.add(value,0.0,out=value)
                        if step%256==0: limits(start)
                record=bill(ctx)
                assert record['flops']==EXPECTED_F and record['operation_count']==EXPECTED_OPS
                assert all(math.isfinite(record[k]) and record[k]>=0 for k in
                           ('context_wall_s','backend_s','overhead_s','residual_s'))
                path=args.out/f'closed_{index}.json'
                write(path,record);result['artifacts'][str(path)]=sha(path)
                result['bills'].append(record)
                context_refs.append(weakref.ref(ctx))
                del value,ctx
                gc.collect()
                limits(start)
            result['contexts_released_before_reset']=all(ref() is None for ref in context_refs)
            result['records_before_reset']=len(budget_source._accumulator._records)
            result['operations_before_reset']=sum(len(r.op_log) for r in budget_source._accumulator._records)
            assert result['contexts_released_before_reset']
            assert result['records_before_reset']==4
            assert result['operations_before_reset']==4*EXPECTED_OPS
            before=flops.budget_summary_dict()
            assert before['flops_used']==4*EXPECTED_F
            flops.budget_reset()
            gc.collect()
            result['records_after_reset']=len(budget_source._accumulator._records)
            result['operations_after_reset']=sum(len(r.op_log) for r in budget_source._accumulator._records)
            result['session_flops_after_reset']=flops.budget_summary_dict()['flops_used']
            assert result['records_after_reset']==result['operations_after_reset']==result['session_flops_after_reset']==0
            assert all(sha(p)==digest for p,digest in result['artifacts'].items())
            # Hold this closed context through reset: its complete bill must not change.
            with flops.BudgetContext(flop_budget=1_000_000,wall_time_limit_s=10,quiet=True) as ctx:
                value=fnp.ones(WIDTH,dtype=fnp.float32)
                for step in range(STEPS):
                    fnp.add(value,0.0,out=value)
                    if step%256==0: limits(start)
            saved=bill(ctx)
            assert saved['flops']==EXPECTED_F and saved['operation_count']==EXPECTED_OPS
            path=args.out/'closed_retained.json'
            write(path,saved);result['artifacts'][str(path)]=sha(path)
            flops.budget_reset()
            result['closed_context_bill_unchanged_by_reset']=saved==bill(ctx)
            assert result['closed_context_bill_unchanged_by_reset']
            assert len(budget_source._accumulator._records)==0
            context_ref=weakref.ref(ctx)
            del value,ctx
            gc.collect()
            result['final_context_released']=context_ref() is None
            assert result['final_context_released']
        limits(start)
        assert all(sha(p)==digest for p,digest in bindings.items())
        assert all(sha(p)==digest for p,digest in result['artifacts'].items())
        result.update(passed=True,sources_and_bills_unchanged=True)
    except BaseException:
        result['error']=traceback.format_exc()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous_handler)
    result.update(completed=True,elapsed_s=time.monotonic()-start,
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  finished_utc=datetime.now(timezone.utc).isoformat())
    if result['elapsed_s']>=60 or result['peak_rss_bytes']>=1_000_000_000:
        result['passed']=False
        result['resource_failure']=True
    write(args.out/'result.json',result)
    write(args.out/'completion.json',dict(passed=result['passed'],result_sha256=sha(args.out/'result.json')))
    print(json.dumps(dict(passed=result['passed'],elapsed_s=result['elapsed_s'],
                         peak_rss_bytes=result['peak_rss_bytes'],error=result.get('error'))),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
