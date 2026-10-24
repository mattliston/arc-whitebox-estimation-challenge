"""Fixed NumPy-free product client. Inert without --role."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
SOURCE = NEW/'birth_direction_fit_v1/build_terminal_v1/estimator.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


@contextmanager
def timed_budget(flops):
    """Client.12 has no wall-limit constructor argument: use a watchdog."""
    started=time.monotonic()
    signal.setitimer(signal.ITIMER_REAL,30)
    try:
        with flops.BudgetContext(flop_budget=2**39,quiet=True) as ctx:
            yield ctx
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    assert time.monotonic()-started<30,'Fixed30s client context guard'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role',choices=('d4','breadth_d6','serial_d6'))
    parser.add_argument('--contract-sha256')
    args=parser.parse_args()
    if args.role is None:
        parser.print_help();return 0
    contract_path=HERE/'run_v1/contract.json'
    assert sha(contract_path)==args.contract_sha256
    contract=json.loads(contract_path.read_text())
    assert all(sha(p)==v for p,v in contract['source_bindings'].items())
    assert all(sha(p)==v for p,v in contract['input_bindings'].items())
    assert importlib.util.find_spec('numpy') is None
    assert importlib.metadata.version('flopscope-client')=='0.12.0'
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
    def expired(*_):
        raise TimeoutError('Fixed30s client context watchdog')
    signal.signal(signal.SIGALRM,expired)
    import flopscope as flops
    import flopscope.numpy as fnp
    import kernel
    spec=importlib.util.spec_from_file_location('d6_product_layout',NEW/'packed_arena_generic.py')
    layout_module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout_module)
    layout=layout_module.layout
    outdir=HERE/'run_v1'/args.role
    outdir.mkdir()
    result=dict(passed=False,completed=False,role=args.role,contract_sha256=args.contract_sha256,
        numpy_absent=True,truth_accessed=False,score_measured=False,trials=[],artifacts={})
    write(outdir/'started.json',result)
    try:
        # Driver source audit already authenticated the embedded exact helpers;
        # compile only that compact bundle in the NumPy-free client.
        namespace=kernel.make_namespace(fnp)
        assert namespace['provenance']['helpers_sha256']==contract['source_audit']['helpers_sha256']
        result['helper_provenance']=namespace['provenance']
        with timed_budget(flops) as setup:
            left=fnp.load(str(HERE/'run_v1/left.npy'))
            right=fnp.load(str(HERE/'run_v1/right.npy'))
            size4,push4,_=layout(1024,1024,8192,4)
            read_size4,read4,_=layout(1024,8192,1024,4)
            assert max(size4,read_size4)*4==1168900096
            depth=4 if args.role=='d4' else 6
            owner_size=read_size4 if args.role!='breadth_d6' else layout(1024,8192,1024,6)[0]
            arena=fnp.empty((owner_size,),dtype=fnp.float32)
            # Both existing bindings are paid, including for serial reuse.
            old_push=namespace['_cp3_bind_arena_views'](arena,push4)
            old_read=namespace['_cp3_bind_arena_views'](arena,read4)
            if args.role=='serial_d6':
                serial_push=kernel.bind(fnp,arena,(1024,1024,8192),namespace)
                serial_read=kernel.bind(fnp,arena,(1024,8192,1024),namespace)
            elif args.role=='breadth_d6':
                deep_push=namespace['_cp3_bind_arena_views'](arena,layout(1024,1024,8192,6)[1])
                deep_read=namespace['_cp3_bind_arena_views'](arena,layout(1024,8192,1024,6)[1])
            output=fnp.empty((1024,1024),dtype=fnp.float32)
        result['setup']=setup.summary_dict()
        result['backing_arena_bytes']=owner_size*4
        reference_bytes=None
        for call in range(3):
            started=time.monotonic()
            with timed_budget(flops) as ctx:
                if args.role=='serial_d6':
                    kernel.product(left,right,output,serial_read)
                else:
                    namespace['_packed_product_cached'](left,right,output,old_read if depth==4 else deep_read,depth)
            trial=dict(call=call,cold_frame=call==0,outer_wall_s=time.monotonic()-started,summary=ctx.summary_dict())
            # Separate verification materialization from the core-kernel timing.
            with timed_budget(flops) as verification:
                raw,shape,dtype=output._fetch_data()
            trial['verification']=verification.summary_dict()
            assert tuple(shape)==(1024,1024) and str(dtype)=='float32'
            if reference_bytes is None:
                reference_bytes=raw
            assert raw==reference_bytes,'Repeated product bytes changed'
            path=outdir/f'call_{call:02d}.f32'
            with path.open('xb') as stream:
                stream.write(raw)
            result['artifacts'][str(path)]=sha(path)
            trial.update(output=str(path),output_sha256=sha(path))
            result['trials'].append(trial)
            write(outdir/f'call_{call:02d}_bill.json',trial)
            print(json.dumps(dict(role=args.role,call=call,flops=trial['summary']['flops_used'],wall_s=trial['outer_wall_s'])),flush=True)
        assert all(sha(p)==v for p,v in contract['source_bindings'].items())
        result.update(passed=True,completed=True)
    except BaseException:
        result['error']=traceback.format_exc()
    write(outdir/'result.json',result)
    write(outdir/'completion.json',dict(passed=result['passed'],result_sha256=sha(outdir/'result.json')))
    print(json.dumps(dict(role=args.role,passed=result['passed'],error=result.get('error'))),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
