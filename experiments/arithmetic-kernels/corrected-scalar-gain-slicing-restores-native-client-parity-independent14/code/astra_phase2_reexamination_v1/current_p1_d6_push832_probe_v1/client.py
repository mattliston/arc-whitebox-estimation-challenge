"""One fixed push product in a fresh NumPy-free client; inert without --role."""
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
OUT = HERE/'run_v1'
SHAPE = (1024,832,8192)
KERNEL = NEW/'current_d50_serial_d6_kernel_v1/kernel.py'
KERNEL_SHA = 'a8d1cb1a972b37afd9eaaa67acf820cba3196dabea10ba1058bb3409c0b20ec7'
LAYOUT = NEW/'packed_arena_generic.py'
LAYOUT_SHA = '4e4a3dea533c222f34ba3a08538245414e149dbc0f918fd097267b0f869fb27e'
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def module(path,name,digest):
    require(sha(path)==digest,'Frozen helper changed')
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@contextmanager
def timed_budget(flops):
    # The pinned client.12 constructor has no wall-time keyword.
    started = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL,30)
    try:
        with flops.BudgetContext(flop_budget=2**39,quiet=True) as context:
            yield context
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    require(time.monotonic()-started < 30,'Fixed30s client context')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role',choices=('d4','breadth_d6','serial_d6'))
    parser.add_argument('--contract-sha256')
    args = parser.parse_args()
    if args.role is None:
        parser.print_help(); return 0
    contract_path = OUT/'contract.json'
    require(sha(contract_path)==args.contract_sha256,'Reviewed synthetic contract')
    contract = read(contract_path)
    for group in ('source_bindings','input_bindings'):
        require(all(sha(path)==digest for path,digest in contract[group].items()),'Unchanged contract inventory')
    require(contract['shape']==list(SHAPE) and contract['seed']==9309094,'Only fixed push832 fixture')
    require(importlib.util.find_spec('numpy') is None,'NumPy-free client')
    require(importlib.metadata.version('flopscope-client')=='0.12.0','Qualified client version')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10 and sys.flags.optimize==0
            and all(os.environ.get(key)=='1' for key in BLAS)
            and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0','nice10/BLAS1/no fallback or bytecode')
    require(os.getpgrp()==contract['measurement_process_group'],'Inherit supervisor-owned group')
    def expired(*unused):
        raise TimeoutError('Fixed30s client context watchdog')
    signal.signal(signal.SIGALRM,expired)
    kernel = module(KERNEL,'unchanged_serial_d6',KERNEL_SHA)
    layout = module(LAYOUT,'unchanged_arena_layout',LAYOUT_SHA).layout
    out = OUT/args.role
    out.mkdir(exist_ok=False)
    result = dict(schema='current_p1_d6_push832_client_v1',role=args.role,passed=False,completed=False,
        hashes_unchanged=False,source_bindings=contract['source_bindings'],
        input_bindings=contract['input_bindings'],contract_sha256=args.contract_sha256,
        shape=list(SHAPE),numpy_absent=True,truth_accessed=False,MLP_executed=False,
        trials=[],artifacts={})
    write(out/'started.json',result)
    try:
        import flopscope as flops
        import flopscope.numpy as fnp
        namespace = kernel.make_namespace(fnp)
        require(namespace['provenance']['helpers_sha256']==contract['source_audit']['helpers_sha256'],
                'Exact frozen helper bundle')
        result['helper_provenance'] = namespace['provenance']
        with timed_budget(flops) as setup:
            left = fnp.load(str(OUT/'left.npy'))
            right = fnp.load(str(OUT/'right.npy'))
            size4,push4,_ = layout(1024,1024,8192,4)
            read_size4,read4,_ = layout(1024,8192,1024,4)
            require(max(size4,read_size4)*4==1168900096,'Original D4 arena')
            depth = 4 if args.role=='d4' else 6
            owner_size = read_size4 if args.role!='breadth_d6' else layout(1024,8192,1024,6)[0]
            arena = fnp.empty((owner_size,),dtype=fnp.float32)
            old_push = namespace['_cp3_bind_arena_views'](arena,push4)
            old_read = namespace['_cp3_bind_arena_views'](arena,read4)
            if args.role=='serial_d6':
                serial_push = kernel.bind(fnp,arena,kernel.PUSH_CAPACITY,namespace)
                serial_read = kernel.bind(fnp,arena,kernel.READ_CAPACITY,namespace)
            elif args.role=='breadth_d6':
                deep_push = namespace['_cp3_bind_arena_views'](arena,layout(1024,1024,8192,6)[1])
                deep_read = namespace['_cp3_bind_arena_views'](arena,layout(1024,8192,1024,6)[1])
            output = fnp.empty((SHAPE[0],SHAPE[2]),dtype=fnp.float32)
        result.update(setup=setup.summary_dict(),backing_arena_bytes=owner_size*4)
        write(out/'setup_bill.json',result['setup'])
        result['artifacts'][str(out/'setup_bill.json')] = sha(out/'setup_bill.json')
        first = None
        for call in range(3):
            started = time.monotonic()
            with timed_budget(flops) as context:
                if args.role=='serial_d6':
                    kernel.product(left,right,output,serial_push)
                else:
                    namespace['_packed_product_cached'](left,right,output,
                        old_push if depth==4 else deep_push,depth)
            trial = dict(call=call,cold_frame=call==0,outer_wall_s=time.monotonic()-started,
                         summary=context.summary_dict())
            # Preserve the original client's separate verification context.
            with timed_budget(flops) as verification:
                raw,shape,dtype = output._fetch_data()
            trial['verification'] = verification.summary_dict()
            trial.update(output_shape=list(shape),output_dtype=str(dtype))
            result['trials'].append(trial)
            require(tuple(shape)==(SHAPE[0],SHAPE[2]) and str(dtype)=='float32','Exact pushed output shape/dtype')
            if first is None:
                first = raw
            require(raw==first,'Own cold/warm product bytes unchanged')
            path = out/f'call_{call:02d}.f32'
            with path.open('xb') as stream:
                stream.write(raw)
            trial.update(output=str(path),output_sha256=sha(path))
            result['artifacts'][str(path)] = trial['output_sha256']
            bill = out/f'call_{call:02d}_bill.json'
            write(bill,trial)
            result['artifacts'][str(bill)] = sha(bill)
            print(json.dumps(dict(role=args.role,call=call,flops=trial['summary']['flops_used'],
                                  wall_s=trial['summary']['wall_time_s'])),flush=True)
            del raw
        for group in ('source_bindings','input_bindings'):
            require(all(sha(path)==digest for path,digest in contract[group].items()),'Final unchanged contract')
        result.update(passed=True,completed=True,hashes_unchanged=True)
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,error=traceback.format_exc())
    write(out/'result.json',result)
    write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],
                                    result_sha256=sha(out/'result.json')))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
