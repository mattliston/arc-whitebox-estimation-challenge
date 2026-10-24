"""One fixed target-free native full-width polynomial block qualification."""
import argparse
import gc
import hashlib
import importlib.util
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

HERE = Path(__file__).resolve().parent
NADJ = HERE.parent/'current_native_trajectory_adjoint_v1'
OFFPY = HERE.parent.parent/'data_gate/.venv/bin/python'
NATIVE_SOURCE_SHA = '814d8bf5f4161ef388ef8c811421fc8c90ca8929ab440dbba4d4de3dffc99e9c'
SUPERVISOR_SOURCE_SHA = '648e5227795a8d2e5a91a1a421e0e54ffea1f76d6d6d65b28335178c8772025d'
NAMES = {'run_pair.py','supervise.py','port.py','unit.py','PROTOCOL.md'}
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(ok,message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def load(path,name,expected):
    require(sha(path)==expected,'Exact reviewed source before inert import')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true');ap.add_argument('--pins')
    args=ap.parse_args()
    if not args.run:ap.print_help();return 0
    began=time.monotonic();output=HERE/'pair_v1'
    require(not output.exists(),'One exclusive never-resumed block unit');output.mkdir()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Whole pair120s')))
    signal.alarm(120)
    result=dict(passed=False,completed=False,phase='pair',truth_accessed=False,fits=0,
        full_model_calls=0,parameter_updates=0,function_calls=0,source_bindings={},
        input_bindings={},artifacts={},deployment_runtime_measured=False)
    def tick():
        require(time.monotonic()-began<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8_000_000_000,
            'Whole120s/8GB; independent watcher owns combined tree/swap')
    try:
        pins=json.loads(Path(args.pins).read_text())
        require(set(pins)==NAMES and all(sha(HERE/k)==v for k,v in pins.items()),'Exact five reviewed unit owners')
        require(Path(sys.executable).resolve()==OFFPY.resolve() and os.getpriority(os.PRIO_PROCESS,0)>=10
            and sys.flags.optimize==0 and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
            and not os.environ.get('FLOPSCOPE_SERVER_URL') and all(os.environ.get(k)=='1' for k in BLAS),
            'Actual native interpreter/nice10/sixBLAS1/assertions/fallback0')
        native=load(NADJ/'native_cell.py','fusion_packet_admission',NATIVE_SOURCE_SHA)
        port=load(HERE/'port.py','fusion_exact_source',pins['port.py'])
        unit=load(HERE/'unit.py','fusion_reviewed_block',pins['unit.py'])
        context,apis=native.prepare(NATIVE_SOURCE_SHA,tick)
        j,c,a,h,backend,legacy,owner=apis
        original,fused,proof=port.source_bundle(tick)
        for key in ('source_bindings','input_bindings','capture_seals'):
            result[key]=dict(context[key])
        j.merge(result['source_bindings'],proof['source_bindings'])
        j.merge(result['source_bindings'],{str(HERE/k):v for k,v in pins.items()})
        j.merge(result['source_bindings'],{str(NADJ/'supervise.py'):SUPERVISOR_SOURCE_SHA})
        result['source_proof']=proof
        result['admission_source_bindings']=dict(result['source_bindings'])
        require(unit.FLOP_CAP==10_000_000_000,'Fixed full unit source/bill cap10BF')
        result['cost_certificate']=unit.cost_certificate()
        j.bind(result['source_bindings'],tick)
        ledger=dict(time=time,hashlib=hashlib,json=json,require=require)
        exec(compile(context['ledger_source'],'<original-closed-block-ledger>','exec'),ledger)
        metrics=ledger['_v2_ledger_metrics'](legacy.metrics)
        backend.backend_select();h.environment()
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits,threadpool_info
        flops.configure(symmetry_warnings=False)
        with threadpool_limits(limits=1):
            result['threadpools']=backend.backend_threadpool_info(threadpool_info,result,tick)()
            refs=[];ctx=None;answer=None;error=None;started=time.monotonic()
            try:
                with flops.BudgetContext(flop_budget=unit.FLOP_CAP,wall_time_limit_s=120,quiet=True) as ctx:
                    answer=unit.evaluate(context,port,np,fnp,ctx,refs,tick)
            except BaseException:
                error=traceback.format_exc()
            require(ctx is not None,'Context construction before numerical work')
            bill=dict(**metrics(ctx,started),complete_research_flops=int(ctx.flops_used),
                includes_all_native_input_and_diagnostics=True,includes_initialization=True,
                call_completed=error is None,call_error=error,not_a_complete_estimator_score=True)
            write(output/'bill.json',bill)
            cr=weakref.ref(ctx);life=legacy.reset_saved_history(flops,ctx,output/'bill.json')
            del ctx;gc.collect()
            retired=cr() is None and all(r() is None for r in refs)
            life.update(context_released=cr() is None,all_numerical_array_owners_retired=retired)
            write(output/'lifecycle.json',life)
            require(retired and life['closed_context_bill_unchanged_by_reset'],'Every native block owner/context retired')
            require(error is None,'Block call failed; attempted own bill preserved:\n'+str(error))
            require(0<bill['flops']==bill['complete_research_flops']<=unit.FLOP_CAP
                and bill['wall_s']<120 and bill['context_wall_s']<120 and bill['peak_rss_bytes']<8_000_000_000
                and all(math.isfinite(bill[k]) and bill[k]>=0 for k in legacy.TIMINGS),'Complete original resource/bill checks')
        # Numerical arrays are gone before detached diagnostics are published.
        require(answer['function_calls']==4 and answer['truth_accessed'] is False,
            'Exactly original/fused pair and terminal blocks')
        write(output/'diagnostics.json',answer)
        backend.backend_origins(result,tick)
        result['artifacts']={str(p):sha(p) for p in output.iterdir() if p.is_file()}
        require(all(sha(k)==v for group in ('source_bindings','input_bindings') for k,v in result[group].items()),
            'All actual source/input owners unchanged')
        result.update(passed=answer['passed'],completed=True,all_owners_retired=True,function_calls=4,
            complete_research_flops=bill['flops'],block_integrity_qualified=answer['passed'],
            full_card_qualified=False,release_ready=False,score_measured=False)
    except BaseException:
        result['error']=traceback.format_exc()
    result.update(elapsed_s=time.monotonic()-began,
        maximum_self_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['elapsed_s']>=120 or result['maximum_self_rss_bytes']>=8_000_000_000:
        result.update(passed=False,completed=False,resource_error=True)
    write(output/'result.json',result);rh=sha(output/'result.json')
    write(output/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=rh))
    write(output/'finalization.json',dict(passed=result['passed'],completed=result['completed'],
        result_sha256=rh,completion_sha256=sha(output/'completion.json'),elapsed_s=time.monotonic()-began))
    signal.alarm(0)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','function_calls','complete_research_flops',
        'elapsed_s','error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
