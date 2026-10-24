"""Fixed sequential child: exact native ENTRY8, or CPU old-only comparison."""
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
P2 = HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
AP = Path('/Users/matt/anaconda3/bin/python')
NAMES = {'entry8_worker_v2.py','watch_entry8_v2.py','entry8_diagnostic_v2.py',
         'ENTRY8_REPAIR_V2_20260913_UTC.md'}
LIMITS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('native','torch'),required=True)
    ap.add_argument('--pins',required=True)
    ap.add_argument('--native-seals')
    args=ap.parse_args()
    began=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Whole120s')))
    signal.alarm(120)
    output=HERE/('entry8_'+args.phase+'_v2')
    require(not output.exists(),'Fresh exclusive phase, no overwrite')
    output.mkdir()
    result=dict(passed=False,completed=False,phase=args.phase,truth_accessed=False,
                score_measured=False,full_model_calls=0,parameter_updates=0)
    def tick():
        require(time.monotonic()-began<120 and
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8_000_000_000,
                'Whole worker120s/8GB; independent watch owns tree and swap')
    try:
        pins=read(args.pins)
        require(set(pins)==NAMES and all(sha(HERE/k)==v for k,v in pins.items()),
                'Reviewed exact four-owner roster')
        require(Path(sys.executable).resolve()==(OFFPY if args.phase=='native' else AP).resolve(),
                'Fixed phase interpreter')
        require(os.getpriority(os.PRIO_PROCESS,0)>=10 and
                all(os.environ.get(k)=='1' for k in LIMITS),'Actual nice10/six BLAS limits1')
        spec=importlib.util.spec_from_file_location('entry8_reviewed_helper',HERE/'entry8_diagnostic_v2.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        context=helper.prepare(pins['entry8_diagnostic_v2.py'],tick)
        for key in ('source_bindings','input_bindings','capture_seals','source_proof','cost_certificate'):
            result[key]=context[key]
        result['admission_source_bindings']=dict(context['source_bindings'])
        result['source_bindings']=dict(context['source_bindings'])
        result['own_source_bindings']=pins
        refs=[]
        if args.phase=='native':
            require(args.native_seals is None,'Native stage cannot borrow later seals')
            j=helper.load(helper.TRACE/'probe.py','entry8_worker_native_owner',
                          helper.PINS[helper.TRACE/'probe.py'])
            c,a,h,backend,legacy=j.helpers()
            ledger=dict(time=time,hashlib=hashlib,json=json,require=require)
            exec(compile(context['source']['ledger'],'<closed-native-ledger>','exec'),ledger)
            metrics=ledger['_v2_ledger_metrics'](legacy.metrics)
            backend.backend_select();h.environment()
            import numpy as np
            import flopscope as flops
            import flopscope.numpy as fnp
            from threadpoolctl import threadpool_limits,threadpool_info
            flops.configure(symmetry_warnings=False)
            with threadpool_limits(limits=1,user_api='blas'):
                result['threadpools']=backend.backend_threadpool_info(threadpool_info,result,tick)()
                started=time.monotonic()
                with flops.BudgetContext(flop_budget=context['cost_certificate']['native_upper'],
                        wall_time_limit_s=120,quiet=True) as ctx:
                    bundle=helper.native_phase(context,np,flops,fnp,ctx,refs)
                bill=dict(**metrics(ctx,started),complete_research_flops=int(ctx.flops_used),
                          includes_initialization=True,includes_all_input_and_snapshot_work=True,
                          not_a_complete_estimator_score=True)
                require(0<bill['flops']==bill['complete_research_flops']<=
                        context['cost_certificate']['native_upper'] and
                        bill['wall_s']<120 and bill['context_wall_s']<120 and
                        bill['peak_rss_bytes']<8_000_000_000 and
                        all(math.isfinite(bill[k]) and bill[k]>=0 for k in legacy.TIMINGS),
                        'Actual complete native bill inside pre-run bound')
                write(output/'bill.json',bill)
                cr=weakref.ref(ctx)
                life=legacy.reset_saved_history(flops,ctx,output/'bill.json')
                del ctx;gc.collect()
                require(cr() is None and all(r() is None for r in refs) and
                        life['closed_context_bill_unchanged_by_reset'],'Native context/all arrays retired')
                life.update(context_released=True,all_numerical_array_owners_retired=True)
            helper.validate_bundle(bundle)
            # All native owners are gone before detached artifacts are published.
            with (output/'bundle.bin').open('xb') as stream:stream.write(bundle['blob'])
            write(output/'bundle.json',bundle['metadata'])
            write(output/'lifecycle.json',life)
            result.update(native_control=bundle['metadata']['native_control'],
                          complete_research_flops=bill['complete_research_flops'],
                          own_complete_bill=True,all_owners_retired=True)
            del bundle
            backend.backend_origins(result,tick)
            artifacts=('bill.json','bundle.bin','bundle.json','lifecycle.json')
        else:
            require(args.native_seals is not None,'Actual retired native seals required')
            native=HERE/'entry8_native_v2'; seals=json.loads(args.native_seals)
            names=('result.json','completion.json','finalization.json')
            require(set(seals)==set(names) and all(sha(native/k)==v for k,v in seals.items()),
                    'Exact native files pinned by independent sequential parent')
            nr,nc,nf=(read(native/k) for k in names)
            require(all(v['passed'] and v['completed'] for v in (nr,nc,nf)) and
                    nc['result_sha256']==nf['result_sha256']==seals['result.json'] and
                    nf['completion_sha256']==seals['completion.json'] and
                    nr['phase']=='native' and nr['all_owners_retired'] and nr['own_complete_bill'],
                    'Closed owned native three-seal chain')
            require(all(sha(k)==v for k,v in nr['artifacts'].items()) and
                    nr['admission_source_bindings']==context['source_bindings'] and
                    all(nr[k]==context[k] for k in ('input_bindings',
                        'capture_seals','source_proof','cost_certificate')),
                    'Full actual native artifact and unchanged input/source chain')
            origins={v['path']:v['sha256'] for v in nr['backend_attestation']['modules'].values()}
            initial=nr['admission_source_bindings'];final=nr['source_bindings']
            require(nr['backend_attestation']['actual_origins_qualified'] and
                    nr['backend_attestation']['production_blas_qualified'] and
                    all(final.get(k)==v for k,v in initial.items()) and
                    all(sha(k)==v and (k in initial or origins.get(k)==v)
                        for k,v in final.items()),'Only actually attested native import-owner additions')
            bundle=dict(metadata=read(native/'bundle.json'),blob=(native/'bundle.bin').read_bytes())
            helper.validate_bundle(bundle)
            require(not {'numpy','torch','flopscope','threadpoolctl'}&set(sys.modules),
                    'Torch stage admission entirely before numerical imports')
            import numpy as np
            import torch
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            diagnostic=helper.torch_phase(context,bundle,np,torch,refs)
            gc.collect()
            require(all(r() is None for r in refs),'All Torch numerical owners retired')
            write(output/'diagnostics.json',diagnostic)
            result.update(native_seals=seals,all_owners_retired=True,
                          rangefinder_dominant_support=diagnostic['rangefinder_dominant_support'],
                          diagnostic_outcome=diagnostic['outcome'],
                          native_Y_over_own_Y_squared_error=diagnostic['native_Y_over_own_Y_squared_error'],
                          complete_source_planning_upper=context['cost_certificate']['complete_planning_upper'])
            artifacts=('diagnostics.json',)
        result['artifacts']={str(output/k):sha(output/k) for k in artifacts}
        require(all(sha(k)==v for roster in ('source_bindings','input_bindings')
                    for k,v in result[roster].items()) and
                all(sha(HERE/k)==v for k,v in pins.items()),'Sources and inputs unchanged')
        tick();result.update(passed=True,completed=True)
    except BaseException:
        result['error']=traceback.format_exc()
    result.update(elapsed_s=time.monotonic()-began,
                  maximum_self_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['elapsed_s']>=120 or result['maximum_self_rss_bytes']>=8_000_000_000:
        result.update(passed=False,completed=False,resource_error=True)
    write(output/'result.json',result)
    rh=sha(output/'result.json')
    write(output/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=rh))
    write(output/'finalization.json',dict(passed=result['passed'],completed=result['completed'],
        result_sha256=rh,completion_sha256=sha(output/'completion.json'),elapsed_s=time.monotonic()-began))
    signal.alarm(0)
    print(json.dumps({k:result.get(k) for k in ('phase','passed','completed','error',
        'diagnostic_outcome','native_Y_over_own_Y_squared_error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
