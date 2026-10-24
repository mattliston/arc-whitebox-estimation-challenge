"""One prospective N/plus/minus native local-cell acquisition; no training."""
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
OFFPY = HERE.parent.parent/'data_gate/.venv/bin/python'
NAMES = {'run_native.py','native_cell.py','fixed_direction.py','boundary_adapter.py',
         'supervise.py','PROTOCOL.md'}
LIMITS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(ok,message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''): h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')


def load(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true'); ap.add_argument('--pins')
    args = ap.parse_args()
    if not args.run: ap.print_help(); return 0
    started = time.monotonic(); output = HERE/'native_v1'
    require(not output.exists(), 'Fresh exclusive native acquisition')
    output.mkdir()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Whole native120s')))
    signal.alarm(120)
    result = dict(passed=False,completed=False,phase='native',truth_accessed=False,
        fits=0,parameter_updates=0,full_model_calls=0,partial_model_calls=0,
        retired_calls=0,trials=[],source_bindings={},input_bindings={},artifacts={})
    def tick():
        require(time.monotonic()-started < 120 and
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 8_000_000_000,
            'Whole120s/8GB; external supervisor owns tree/swap')
    try:
        pins = json.loads(Path(args.pins).read_text())
        require(set(pins) == NAMES and all(sha(HERE/k) == v for k,v in pins.items()),
            'Exact six-owner prospective native roster')
        require(Path(sys.executable).resolve() == OFFPY.resolve()
            and os.getpriority(os.PRIO_PROCESS,0) >= 10
            and sys.flags.optimize == 0 and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
            and not os.environ.get('FLOPSCOPE_SERVER_URL')
            and all(os.environ.get(k) == '1' for k in LIMITS),
            'Fixed interpreter/nice10/sixBLAS1/assertions enabled/fallback0/native backend')
        helper = load(HERE/'native_cell.py','reviewed_native_cell')
        direction = load(HERE/'fixed_direction.py','reviewed_fixed_direction')
        boundary = load(HERE/'boundary_adapter.py','reviewed_boundary_schema')
        context,apis = helper.prepare(pins['native_cell.py'],tick)
        j,c,a,h,backend,legacy,owner = apis
        for k in ('source_bindings','input_bindings','capture_seals','source_proof','cost_certificate'):
            result[k] = context[k]
        for name,value in pins.items(): result['source_bindings'][str(HERE/name)] = value
        result['admission_source_bindings'] = dict(result['source_bindings'])
        boundary.describe(context['manifest'],'entry8')
        ledger = dict(time=time,hashlib=hashlib,json=json,require=require)
        exec(compile(context['ledger_source'],'<original-closed-native-ledger>','exec'),ledger)
        metrics = ledger['_v2_ledger_metrics'](legacy.metrics)
        backend.backend_select(); h.environment()
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits,threadpool_info
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        branches = {}
        with threadpool_limits(limits=1):
            result['threadpools'] = backend.backend_threadpool_info(threadpool_info,result,tick)()
            for arm in ('N','plus','minus'):
                tick(); refs=[]; began=time.monotonic(); call_error=None; answer=None; ctx=None
                try:
                    with flops.BudgetContext(flop_budget=context['cost_certificate']['native_upper'],
                            wall_time_limit_s=120,quiet=True) as ctx:
                        result['partial_model_calls'] += 1
                        answer = helper.phase(context,apis,np,flops,fnp,ctx,refs,arm,direction.perturb,
                            MLP,derive_estimator_seed,materialise_predictions)
                except BaseException:
                    # Preserve actual attempted-call cost even if a numerical,
                    # parity or ownership assertion failed inside the context.
                    call_error=traceback.format_exc()
                require(ctx is not None,'Context construction failed before any numerical call')
                bill = dict(arm=arm,**metrics(ctx,began),complete_research_flops=int(ctx.flops_used),
                    includes_initialization=True,includes_all_native_input_and_snapshot_work=True,
                    not_a_complete_estimator_score=True,call_completed=call_error is None,
                    call_error=call_error)
                bp = output/(arm+'_bill.json'); write(bp,bill)
                cr = weakref.ref(ctx); life = legacy.reset_saved_history(flops,ctx,bp)
                del ctx; gc.collect()
                retired=cr() is None and all(r() is None for r in refs)
                life.update(context_released=cr() is None,all_numerical_array_owners_retired=retired)
                if retired: result['retired_calls'] += 1
                write(output/(arm+'_lifecycle.json'),life)
                require(retired and life['closed_context_bill_unchanged_by_reset'],
                    'All owned native arrays/context retired')
                require(call_error is None,'Native call failed; complete attempted bill retained:\n'+str(call_error))
                require(0 < bill['flops'] == bill['complete_research_flops'] <=
                    context['cost_certificate']['native_upper'] and bill['wall_s'] < 120
                    and bill['context_wall_s'] < 120 and bill['peak_rss_bytes'] < 8_000_000_000
                    and all(math.isfinite(bill[k]) and bill[k] >= 0 for k in legacy.TIMINGS),
                    'Closed own complete native bill inside prospective envelope')
                write(output/(arm+'_diagnostics.json'),answer['diagnostic'])
                branches[arm]={k:{name:row[name] for name in
                    ('count','floor','indices_sha256','mask_sha256')}
                    for k,row in answer['diagnostic']['domains']['selection_cells'].items()}
                with (output/(arm+'_served.bin')).open('xb') as stream: stream.write(answer['served'])
                if arm == 'N':
                    bundle = answer['bundle']
                    for name in ('entry8','end8'): boundary.describe(bundle['metadata'],name)
                    with (output/'boundary.bin').open('xb') as stream: stream.write(bundle['blob'])
                    write(output/'boundary.json',bundle['metadata'])
                    del bundle
                result['trials'].append(dict(arm=arm,flops=bill['flops'],wall_s=bill['wall_s'],
                    served_sha256=answer['served_sha256'],all_owners_retired=True))
                del answer; gc.collect(); tick()
        backend.backend_origins(result,tick)
        result['artifacts'] = {str(p):sha(p) for p in output.iterdir() if p.is_file()}
        require(result['partial_model_calls'] == result['retired_calls'] == 3,
            'Exactly three fully retired prospective native calls')
        require(all(sha(k) == v for roster in ('source_bindings','input_bindings')
            for k,v in result[roster].items()), 'All actual immutable owners unchanged')
        identical=all(branches[arm]==branches['N'] for arm in ('plus','minus'))
        result.update(passed=identical,completed=True,all_owners_retired=True,
            native_ordered_selectors_and_masks_identical=identical,branch_inventory=branches,
            outcome='native_primal_and_branch_qualified' if identical else 'negative_changed_native_branch',
            complete_research_flops=sum(x['flops'] for x in result['trials']),
            secant_qualified=False,score_measured=False,native_gradient_certified=False)
    except BaseException:
        result['error'] = traceback.format_exc()
    result.update(elapsed_s=time.monotonic()-started,
        maximum_self_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['elapsed_s'] >= 120 or result['maximum_self_rss_bytes'] >= 8_000_000_000:
        result.update(passed=False,completed=False,resource_error=True)
    write(output/'result.json',result); rh=sha(output/'result.json')
    write(output/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=rh))
    write(output/'finalization.json',dict(passed=result['passed'],completed=result['completed'],
        result_sha256=rh,completion_sha256=sha(output/'completion.json'),elapsed_s=time.monotonic()-started))
    signal.alarm(0)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','partial_model_calls',
        'retired_calls','complete_research_flops','elapsed_s','error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
