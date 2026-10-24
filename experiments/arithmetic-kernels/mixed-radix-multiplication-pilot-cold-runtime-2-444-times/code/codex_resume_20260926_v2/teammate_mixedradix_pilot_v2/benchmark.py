"""ROOT-ONLY native execution; fixed cases, no search or estimator run.

Use the existing OpenBLAS runtime and root's serial 7.5GB guard. This script
does not itself authorize or launch any native work from a teammate.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

from kernel import CASES, HERE, Hybrid, binary_environment


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',required=True,choices=[c['name'] for c in CASES])
    args=parser.parse_args()
    case=next(c for c in CASES if c['name']==args.case)
    output=HERE/f"native_{case['name']}.json"
    assert not output.exists(),f'Preserve existing result: {output}'
    for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
        assert os.environ.get(variable)=='1',(variable,os.environ.get(variable))
    assert os.getpriority(os.PRIO_PROCESS,0)>=10,'Root must launch nice 10'
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    assert importlib.metadata.version('flopscope')=='0.12.0'
    static=json.loads((HERE/'static.json').read_text())
    expected=next(r for r in static['cases'] if r['case']['name']==case['name'])
    seed=930620+next(i for i,c in enumerate(CASES) if c['name']==case['name'])
    rng=np.random.default_rng(seed)
    m,k,n=case['shape']
    # Harness-only fixture generation/reference. Candidate operations below
    # receive only FNP inputs and use the fully metered native backend.
    aa=rng.standard_normal((m,k),dtype=np.float32)/np.float32(np.sqrt(k))
    bb=rng.standard_normal((k,n),dtype=np.float32)
    reference=aa.astype(np.float64)@bb.astype(np.float64)
    reference_norm=float(np.linalg.norm(reference))
    A=fnp.asarray(aa);B=fnp.asarray(bb);out=fnp.empty((m,n),dtype=fnp.float32)
    del aa,bb
    shared=fnp.empty(static['shared_arena_elements'],dtype=fnp.float32)
    base=binary_environment(fnp,shared,case['shape'],1)
    hybrid=Hybrid(fnp,shared,case,expected['temporary_high_water_elements'])
    records=[]
    result=dict(case=case,seed=seed,records=records,completed=False,passed=False,
                source_sha256=sha(Path(__file__)),kernel_sha256=sha(HERE/'kernel.py'),
                static_sha256=sha(HERE/'static.json'),python=sys.executable,
                numpy_version=np.__version__,flopscope_version=importlib.metadata.version('flopscope'),
                shared_arena_bytes=static['shared_arena_bytes'],
                started_utc=datetime.now(timezone.utc).isoformat(),
                qualification='Synthetic kernel fixture only; no estimator accuracy or bankable global score claim.')
    started=time.monotonic()
    def save(): output.write_text(json.dumps(result,indent=2)+'\n')
    base_value=None
    try:
        # Fixed interleaving limits drift; both kernels rewrite every input tree.
        for repetition in ('cold','warm'):
            for kind in ('parent','smirnov'):
                gc.collect()
                operation=(lambda:base['_ipt_product_ab'](A,B,out,5)) if kind=='parent' else (lambda:hybrid(A,B,out))
                target=next(r for r in expected['records'] if r['kind']==kind and r['repetition']==repetition)
                usage0=resource.getrusage(resource.RUSAGE_SELF)
                with flops.BudgetContext(flop_budget=2**36,wall_time_limit_s=60,quiet=True) as context:
                    cpu0=time.process_time(); wall0=time.perf_counter()
                    operation()
                    elapsed=time.perf_counter()-wall0;cpu=time.process_time()-cpu0
                usage1=resource.getrusage(resource.RUSAGE_SELF)
                value=np.asarray(out).copy()
                value64=value.astype(np.float64)
                error=float(np.linalg.norm(value64-reference)/reference_norm)
                row=dict(kind=kind,repetition=repetition,wall_s=elapsed,cpu_s=cpu,
                         user_s=usage1.ru_utime-usage0.ru_utime,system_s=usage1.ru_stime-usage0.ru_stime,
                         peak_process_rss_bytes=int(usage1.ru_maxrss),
                         flops=int(context.flops_used),expected_flops=target['flops'],
                         bill_pass=int(context.flops_used)==target['flops'],
                         relative_frobenius=error,max_absolute_error=float(np.max(np.abs(value64-reference))),
                         finite=bool(np.isfinite(value).all()),
                         output_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                         context=context.summary_dict())
                if kind=='parent': base_value=value
                else:
                    row['relative_to_parent']=float(np.linalg.norm(value64-base_value.astype(np.float64))/reference_norm)
                records.append(row)
                result['outer_workspace_bytes']=hybrid.workspace_bytes
                result['peak_process_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                save()
                print(json.dumps({k:v for k,v in row.items() if k!='context'}),flush=True)
                assert row['bill_pass'],('Native FLOPs disagree with exact executable static accounting',row)
                assert row['finite'],('Nonfinite fixed-fixture output',row)
                del value64,value
        timing={}
        for repetition,bound in [('cold',1.5),('warm',1.2)]:
            parent=next(r for r in records if r['kind']=='parent' and r['repetition']==repetition)
            trial=next(r for r in records if r['kind']=='smirnov' and r['repetition']==repetition)
            timing[repetition]=dict(wall_ratio=trial['wall_s']/parent['wall_s'],
                                    cpu_ratio=trial['cpu_s']/parent['cpu_s'],threshold=bound,
                                    passed=trial['wall_s']<=bound*parent['wall_s'],
                                    flop_saving_fraction=1-trial['flops']/parent['flops'])
        numeric=all(r['relative_frobenius']<=2e-4 for r in records)
        memory=result['peak_process_rss_bytes']<7_500_000_000
        result.update(completed=True,timing=timing,numerical_pass=numeric,memory_pass=memory,
                      passed=numeric and memory and all(x['passed'] for x in timing.values()))
    except BaseException as error:
        result['error']=repr(error)
        raise
    finally:
        result['elapsed_s']=time.monotonic()-started
        result['finished_utc']=datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__': main()
