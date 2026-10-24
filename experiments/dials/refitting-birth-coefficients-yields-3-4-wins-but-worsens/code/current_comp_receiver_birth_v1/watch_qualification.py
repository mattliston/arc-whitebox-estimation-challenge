"""Bounded one-MPS qualification supervisor. No numerical imports."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
NEW,P2=HERE.parent,HERE.parent.parent
MON=NEW/'current_p1_native_objective_p2_monitor_v2/monitor.py'
MON_SHA='d4b3af53b716593670d8e0c7f0fc6f2133b57f1088d9966c06635d3a23f39873'
P1W=NEW/'current_d50_native_objective_p1_v1/watch.py'
P1W_SHA='4e8177d6ff943cdee37e48ccca750bf9d552b32cec344aa2b37e6304cbd55fb3'
AP='/Users/matt/anaconda3/bin/python'


def check(ok,message):
    if not ok: raise RuntimeError(message)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true');ap.add_argument('--mps-slot-clear',action='store_true')
    for k in ('source','worker','model','protocol'):ap.add_argument('--'+k+'-sha256')
    args=ap.parse_args()
    if not args.run:ap.print_help();return 0
    began=time.monotonic();check(args.mps_slot_clear,'Explicit MPS slot attestation')
    check(os.getpriority(os.PRIO_PROCESS,0)>=10 and sys.flags.optimize==0,'nice10/no optimized checks')
    pins={Path(__file__).resolve():args.source_sha256,HERE/'qualification.py':args.worker_sha256,
          HERE/'model.py':args.model_sha256,HERE/'PROTOCOL.md':args.protocol_sha256,
          HERE/'WATCH_SPEC.md':sha(HERE/'WATCH_SPEC.md'),MON:MON_SHA,P1W:P1W_SHA}
    for p,h in pins.items():check(re.fullmatch('[0-9a-f]{64}',h or '') and sha(p)==h,'Reviewed source '+str(p))
    spec=importlib.util.spec_from_file_location('receiver_monitor',MON)
    mon=importlib.util.module_from_spec(spec);spec.loader.exec_module(mon)
    source=P1W.read_text();tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='other_workers']
    check(len(functions)==1,'Unique unchanged classifier')
    env=dict(Path=Path,re=re,shlex=shlex,P2=P2,NEW=NEW)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(P1W)+':classifier','exec'),env)
    out=HERE/'qualification_watch_v1';out.mkdir(exist_ok=False)
    result=dict(passed=False,source_bindings={str(p):h for p,h in pins.items()},
                mps_slot_clear_attested=True,numerical_imports=False,whole_cap_s=180,
                maximum_combined_rss_bytes=0,maximum_interval_s=0,retired_calls=0)
    child=None;log=None;readlog=None;last_progress=began;previous=time.monotonic()
    swap0=None;pending='';samples=0
    def interrupted(*_):raise RuntimeError('Qualification interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def observe(stream):
        nonlocal previous,swap0,last_progress,pending,samples
        deadline=previous+1.
        sample=mon.collect(deadline)
        s=mon.scope(sample['rows'],os.getpid(),None if child is None else child.pid)
        check(not env['other_workers'](sample['rows'],s['allowed']), 'Foreign numeric worker; stop only ours')
        if swap0 is None:swap0=sample['swap']['used_bytes']
        check(sample['swap']['used_bytes']<=swap0,'No added swap')
        check(s['combined_rss_bytes']<12_000_000_000,'12GB combined RSS')
        check(time.monotonic()-began<180,'180s whole cap')
        if readlog is not None:
            pending+=readlog.read()
            while '\n' in pending:
                line,pending=pending.split('\n',1)
                try:item=json.loads(line)
                except (ValueError,TypeError):continue
                if isinstance(item,dict) and item.get('event')=='retired':
                    check(time.monotonic()-last_progress<120,'Prior retired-call deadline already expired')
                    check(item['call']==result['retired_calls'],'Ordered retired call marker')
                    result['retired_calls']+=1;last_progress=time.monotonic()
        check(time.monotonic()-last_progress<120,'120s without fully retired call')
        result['maximum_combined_rss_bytes']=max(result['maximum_combined_rss_bytes'],s['combined_rss_bytes'])
        sample_out=dict(elapsed_s=time.monotonic()-began,combined_rss_bytes=s['combined_rss_bytes'],
                        swap_bytes=sample['swap']['used_bytes'],child_pids=s['child_pids'],retired_calls=result['retired_calls'])
        stream.write(json.dumps(sample_out,sort_keys=True)+'\n');stream.flush()
        previous,interval=mon.complete(previous,deadline)
        result['maximum_interval_s']=max(result['maximum_interval_s'],interval);samples+=1
    try:
        with (out/'samples.jsonl').open('x') as stream:
            observe(stream)
            child_env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0',
                           PYTORCH_MPS_LOW_WATERMARK_RATIO='0.1',PYTHONUNBUFFERED='1')
            for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):child_env[k]='1'
            argv=[AP,str(HERE/'qualification.py'),'--run','--source-sha256',args.worker_sha256,
                  '--model-sha256',args.model_sha256,'--protocol-sha256',args.protocol_sha256,
                  '--watcher-sha256',args.source_sha256]
            log=(out/'worker.log').open('x');readlog=(out/'worker.log').open('r')
            child=subprocess.Popen(argv,cwd=NEW,env=child_env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            result['child_pid']=child.pid;write(out/'started.json',result)
            while child.poll() is None:
                observe(stream);time.sleep(.15)
            observe(stream);check(child.wait(timeout=1)==0,'Worker exit')
            mon.cleanup([child],time.monotonic()+.2)
            record=HERE/'qualification_v1/result.json';end=record.with_name('completion.json')
            d=json.loads(record.read_text());c=json.loads(end.read_text())
            final=record.with_name('finalization.json');f=json.loads(final.read_text())
            check(d['passed'] and d['completed'] and c['passed'] and c['result_sha256']==sha(record)
                  and f['passed'] and f['completed'] and f['result_sha256']==sha(record)
                  and f['completion_sha256']==sha(end) and f['elapsed_s']<180,'Complete passed worker/finalization seals')
            check(result['retired_calls']==d['retired_call_count']==8
                  and d['full_model_calls']==7 and d['tiny_model_calls']==8,'Complete fixed qualification roster')
            for p,h in pins.items():check(sha(p)==h,'Sources unchanged')
            observe(stream)
            result.update(passed=True,child_result_sha256=sha(record),child_completion_sha256=sha(end),
                          child_finalization_sha256=sha(final),whole_owned_tree_retired=True,sources_unchanged=True)
    except BaseException as e:
        result.update(passed=False,error=f'{type(e).__name__}: {e}')
    finally:
        try:
            if child is not None:mon.cleanup([child],time.monotonic()+.5)
            mon.retire_pending(time.monotonic()+.5)
            result['cleanup_passed']=True
        except BaseException as e:result.update(passed=False,cleanup_error=str(e))
        if log:log.close()
        if readlog:readlog.close()
        result.update(elapsed_s=time.monotonic()-began,observation_samples=samples,initial_swap_bytes=swap0)
        if result['elapsed_s']>=180:
            result.update(passed=False,error='180s whole cap including cleanup and log close')
        write(out/'result.json',result)
        result_digest=sha(out/'result.json')
        sealed_elapsed=time.monotonic()-began
        completed_passed=result['passed'] and sealed_elapsed<180
        write(out/'completion.json',dict(passed=completed_passed,result_sha256=result_digest,
              elapsed_through_result_hash_s=sealed_elapsed))
        # A separately required final seal covers completion I/O, too. A late
        # or absent finalization receipt cannot qualify the experiment.
        completion_digest=sha(out/'completion.json')
        final_elapsed=time.monotonic()-began
        final_passed=completed_passed and final_elapsed<180
        write(out/'finalization.json',dict(passed=final_passed,
              elapsed_through_completion_io_s=final_elapsed,
              completion_sha256=completion_digest))
        result['passed']=final_passed
    print(json.dumps(result,indent=2));return 0 if result['passed'] else 1


if __name__=='__main__':sys.exit(main())
