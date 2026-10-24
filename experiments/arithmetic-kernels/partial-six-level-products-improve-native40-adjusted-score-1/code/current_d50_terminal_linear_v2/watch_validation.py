"""One explicit native40 acquisition OR its offline actual-objective judge.

Reuse the frozen local stdlib process-group/RSS/swap observer. No automatic
scoring, refit, panel or release. A completed scientific non-advance is an
operationally completed score, not a successful scientific gate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
P2 = HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
PROTOCOL_SHA = 'febed4bdccc7c031cb846e6d97f695536ca98d3bef10d07de48f1418a91b6a58'
HELPER_SHA = 'b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd'
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--stage',choices=('native','score'))
    for name in ('source','worker','protocol','fit','native-source','native-result','native-watch'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(args.stage is not None,'Explicit separate native or score stage')
    fields = ('source','worker','protocol','fit') + (() if args.stage == 'native' else
              ('native-source','native-result','native-watch'))
    for name in fields:
        value = getattr(args,name.replace('-','_')+'_sha256')
        require(isinstance(value,str) and len(value) == 64 and set(value) <= set('0123456789abcdef'),
                'Reviewed future hash '+name)
    require(args.protocol_sha256 == PROTOCOL_SHA and os.getpriority(os.PRIO_PROCESS,0) >= 10
            and all(os.environ.get(k) == '1' for k in BLAS)
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0','Fixed nice10 BLAS1 source frame')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')),'Stdlib-only supervisor')
    worker = HERE/('native_validation.py' if args.stage == 'native' else 'judge_terminal.py')
    child_out = HERE/('validation_terminal_v1' if args.stage == 'native' else 'terminal_score_v1')
    out = HERE/('validation_watch_v1' if args.stage == 'native' else 'terminal_score_watch_v1')
    wall_limit,rss_limit = (1800.0,8_000_000_000) if args.stage == 'native' else (120.0,2_000_000_000)
    pins = {str(Path(__file__).resolve()):args.source_sha256,str(worker):args.worker_sha256,
            str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,str(HERE/'watch_fit.py'):HELPER_SHA,
            str(HERE/'fit_v1/result.json'):args.fit_sha256}
    fit_end = HERE/'fit_v1/completion.json'
    fit,seal = read(HERE/'fit_v1/result.json'),read(fit_end)
    require(fit['passed'] and fit['completed'] and fit['hashes_unchanged']
            and seal['passed'] and seal['completed'] and seal['result_sha256'] == args.fit_sha256,
            'Completed sole fit before any validation stage')
    pins[str(fit_end)] = sha(fit_end)
    if args.stage == 'score':
        pins.update({str(HERE/'native_validation.py'):args.native_source_sha256,
            str(HERE/'validation_terminal_v1/result.json'):args.native_result_sha256,
            str(HERE/'validation_watch_v1/result.json'):args.native_watch_sha256})
    else:
        require(all(getattr(args,k) is None for k in
            ('native_source_sha256','native_result_sha256','native_watch_sha256')),'No score inputs to native stage')
    for path,digest in pins.items():
        require(sha(path) == digest,'Reviewed stage input changed '+path)
    require(not out.exists() and not child_out.exists(),'Exclusive stage and child paths; no resume')
    spec = importlib.util.spec_from_file_location('tl_validation_resource_helpers',HERE/'watch_fit.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    command = [str(OFFPY),'-B','-u',str(worker),'--run','--source-sha256',args.worker_sha256,
               '--protocol-sha256',PROTOCOL_SHA,'--fit-sha256',args.fit_sha256]
    if args.stage == 'score':
        for key in ('native-source','native-result','native-watch'):
            command += ['--'+key+'-sha256',getattr(args,key.replace('-','_')+'_sha256')]
    out.mkdir()
    result = dict(schema='current_d50_terminal_linear_validation_watch_v1',stage=args.stage,
        passed=False,completed=False,hashes_unchanged=False,truth_arrays_opened=False,
        child_execution_completed=False,whole_owned_tree_retired=False,scientific_advancement=False,
        source_bindings=pins,artifacts={},requested_seals=vars(args),command=command,
        wall_limit_s=wall_limit,combined_rss_limit_bytes=rss_limit,swap_growth_limit_bytes=512*1024**2,
        requested_sample_period_s=.2,sample_interval_limit_s=.5,maximum_sample_interval_s=0.0,
        maximum_sampled_rss_bytes=0,maximum_swap_growth_bytes=0.0,samples=0,
        started_utc=datetime.now(timezone.utc).isoformat(),
        limitations=['Sampled RSS/swap, not continuous peaks.','No production or independent-panel qualification.'])
    child = None; handlers = {}; started = last = time.monotonic(); before = None
    def interrupted(signum,frame):
        raise RuntimeError('Supervisor interrupted '+str(signum))
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig] = signal.signal(sig,interrupted)
        before = helper.swap_state(); result['before_swap'] = before
        write(out/'started.json',result)
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr, \
                (out/'resources.jsonl').open('x') as resources:
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                     env=os.environ.copy(),start_new_session=True)
            result['child_pid'] = result['child_process_group'] = child.pid
            while True:
                code = child.poll()
                memory = helper.memory_snapshot(child.pid); swap = helper.swap_state()
                now = time.monotonic(); interval = now-last; last = now
                growth = swap['used_bytes']-before['used_bytes']
                memory.update(elapsed_s=now-started,interval_s=interval,swap=swap,swap_growth_bytes=growth)
                resources.write(json.dumps(memory,allow_nan=False)+'\n'); resources.flush()
                result['samples'] += 1
                result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
                result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],memory['combined_rss_bytes'])
                result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
                require(now-started < wall_limit and interval <= .5 and memory['combined_rss_bytes'] < rss_limit
                        and growth <= 512*1024**2,'Whole-stage wall/sample/RSS/swap guard')
                require(memory['child_pids'] or code is not None or child.poll() is not None,'Live child observable')
                if code is not None:
                    result['child_exit_code'] = code
                    require(not memory['child_pids'],'No surviving child descendants')
                    break
                time.sleep(.2)
        require(result['child_exit_code'] == 0,'Native acquisition or score integrity failed')
        rp,cp = child_out/'result.json',child_out/'completion.json'
        record,end = read(rp),read(cp)
        result.update(child_result_sha256=sha(rp),child_completion_sha256=sha(cp))
        require(record['completed'] and record['hashes_unchanged'] and end['completed']
                and end['result_sha256'] == sha(rp) and end['passed'] is record['passed'],'Complete sealed child')
        if args.stage == 'native':
            require(record['passed'] and len(record['trials']) == 41 and record['truth_accessed'] is False,
                    'Completed target-free native40+repeat')
        else:
            require(record['integrity_passed'],'Score integrity; scientific non-advance is allowed')
        result.update(child_execution_completed=True,whole_owned_tree_retired=True,
                      scientific_advancement=bool(record.get('advances_to_independent14',False)))
        for path in (out/'started.json',out/'stdout.log',out/'stderr.log',out/'resources.jsonl',rp,cp):
            result['artifacts'][str(path)] = sha(path)
        for path,digest in pins.items():
            require(sha(path) == digest,'Reviewed source/receipt changed '+path)
        memory = helper.memory_snapshot(child.pid); swap = helper.swap_state()
        result.update(final_resource_observation=memory,after_swap=swap)
        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],memory['combined_rss_bytes'])
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],swap['used_bytes']-before['used_bytes'])
        require(not memory['child_pids'] and memory['combined_rss_bytes'] < rss_limit
                and result['maximum_swap_growth_bytes'] <= 512*1024**2 and time.monotonic()-started < wall_limit,
                'Final guard after completion/artifact hashing')
        result.update(passed=True,completed=True,hashes_unchanged=True)
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,error=traceback.format_exc())
        if child is not None:
            try:
                helper.stop_owned_group(child); result['owned_group_cleanup_complete'] = True
            except BaseException:
                result['cleanup_error'] = traceback.format_exc()
    finally:
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
        for path in (out/'started.json',out/'stdout.log',out/'stderr.log',out/'resources.jsonl',
                     child_out/'result.json',child_out/'completion.json'):
            if path.is_file() and str(path) not in result['artifacts']:
                result['artifacts'][str(path)] = sha(path)
        result.update(elapsed_s=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/'result.json',result)
        write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            result_sha256=sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('stage','passed','completed','elapsed_s','scientific_advancement','error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
