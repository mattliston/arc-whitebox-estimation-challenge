"""Inert one-child resource watch for the fixed two-TRAIN TREE judge.

Operational PASS requires a completed, integrity-qualified judge, regardless
of its scientific PASS/FAIL. No model, fitting, target IO, or numerical imports
in this supervisor. Only its fresh owned child process group may be signalled.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent,HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
HELPER = NEW/'current_d50_return_quantization_v1/watch.py'
HELPER_SHA = 'f79749448c5a6015540474b18b58b6981944fc8b8c9182200effc0095728e32b'
JUDGE_SHA = '6ca49261fc358cc8f4310dd9e2296e63ff32c18328e8720733d65b5e1180cd83'
NATIVE_WATCH_SHA = 'e4d9b850af941b900ff33117b8a0163740cb20dde5b27173c9b01f8570445131'
NATIVE_SHA = '55d280e2ceac4373d68e46702636d1ed70a4fffeb473e87e0505ed7ec5d34484'
PROTOCOL_SHA = 'f4bf2907fa1973b6076d6f485cd2436d6b5bf13cfe26077653e52af3c3e59dad'
OUT, CHILD_OUT = HERE/'judgment_watch_v1',HERE/'judgment_v1'
WALL, RSS, SWAP = 120.0,2_000_000_000,512*1024**2
PERIOD, MAX_INTERVAL = .2,1.0
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),'Watcher never accesses target arrays')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for key in ('source','judge','protocol','native-watch-result'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    require(all(re.fullmatch('[0-9a-f]{64}',v or '') for k,v in vars(args).items() if k != 'run'),
            'Every reviewed SHA is required')
    require(args.judge_sha256 == JUDGE_SHA and args.protocol_sha256 == PROTOCOL_SHA,'Sole fixed judge/protocol')
    started = time.monotonic()
    sources = {str(Path(__file__).resolve()):args.source_sha256,str(HERE/'judge.py'):JUDGE_SHA,
        str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,str(HERE/'watch.py'):NATIVE_WATCH_SHA,
        str(HERE/'native.py'):NATIVE_SHA,str(HELPER):HELPER_SHA}
    for path,digest in sources.items():
        require(sha(path) == digest,'Reviewed source: '+path)
    result_path,end_path = HERE/'native_watch_v1/result.json',HERE/'native_watch_v1/completion.json'
    require(sha(result_path) == args.native_watch_result_sha256,'Immutable complete five-call result')
    native,end = read(result_path),read(end_path)
    require(native['schema'] == 'terminal_full_fourth_tree_native_watch_v1' and
            all(native.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
                'all5_complete','all_predictions_and_bills_sealed_before_scoring','repeat_byte_and_bill_exact',
                'zero_all16_bytes_and_bill_exact','earlier15_rows_byte_exact','whole_owned_tree_retired')) and
            end['passed'] is True and end['completed'] is True and
            end['result_sha256'] == args.native_watch_result_sha256 and native['truth_accessed'] is False and
            native['source_bindings'][str(HERE/'watch.py')] == NATIVE_WATCH_SHA and
            native['source_bindings'][str(HERE/'native.py')] == NATIVE_SHA,
            'Fixed target-free native acquisition completed')
    inputs = {str(result_path):args.native_watch_result_sha256,str(end_path):sha(end_path)}
    require(not any(x in sys.modules for x in ('numpy','torch','flopscope')),'Stdlib-only supervisor')
    spec = importlib.util.spec_from_file_location('terminal_tree_judge_resource_helpers',HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    require(OFFPY.is_file() and os.getpriority(os.PRIO_PROCESS,0) >= 10,'OFFPY/nice10')
    environment = {key:'1' for key in BLAS}
    environment.update(PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
    require(all(os.environ.get(k) == v for k,v in environment.items()),'Explicit inherited BLAS1/no-MPS environment')
    require(not CHILD_OUT.exists(),'No repeated or resumed judgment')
    OUT.mkdir(exist_ok=False)
    command = [str(OFFPY),'-B','-u',str(HERE/'judge.py'),'--run','--source-sha256',JUDGE_SHA,
        '--protocol-sha256',PROTOCOL_SHA,'--native-watch-result-sha256',args.native_watch_result_sha256]
    result = dict(schema='terminal_full_fourth_tree_judge_watch_v1',passed=False,completed=False,
        hashes_unchanged=False,sources_unchanged=False,source_bindings=sources,input_bindings=inputs,artifacts={},
        prerequisites=dict(native_watch_result_sha256=args.native_watch_result_sha256),command=command,
        operational_verdict_only=True,scientific_advancement_claimed=False,truth_accessed=False,numerical_imports=False,
        wall_limit_s=WALL,combined_rss_limit_bytes=RSS,swap_growth_limit_bytes=SWAP,
        requested_sample_period_s=PERIOD,sample_interval_limit_s=MAX_INTERVAL,
        maximum_sample_interval_s=0.0,maximum_sampled_rss_bytes=0,maximum_swap_growth_bytes=0.0,samples=0,
        limitations=['RSS/swap are sampled, not continuous peaks.',
            'The one-second interval bound covers the live child; final metadata sealing has a separate final sample.',
            'Operational PASS does not require or imply scientific improvement.'],
        started_utc=datetime.now(timezone.utc).isoformat())
    child,previous,before = None,{},None
    last = time.monotonic()
    def interrupted(signum,frame):
        raise RuntimeError('Judge watcher interrupted: '+str(signum))
    def observe(stream=None,active=True):
        nonlocal last
        memory = helper.memory_snapshot(child.pid if child is not None else -1)
        sampled = time.monotonic()
        interval,last = sampled-last,sampled
        swap = helper.swap_state()
        growth = swap['used_bytes']-before['used_bytes']
        item = dict(elapsed_s=time.monotonic()-started,interval_s=interval,**memory,
                    swap=swap,swap_growth_bytes=growth,live_child_interval_gate=active)
        result['samples'] += 1
        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],memory['combined_rss_bytes'])
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        if active:
            result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
            require(interval <= MAX_INTERVAL,'Independent resource sampling cadence')
        if stream is not None:
            stream.write(json.dumps(item,allow_nan=False)+'\n');stream.flush()
        require(item['elapsed_s'] < WALL and memory['combined_rss_bytes'] < RSS and growth <= SWAP,
                'Independent whole120s/combined2GB/swap limit')
        return item
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            previous[sig] = signal.signal(sig,interrupted)
        before = helper.swap_state()
        result['before_swap'] = before
        write(OUT/'started.json',result)
        with (OUT/'stdout.log').open('xb') as stdout,(OUT/'stderr.log').open('xb') as stderr,\
                (OUT/'resources.jsonl').open('x') as resources:
            observe(resources,active=False)
            last = time.monotonic()
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                env=os.environ.copy(),start_new_session=True)
            result['child_pid'] = result['child_process_group'] = child.pid
            while True:
                code = child.poll()
                item = observe(resources)
                require(item['child_pids'] or code is not None or child.poll() is not None,'Live child tree visible')
                if code is not None:
                    result['child_exit_code'] = code
                    require(not item['child_pids'],'No surviving owned descendants')
                    break
                time.sleep(PERIOD)
        require(result['child_exit_code'] == 0,'Judge did not complete with integrity')
        rp,cp = CHILD_OUT/'result.json',CHILD_OUT/'completion.json'
        record,seal = read(rp),read(cp)
        result.update(child_result_sha256=sha(rp),child_completion_sha256=sha(cp))
        require(record['schema'] == 'terminal_full_fourth_tree_two_train_score_v1' and
                all(record.get(k) is True for k in ('completed','integrity_passed','hashes_unchanged','sources_unchanged',
                    'all5_native_verified','all_predictions_verified_before_truth')) and
                seal['completed'] is True and seal['integrity_passed'] is True and
                seal['result_sha256'] == result['child_result_sha256'] and seal['passed'] is record['passed'] and
                record['source_bindings'][str(HERE/'judge.py')] == JUDGE_SHA and
                record['prerequisites']['native_watch_result_sha256'] == args.native_watch_result_sha256,
                'Complete fixed judge integrity and source seals; scientific PASS is not required')
        require(record['target_ranks_accessed'] == [35,100] and len(record['records']) == 2 and
                record['model_calls'] == record['gradient_calls'] == record['parameter_updates'] == 0 and
                record['validation_truths_accessed'] is False and record['independent14_score_measured'] is False and
                record['board_qualified'] is False and record['peak_rss_bytes'] < RSS and record['elapsed_s'] < WALL,
                'Only the fixed two-TRAIN offline score')
        result.update(passed=True,completed=True,child_execution_completed=True,whole_owned_tree_retired=True,
            child_scientific_passed=record['passed'],child_both_adjusted_improve=record['both_adjusted_improve'])
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
        if child is not None:
            try:
                helper.stop_owned_group(child)
                result['whole_owned_tree_retired'] = True
            except BaseException:
                result['cleanup_error'] = traceback.format_exc()
    finally:
        try:
            for path in (OUT/'started.json',OUT/'stdout.log',OUT/'stderr.log',OUT/'resources.jsonl',
                         CHILD_OUT/'result.json',CHILD_OUT/'completion.json'):
                if path.is_file():
                    result['artifacts'][str(path)] = sha(path)
            for path,digest in {**sources,**inputs}.items():
                require(sha(path) == digest,'Final source/input seal changed: '+path)
            result.update(hashes_unchanged=True,sources_unchanged=True)
            if result['passed']:
                item = observe(active=False)
                require(not item['child_pids'],'No descendants after final sealing')
                result['final_resource_observation'] = item
                require(sha(CHILD_OUT/'result.json') == result['child_result_sha256'] and
                        sha(CHILD_OUT/'completion.json') == result['child_completion_sha256'],'Child receipts remain immutable')
        except BaseException:
            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
                          final_sealing_error=traceback.format_exc())
        for sig,handler in previous.items():
            signal.signal(sig,handler)
        result.update(elapsed_s=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write(OUT/'result.json',result)
        write(OUT/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            hashes_unchanged=result['hashes_unchanged'],sources_unchanged=result['sources_unchanged'],
            result_sha256=sha(OUT/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s','child_scientific_passed',
        'maximum_sampled_rss_bytes','maximum_swap_growth_bytes','error','cleanup_error','final_sealing_error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
