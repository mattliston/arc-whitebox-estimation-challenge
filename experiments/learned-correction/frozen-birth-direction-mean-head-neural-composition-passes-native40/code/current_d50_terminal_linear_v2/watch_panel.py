"""Explicit, separate independent14 acquisition or score supervision.

Stdlib only. Native: one OFFPY process, 15 ordered complete calls, 900s/8GB
and 120s without a newly completed prediction bill. Score: 120s/2GB. Only
the fresh owned child group may be signalled. No automatic next stage,
refit, candidate change, numerical import, target access, or release claim.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
P2 = HERE.parent.parent
ROOT = HERE/'panel_v1'
OFFPY = P2/'data_gate/.venv/bin/python'
PROTOCOL_SHA = '7e2c4d86de5d600b19134fd3127d0878d05a527cab811ec36b2ebca4b3c69f2f'
OLD_PROTOCOL_SHA = 'febed4bdccc7c031cb846e6d97f695536ca98d3bef10d07de48f1418a91b6a58'
HELPER_SHA = 'b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd'
TERMINAL_SCORE_SHA = '9e327c7e7f8df5471679eb7532a4531b5a763ed6b051d32408e0231b6ceda030'
TERMINAL_END_SHA = '61765bd814261708ca6a32c2bc8dee4828d465e9bf80fa558a186c59b61d4e63'
TERMINAL_WATCH_SHA = '06a97b4215360af3446b1b166c78403ebf87d66ca0019ad6cd8eacaad6683e61'
TERMINAL_WATCH_END_SHA = 'fc36b81555a7b2dad7cdebd4e782bf7c7b949537b5b7ec1205e2de0fb462859e'
CANDIDATE_SHA = '03d78ee8182749e6c5bed07618420bd9b134b242236a079dddba14837c823641'
SCHEMA = 'current_d50_terminal_linear_independent14_v1'
WATCH_SCHEMA = 'current_d50_terminal_linear_independent14_watch_v1'
RANKS = (35,100,150,200,250,300,400,500,3016500,3016501,3016502,3016503,3016504,3016505)
CALLS = RANKS+(35,)
INITIALIZATION = 1_383_913_786
BUDGET = 2**41
SWAP_LIMIT, PERIOD, MAX_INTERVAL, PROGRESS_LIMIT = 512*1024**2,.2,1.0,120.0
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),
            'Supervisor never opens target arrays')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def bind(items):
    require(isinstance(items,dict),'Flat source/artifact bindings')
    for path,digest in items.items():
        require(isinstance(path,str) and isinstance(digest,str) and
                re.fullmatch('[0-9a-f]{64}',digest) and sha(path) == digest,
                'Immutable source/artifact: '+str(path))


def validate_native(record,end,digest,args):
    require(record['schema'] == SCHEMA and all(record.get(k) is True for k in
        ('passed','completed','hashes_unchanged','sources_unchanged','earlier15_rows_byte_exact',
         'own_complete_F_and_opcounts_exact','all_outputs_and_bills_sealed_before_scoring',
         'repeat_byte_and_bill_exact')) and end['passed'] is True and end['completed'] is True
        and end['result_sha256'] == digest,'Complete qualified independent14 acquisition')
    require(record['candidate_sha256'] == CANDIDATE_SHA and
            record['terminal_score_sha256'] == TERMINAL_SCORE_SHA and
            record['source_bindings'][str(HERE/'panel.py')] == args.worker_sha256 and
            record['source_bindings'][str(HERE/'PANEL_PROTOCOL.md')] == PROTOCOL_SHA,
            'Exact fixed candidate, panel source and prerequisite')
    require(record['truth_files_accessed'] is False and
            record['initialization_flops'] == INITIALIZATION and
            len(record['trials']) == len(record['input_preparation']) == 15 and
            len(record['artifacts']) == 75,'Target-free original fifteen five-artifact calls')
    for call,trial in enumerate(record['trials']):
        require(trial['call'] == call and trial['rank'] == CALLS[call] and
                trial['includes_initialization'] is (call == 0) and
                type(trial['flops']) is int and 0 < trial['flops'] <= BUDGET and
                all(math.isfinite(trial[k]) and trial[k] < 120 for k in ('wall_s','context_wall_s')) and
                trial['peak_rss_bytes'] < 8_000_000_000,'Ordered actual complete native call/resource bill')
    first,last = record['trials'][0],record['trials'][-1]
    require(first['flops']-last['flops'] == INITIALIZATION and
            first['output_bytes_sha256'] == last['output_bytes_sha256'],'Exact repeat and measured init once')
    bind(record['source_bindings']); bind(record['artifacts'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--stage',choices=('native','score'))
    for key in ('source','worker','protocol','terminal-score','acquisition','acquisition-watch'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    require(args.stage is not None,'Separate explicit native or score invocation')
    required = ('source','worker','protocol','terminal_score') + (
        ('acquisition','acquisition_watch') if args.stage == 'score' else ())
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '') for k in required),
            'Every reviewed source/receipt digest is required')
    require(args.stage == 'score' or (args.acquisition_sha256 is None and args.acquisition_watch_sha256 is None),
            'No score-only inputs in acquisition mode')
    require(args.protocol_sha256 == PROTOCOL_SHA and args.terminal_score_sha256 == TERMINAL_SCORE_SHA,
            'Only the prospectively fixed panel after the advancing terminal40 result')
    require(OFFPY.is_file() and os.getpriority(os.PRIO_PROCESS,0) >= 10 and
            all(os.environ.get(k) == '1' for k in BLAS) and
            os.environ.get('PYTHONDONTWRITEBYTECODE') == '1' and
            os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0','OFFPY/nice10/BLAS1/no-fallback environment')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')),'Stdlib-only watcher')
    worker = HERE/'panel.py'
    sources = {str(Path(__file__).resolve()):args.source_sha256,str(worker):args.worker_sha256,
        str(HERE/'PANEL_PROTOCOL.md'):PROTOCOL_SHA,str(HERE/'PROTOCOL.md'):OLD_PROTOCOL_SHA,
        str(HERE/'watch_fit.py'):HELPER_SHA,str(HERE/'fit_v1/estimator.py'):CANDIDATE_SHA}
    inputs = {str(HERE/'terminal_score_v1/result.json'):TERMINAL_SCORE_SHA,
        str(HERE/'terminal_score_v1/completion.json'):TERMINAL_END_SHA,
        str(HERE/'terminal_score_watch_v1/result.json'):TERMINAL_WATCH_SHA,
        str(HERE/'terminal_score_watch_v1/completion.json'):TERMINAL_WATCH_END_SHA}
    bind(sources); bind(inputs)
    terminal,terminal_end = read(HERE/'terminal_score_v1/result.json'),read(HERE/'terminal_score_v1/completion.json')
    old_watch,old_end = read(HERE/'terminal_score_watch_v1/result.json'),read(HERE/'terminal_score_watch_v1/completion.json')
    require(all(terminal.get(k) is True for k in ('passed','completed','integrity_passed','hashes_unchanged',
                'advances_to_independent14')) and terminal['candidate_sha256'] == CANDIDATE_SHA and
            terminal_end['passed'] is True and terminal_end['completed'] is True and
            terminal_end['result_sha256'] == TERMINAL_SCORE_SHA,'Sole completed advancing terminal40')
    require(all(old_watch.get(k) is True for k in ('passed','completed','hashes_unchanged',
                'child_execution_completed','whole_owned_tree_retired')) and old_watch['stage'] == 'score' and
            old_watch['child_result_sha256'] == TERMINAL_SCORE_SHA and
            old_watch['child_completion_sha256'] == TERMINAL_END_SHA and old_end['passed'] is True and
            old_end['completed'] is True and old_end['result_sha256'] == TERMINAL_WATCH_SHA,
            'Independent terminal40 resource supervision completed')
    if args.stage == 'score':
        folder,watch_folder = ROOT/'terminal',ROOT/'terminal_watch'
        for path,digest in ((folder/'result.json',args.acquisition_sha256),
                            (watch_folder/'result.json',args.acquisition_watch_sha256)):
            inputs[str(path)] = digest
        for path in (folder/'completion.json',watch_folder/'completion.json'):
            inputs[str(path)] = sha(path)
        bind(inputs)
        acquisition,end = read(folder/'result.json'),read(folder/'completion.json')
        validate_native(acquisition,end,args.acquisition_sha256,args)
        previous,previous_end = read(watch_folder/'result.json'),read(watch_folder/'completion.json')
        require(previous['schema'] == WATCH_SCHEMA and previous['stage'] == 'native' and
                all(previous.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
                    'child_execution_completed','whole_owned_tree_retired')) and
                previous['child_result_sha256'] == args.acquisition_sha256 and
                previous['child_completion_sha256'] == inputs[str(folder/'completion.json')] and
                previous['source_bindings'][str(Path(__file__).resolve())] == args.source_sha256 and
                previous['source_bindings'][str(worker)] == args.worker_sha256 and
                previous['completed_bill_count'] == 15 and previous_end['passed'] is True and
                previous_end['completed'] is True and previous_end['result_sha256'] == args.acquisition_watch_sha256,
                'Successful new native resource watch before any scoring')
        bind(previous['source_bindings']); bind(previous['input_bindings']); bind(previous['artifacts'])
    child_out = ROOT/('terminal' if args.stage == 'native' else 'score')
    out = ROOT/('terminal_watch' if args.stage == 'native' else 'score_watch')
    require(not ROOT.is_symlink() and not child_out.exists() and not out.exists(),'Exclusive nonsymlink stage paths')
    ROOT.mkdir(exist_ok=True); out.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('tl_panel_resource_helpers',HERE/'watch_fit.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    command = [str(OFFPY),'-B','-u',str(worker),'--acquire' if args.stage == 'native' else '--score',
        '--source-sha256',args.worker_sha256,'--protocol-sha256',PROTOCOL_SHA,
        '--terminal-score-sha256',TERMINAL_SCORE_SHA]
    if args.stage == 'score':
        command += ['--acquisition-sha256',args.acquisition_sha256,
                    '--acquisition-watch-sha256',args.acquisition_watch_sha256]
    wall_limit,rss_limit = (900.0,8_000_000_000) if args.stage == 'native' else (120.0,2_000_000_000)
    result = dict(schema=WATCH_SCHEMA,stage=args.stage,passed=False,completed=False,hashes_unchanged=False,
        sources_unchanged=False,source_bindings=sources,input_bindings=inputs,artifacts={},command=command,
        prerequisites=dict(terminal_score_sha256=TERMINAL_SCORE_SHA,terminal_score_watch_sha256=TERMINAL_WATCH_SHA,
            acquisition_sha256=args.acquisition_sha256,acquisition_watch_sha256=args.acquisition_watch_sha256),
        candidate_sha256=CANDIDATE_SHA,truth_arrays_opened=False,numerical_imports=False,
        child_execution_completed=False,whole_owned_tree_retired=False,operational_verdict_only=True,
        retain_local_component=False,production_qualified=False,official_score=False,
        wall_limit_s=wall_limit,combined_rss_limit_bytes=rss_limit,swap_growth_limit_bytes=SWAP_LIMIT,
        requested_sample_period_s=PERIOD,sample_interval_limit_s=MAX_INTERVAL,
        maximum_sample_interval_s=0.0,maximum_sampled_rss_bytes=0,maximum_swap_growth_bytes=0.0,samples=0,
        no_completed_bill_limit_s=PROGRESS_LIMIT if args.stage == 'native' else None,
        first_progress_elapsed_s=None,last_progress_elapsed_s=None,completed_bill_count=0,progress=[],
        limitations=['RSS and swap are sampled, not continuous peaks.',
            'The native progress bound includes initial admission and all between-call bookkeeping.',
            'The per-context120s guard remains in the unchanged native worker.',
            'Operational score completion does not imply scientific retention or production qualification.'],
        started_utc=datetime.now(timezone.utc).isoformat())
    child,before,handlers = None,None,{}
    started = last_progress = began
    last = time.monotonic()

    def interrupted(signum,frame):
        raise RuntimeError('Panel watcher interrupted: '+str(signum))

    def progress():
        nonlocal last_progress
        if args.stage != 'native':
            return
        found = []
        for path in child_out.glob('call_*_bill.json'):
            if path.name.endswith('_input_bill.json'):
                continue
            match = re.fullmatch(r'call_([0-9]{2})_bill\.json',path.name)
            require(match is not None and not path.is_symlink(),'Only exact regular prediction-bill paths')
            found.append(int(match[1]))
        require(sorted(found) == list(range(len(found))) and len(found) <= 15,'Unique ordered call00..14 bill names')
        for call in range(result['completed_bill_count'],len(found)):
            path = child_out/f'call_{call:02d}_bill.json'
            try:
                bill = read(path)
            except json.JSONDecodeError:
                break  # The one writer has not closed this newly created JSON yet.
            require(bill['call'] == call and bill['rank'] == CALLS[call] and
                    bill['includes_initialization'] is (call == 0) and
                    type(bill['flops']) is int and 0 < bill['flops'] <= BUDGET,
                    'Progress is only a complete next fixed prediction bill')
            observed = time.monotonic()
            require(observed-last_progress < PROGRESS_LIMIT,
                    'Next completed prediction bill was first observed after the120s progress deadline')
            last_progress = observed
            elapsed = last_progress-started
            if result['first_progress_elapsed_s'] is None:
                result['first_progress_elapsed_s'] = elapsed
            result['last_progress_elapsed_s'] = elapsed
            result['completed_bill_count'] = call+1
            result['progress'].append(dict(call=call,rank=CALLS[call],path=str(path),observed_elapsed_s=elapsed))

    def observe(stream=None,active=True):
        nonlocal last
        memory = helper.memory_snapshot(child.pid if child is not None else -1)
        sampled = time.monotonic(); interval,last = sampled-last,sampled
        swap = helper.swap_state(); growth = swap['used_bytes']-before['used_bytes']
        progress()
        now = time.monotonic()
        item = dict(elapsed_s=now-started,interval_s=interval,**memory,swap=swap,swap_growth_bytes=growth,
                    completed_bill_count=result['completed_bill_count'],no_completed_bill_s=now-last_progress,
                    live_child_interval_gate=active)
        result['samples'] += 1
        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],memory['combined_rss_bytes'])
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        if active:
            result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
            require(interval <= MAX_INTERVAL,'Live-child sampling interval<=1s')
        if stream is not None:
            stream.write(json.dumps(item,allow_nan=False)+'\n'); stream.flush()
        require(now-started < wall_limit and memory['combined_rss_bytes'] < rss_limit and growth <= SWAP_LIMIT,
                'Independent whole-phase wall/combined-RSS/swap guard')
        if active and args.stage == 'native':
            require(now-last_progress < PROGRESS_LIMIT,'Native120s without the next completed prediction bill')
        return item

    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig] = signal.signal(sig,interrupted)
        before = helper.swap_state(); result['before_swap'] = before
        write(out/'started.json',result)
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr,\
                (out/'resources.jsonl').open('x') as resources:
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
        require(result['child_exit_code'] == 0,'Native acquisition or score integrity failed')
        rp,cp = child_out/'result.json',child_out/'completion.json'
        record,end = read(rp),read(cp)
        result.update(child_result_sha256=sha(rp),child_completion_sha256=sha(cp))
        require(record['schema'] == SCHEMA and all(record.get(k) is True for k in
                    ('completed','hashes_unchanged','sources_unchanged')) and
                end['completed'] is True and end['result_sha256'] == result['child_result_sha256'] and
                end['passed'] is record['passed'],'Completed immutable child result')
        if args.stage == 'native':
            validate_native(record,end,result['child_result_sha256'],args)
            require(result['completed_bill_count'] == 15,'Every completed bill observed in fixed order')
            for item,trial in zip(result['progress'],record['trials']):
                require(read(item['path']) == trial,'Observed progress is the final sealed bill')
                item['sha256'] = sha(item['path'])
        else:
            require(record['mode'] == 'score' and record['qualification_passed'] is True and
                    record['integrity_passed'] is True and isinstance(record['retain_local_component'],bool) and
                    record['source_bindings'][str(worker)] == args.worker_sha256 and
                    record['source_bindings'][str(HERE/'PANEL_PROTOCOL.md')] == PROTOCOL_SHA,
                    'Complete fixed independent14 score integrity; scientific gain is separate')
            result['retain_local_component'] = record['retain_local_component']
        result.update(passed=True,completed=True,child_execution_completed=True,whole_owned_tree_retired=True)
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
                      retain_local_component=False,error=traceback.format_exc())
        if child is not None:
            try:
                helper.stop_owned_group(child); result['whole_owned_tree_retired'] = True
            except BaseException:
                result['cleanup_error'] = traceback.format_exc()
    finally:
        try:
            for path in (out/'started.json',out/'stdout.log',out/'stderr.log',out/'resources.jsonl',
                         child_out/'result.json',child_out/'completion.json'):
                if path.is_file():
                    result['artifacts'][str(path)] = sha(path)
            bind(sources); bind(inputs)
            result.update(hashes_unchanged=True,sources_unchanged=True)
            if result['passed']:
                item = observe(active=False)
                require(not item['child_pids'],'No descendants after final metadata verification')
                result['final_resource_observation'] = item
                require(sha(child_out/'result.json') == result['child_result_sha256'] and
                        sha(child_out/'completion.json') == result['child_completion_sha256'],
                        'Child completion remains immutable after final verification')
        except BaseException:
            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
                          retain_local_component=False,final_sealing_error=traceback.format_exc())
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
        result.update(elapsed_s=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/'result.json',result)
        write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            hashes_unchanged=result['hashes_unchanged'],sources_unchanged=result['sources_unchanged'],
            result_sha256=sha(out/'result.json'),retain_local_component=result['retain_local_component']))
    print(json.dumps({k:result.get(k) for k in ('stage','passed','completed','elapsed_s',
        'completed_bill_count','retain_local_component','maximum_sampled_rss_bytes','error','final_sealing_error')},
        allow_nan=False),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
