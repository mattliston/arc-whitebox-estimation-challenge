"""Inert five-call, target-free TREE supervisor; all calls are fresh and cold.

Requires completed TL validation (not a positive scientific result), the tiny
algebra/watch seals, and an explicit exclusive-slot launch attestation. No fit,
scoring, target access, numerical import, reuse of partial runs, or retry.
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
NEW, P2 = HERE.parent,HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
HELPER = NEW/'current_d50_return_quantization_v1/watch.py'
HELPER_SHA = 'f79749448c5a6015540474b18b58b6981944fc8b8c9182200effc0095728e32b'
TINY_WATCH_SHA = 'f3e07387b979d039eb36b5a3b2269cda05b60f80f0f9ce0983602296ff368b76'
TL = NEW/'current_d50_terminal_linear_v2'
TL_JUDGE_SHA = 'd755605c7a1773cf3fe740ee7a494435fc67efeb7377fc0f442b9e00f9277c9e'
TL_PROTOCOL_SHA = 'febed4bdccc7c031cb846e6d97f695536ca98d3bef10d07de48f1418a91b6a58'
PROTOCOL_SHA = 'f4bf2907fa1973b6076d6f485cd2436d6b5bf13cfe26077653e52af3c3e59dad'
OUT, CALLS = HERE/'native_watch_v1', HERE/'native_v1'
RSS_LIMIT, CHILD_WALL, PHASE_WALL = 8_000_000_000,120.0,900.0
SWAP_LIMIT, PERIOD, MAX_INTERVAL = 512*1024**2,.2,1.0
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),'No target-array access')
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


def module(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def pin_inventory(record,pins,groups=('source_bindings','input_bindings','artifacts')):
    for group in groups:
        for path,item in record.get(group,{}).items():
            digest = item['sha256'] if isinstance(item,dict) else item
            path = item['path'] if isinstance(item,dict) else path
            require(sha(path) == digest,'Sealed file changed: '+str(path))
            require(str(path) not in pins or pins[str(path)] == digest,'Conflicting sealed file')
            pins[str(path)] = digest


def sealed(folder,digest,pins,build=False):
    path = folder/('build.json' if build else 'result.json')
    require(sha(path) == digest,'Explicit immutable receipt')
    record,end = read(path),read(folder/'completion.json')
    require(end['completed'] is True and end['result_sha256'] == digest,'Completed receipt seal')
    pins[str(path)],pins[str(folder/'completion.json')] = digest,sha(folder/'completion.json')
    return record,end


def authenticate(args):
    require(args.protocol_sha256 == PROTOCOL_SHA and args.terminal_validation_finished
            and args.exclusive_numerical_slot,'Fixed protocol and explicit launch attestations')
    pins = {str(Path(__file__).resolve()):args.source_sha256,str(HERE/'native.py'):args.native_sha256,
            str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,str(HELPER):HELPER_SHA,
            str(HERE/'watch_tiny.py'):TINY_WATCH_SHA,str(TL/'judge_terminal.py'):TL_JUDGE_SHA,
            str(TL/'PROTOCOL.md'):TL_PROTOCOL_SHA}
    for path,digest in pins.items():
        require(sha(path) == digest,'Reviewed source: '+path)
    worker = module(HERE/'native.py','terminal_tree_worker_metadata')
    bound,prerequisites = worker.admission(args.build_result_sha256,args.tiny_result_sha256)
    for path,digest in bound.items():
        require(path not in pins or pins[path] == digest,'Source admission conflict')
        pins[path] = digest
    score,end = sealed(TL/'terminal_score_v1',args.terminal_score_result_sha256,pins)
    require(score['schema'] == 'current_d50_terminal_linear_native40_score_v1' and
            all(score.get(k) is True for k in ('completed','integrity_passed','hashes_unchanged',
                'all_predictions_verified_before_truth','all_earlier15_outputs_byte_exact',
                'all_own_complete_bills_verified')) and len(score['records']) == 40 and
            end['passed'] is score['passed'] and
            score['source_bindings'][str(TL/'judge_terminal.py')] == TL_JUDGE_SHA and
            score['source_bindings'][str(TL/'PROTOCOL.md')] == TL_PROTOCOL_SHA,
            'TL fixed fit/native40/scoring completed; advancement is not required')
    # Read only the pinned completed score JSON, never its targets/predictions.
    tiny_watch,end = sealed(HERE/'tiny_watch_v1',args.tiny_watch_result_sha256,pins)
    require(tiny_watch['schema'] == 'terminal_full_fourth_tree_tiny_watch_v1' and
            all(tiny_watch.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
                'child_execution_completed','whole_owned_tree_retired')) and end['passed'] is True and
            tiny_watch['source_bindings'][str(HERE/'watch_tiny.py')] == TINY_WATCH_SHA and
            tiny_watch['child_result_sha256'] == args.tiny_result_sha256 and
            tiny_watch['child_completion_sha256'] == sha(HERE/'tiny_v1/completion.json') and
            tiny_watch['prerequisites']['terminal_score_sha256'] == args.terminal_score_result_sha256 and
            tiny_watch['prerequisites']['build_sha256'] == args.build_result_sha256,
            'Same completed tiny and predecessor validation under independent resource watch')
    require(tiny_watch['elapsed_s'] < 30 and tiny_watch['maximum_sampled_rss_bytes'] < 1_000_000_000 and
            tiny_watch['maximum_swap_growth_bytes'] <= SWAP_LIMIT,'Tiny external limits')
    pin_inventory(tiny_watch,pins)
    require(not any(x in sys.modules for x in ('numpy','torch','flopscope')),'Stdlib-only supervisor')
    return worker,pins


def verify_worker(worker,args,role,rank,folder,pins):
    path = folder/'result.json'
    digest = sha(path)
    record,end = sealed(folder,digest,pins)
    require(record['schema'] == 'terminal_full_fourth_tree_native_v1' and
            all(record.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged')) and
            end['passed'] is True and (record['role'],record['rank']) == (role,rank) and
            record['candidate_sha256'] == worker.CANDIDATES[role][1] and
            record['source_bindings'][str(HERE/'native.py')] == args.native_sha256 and
            record['prerequisites'] == dict(build_result_sha256=args.build_result_sha256,
                                            tiny_result_sha256=args.tiny_result_sha256),
            'Exact successful current role/candidate/source/prerequisite')
    require(record['truth_accessed'] is False and record['MPS_used'] is False and
            record['head_replay_performed'] is False and
            record['parameter_updates'] == record['gradient_calls'] == 0 and
            record['network_calls'] == record['observed_predict_calls'] == 1 and
            record['observed_gnn_calls'] == 16 and record['original_P_unchanged'] and
            record['captured_parameter_bytes_exact'] and
            record['tree_source_calls'] == record['tree_readout_calls'] == int(role == 'enabled'),
            'Target-free sole cold call and natural source ownership')
    require(record['own_complete_bill'] and record['all_primary_cold'] and record['includes_initialization'] and
            record['initialization_subtracted'] is False and record['input_preparation_in_participant_bill'] is False and
            record['served_reconstruction_byte_exact'] and record['all16_domains_present'] and
            record['all47_selectors_present'] and all(record['archived_cold_reference'].values()),
            'Cold accounting, state and exact required reference rows')
    proof = record['source_adapter']
    require(proof['original_native_source_sha256'] == worker.TL_NATIVE_SHA and proof['exact_inverse'] and
            proof['paid_blocks_and_reset_spans_unchanged'] and proof['head_replay_removed'] and
            all(record['ownership_proof'][k] for k in ('paid_blocks_ast_identical',
                'original_numeric_prefix_unchanged','reset_order_ast_identical')),'Unchanged paid/reset blocks')
    bill = record['complete_bill']
    require(bill['cold'] and bill['includes_initialization'] and type(bill['flops']) is int and
            0 < bill['flops'] <= 2**41 and bill['operation_count'] > 0 and bill['observed_predict_calls'] == 1 and
            0 < bill['peak_rss_bytes'] < RSS_LIMIT and record['elapsed_s'] < CHILD_WALL,
            'Actual complete cold cost/resources')
    for key in ('wall_s','context_wall_s','flopscope_backend_s','flopscope_overhead_s','native_residual_s'):
        require(math.isfinite(bill[key]) and 0 <= bill[key] < CHILD_WALL,'Finite timing: '+key)
    require(read(record['bill']['path']) == bill,'Saved complete bill content')
    for key in ('input_lifecycle','lifecycle'):
        life = record[key]
        require(life['complete'] and life['context_released'] and life['after_context_exit'] and
                life['closed_context_bill_unchanged_by_reset'] and life['candidate_state_reset'] is False,
                'Original post-bill context release')
    prep = record['input_preparation']
    require(prep['flops'] == 16*1024*1024 and prep['charged_to_participant'] is False and
            all(prep[k] for k in ('official_fnp_type','exact_bytes','independent_weight_buffers','context_closed')),
            'Separate official-owned FP32 inputs')
    diag = record['diagnostics']
    require(diag['variance_count'] == 16 and diag['selector_count'] == 47 and
            diag['added_participant_array_operations'] == 0 and
            all(x['finite'] and math.isfinite(x['minimum']) and x['minimum'] >= 1e-10 and
                x['below_floor_count'] == 0 for x in diag['variance_domain'].values()),'Unfloored complete state')
    require(set(record['captures']) == {'host','hidden','sigma','delta','raw','served'},'Fixed capture roster')
    expected_files = {'input_bill.json','input_lifecycle.json','bill.json','lifecycle.json',
                      'host.npy','hidden.npy','sigma.npy','delta.npy','raw.npy','served.npy'}
    require(set(record['artifacts']) == {str(folder/name) for name in expected_files},'Exact capture/accounting files')
    pin_inventory(record,pins,('bindings','source_bindings','artifacts'))
    for item in record['captures'].values():
        require(sha(item['path']) == item['sha256'],'Capture sealed before any scoring')
    item = dict(role=role,rank=rank,path=str(path),sha256=digest)
    return item,record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--terminal-validation-finished',action='store_true')
    parser.add_argument('--exclusive-numerical-slot',action='store_true')
    for key in ('source','native','protocol','build-result','tiny-result','tiny-watch-result','terminal-score-result'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    for key,value in vars(args).items():
        if key.endswith('_sha256'):
            require(re.fullmatch('[0-9a-f]{64}',value or '') is not None,'Explicit reviewed hash: '+key)
    started = time.monotonic()
    worker,pins = authenticate(args)
    helper = module(HELPER,'terminal_tree_resource_helpers')
    require(OFFPY.is_file() and os.getpriority(os.PRIO_PROCESS,0) >= 10,'OFFPY and inherited nice10')
    environment = {k:'1' for k in BLAS}
    environment.update(PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
    require(all(os.environ.get(k) == v for k,v in environment.items()),'Explicit BLAS1/no-bytecode/no-MPS environment')
    require(not CALLS.exists(),'No resumed or blended native attempts')
    OUT.mkdir(exist_ok=False)
    CALLS.mkdir(exist_ok=False)
    result = dict(schema='terminal_full_fourth_tree_native_watch_v1',passed=False,completed=False,
        hashes_unchanged=False,sources_unchanged=False,source_bindings=dict(pins),artifacts={},calls=[],
        prerequisites=dict(build_result_sha256=args.build_result_sha256,tiny_result_sha256=args.tiny_result_sha256,
            tiny_watch_result_sha256=args.tiny_watch_result_sha256,terminal_score_result_sha256=args.terminal_score_result_sha256),
        truth_accessed=False,score_measured=False,scientific_advancement=False,capacity=1,
        launch_attestations=dict(terminal_validation_finished=True,exclusive_numerical_slot=True),
        roster=[dict(role=r,rank=n,label=l) for r,n,l in worker.ROSTER],
        maximum_sampled_rss_bytes=0,maximum_swap_growth_bytes=0.0,maximum_sample_interval_s=0.0,samples=0,
        child_wall_limit_s=CHILD_WALL,phase_wall_limit_s=PHASE_WALL,combined_rss_limit_bytes=RSS_LIMIT,
        swap_growth_limit_bytes=SWAP_LIMIT,requested_sample_period_s=PERIOD,sample_interval_limit_s=MAX_INTERVAL,
        limitations=['RSS/swap are sampled, not continuous peaks.',
            'Sampling interval bound applies to active numerical-child lifetimes; metadata-only sealing is separately bounded.',
            'PASS seals a target-free acquisition, not scientific advancement.'],
        started_utc=datetime.now(timezone.utc).isoformat())
    child,handlers,checked,log_paths = None,{},[],[]
    def interrupted(signum,frame):
        raise RuntimeError('Supervisor interrupted: '+str(signum))
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig] = signal.signal(sig,interrupted)
        before = helper.swap_state()
        result['before_swap'] = before
        write(OUT/'started.json',result)
        with (OUT/'resources.jsonl').open('x') as resources:
            for role,rank,label in worker.ROSTER:
                require(time.monotonic()-started < PHASE_WALL,'Whole phase900s')
                folder = CALLS/label
                command = [str(OFFPY),'-B','-u',str(HERE/'native.py'),'--run','--role',role,'--rank',str(rank),
                    '--output',str(folder),'--source-sha256',args.native_sha256,'--protocol-sha256',PROTOCOL_SHA,
                    '--build-result-sha256',args.build_result_sha256,'--tiny-result-sha256',args.tiny_result_sha256]
                stdout_path,stderr_path = OUT/(label+'.stdout.log'),OUT/(label+'.stderr.log')
                log_paths.extend((stdout_path,stderr_path))
                with stdout_path.open('xb') as stdout,stderr_path.open('xb') as stderr:
                    call_started = last = time.monotonic()
                    child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                        env=os.environ.copy(),start_new_session=True)
                    attempted = dict(role=role,rank=rank,label=label,pid=child.pid,command=command)
                    result['calls'].append(attempted)
                    while True:
                        code = child.poll()
                        memory = helper.memory_snapshot(child.pid)
                        sampled = time.monotonic()
                        interval,last = sampled-last,sampled
                        swap = helper.swap_state()
                        now = time.monotonic()
                        growth = swap['used_bytes']-before['used_bytes']
                        sample = dict(label=label,elapsed_s=now-started,child_elapsed_s=now-call_started,
                            interval_s=interval,**memory,swap=swap,swap_growth_bytes=growth)
                        resources.write(json.dumps(sample,allow_nan=False)+'\n');resources.flush()
                        result['samples'] += 1
                        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],memory['combined_rss_bytes'])
                        result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
                        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
                        require(now-started < PHASE_WALL and now-call_started < CHILD_WALL,'Independent phase/child wall')
                        require(interval <= MAX_INTERVAL and memory['combined_rss_bytes'] < RSS_LIMIT and growth <= SWAP_LIMIT,
                                'Sampled interval/combined8GB/swap guard')
                        require(memory['child_pids'] or code is not None or child.poll() is not None,'Live child visible')
                        if code is not None:
                            attempted.update(exit_code=code,elapsed_s=now-call_started)
                            require(not memory['child_pids'],'No surviving owned descendants')
                            require(code == 0,'Cold worker failed; preserve attempt and stop')
                            break
                        time.sleep(PERIOD)
                item,record = verify_worker(worker,args,role,rank,folder,pins)
                attempted['result'] = item
                checked.append(record)
                child = None
                print(json.dumps(dict(role=role,rank=rank,completed=len(checked),flops=record['complete_bill']['flops'])),flush=True)
        require(len(checked) == 5,'Exact fixed five-call completion')
        for key in ('host','hidden','sigma','delta','raw','served'):
            require(checked[2]['captures'][key]['raw_sha256'] == checked[4]['captures'][key]['raw_sha256'],
                    'Enabled fresh-cold35 repeat bytes: '+key)
        require(checked[2]['complete_bill']['flops'] == checked[4]['complete_bill']['flops'] and
                checked[2]['complete_bill']['operation_count'] == checked[4]['complete_bill']['operation_count'] and
                checked[2]['diagnostics'] == checked[4]['diagnostics'],'Enabled repeat cost/state identity')
        for path,digest in pins.items():
            require(sha(path) == digest,'Final source/input/output hash changed: '+path)
        seal = dict(truth_accessed=False,all_predictions_and_bills_sealed=True,
            calls=[x['result'] for x in result['calls']],candidates={k:dict(path=str(v[0]),sha256=v[1])
            for k,v in worker.CANDIDATES.items()},bindings=pins,complete_cold_bills=[x['complete_bill'] for x in checked],
            repeat_byte_and_bill_exact=True,earlier15_rows_exact=True,zero_all16_bytes_and_bill_exact=True)
        write(OUT/'predictions_and_bills_sealed.json',seal)
        result['predictions_and_bills_seal_sha256'] = sha(OUT/'predictions_and_bills_sealed.json')
        result.update(passed=True,completed=True,all5_complete=True,all_predictions_and_bills_sealed_before_scoring=True,
                      zero_all16_bytes_and_bill_exact=True,earlier15_rows_byte_exact=True,
                      repeat_byte_and_bill_exact=True,whole_owned_tree_retired=True)
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
            for path in (OUT/'started.json',OUT/'resources.jsonl',OUT/'predictions_and_bills_sealed.json',*log_paths):
                if path.is_file():
                    result['artifacts'][str(path)] = sha(path)
            for path,digest in pins.items():
                require(sha(path) == digest,'Final immutable binding changed: '+path)
            result['bindings'] = pins
            result.update(hashes_unchanged=True,sources_unchanged=True)
            memory,swap = helper.memory_snapshot(-1),helper.swap_state()
            result.update(final_resource_observation=memory,after_swap=swap)
            result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],memory['combined_rss_bytes'])
            result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],swap['used_bytes']-before['used_bytes'])
            require(not memory['child_pids'] and memory['combined_rss_bytes'] < RSS_LIMIT and
                    result['maximum_swap_growth_bytes'] <= SWAP_LIMIT and time.monotonic()-started < PHASE_WALL,
                    'Final phase resources and retired groups')
        except BaseException:
            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
                          final_sealing_error=traceback.format_exc())
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
        result.update(elapsed_s=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write(OUT/'result.json',result)
        write(OUT/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            hashes_unchanged=result['hashes_unchanged'],sources_unchanged=result['sources_unchanged'],
            result_sha256=sha(OUT/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s','maximum_sampled_rss_bytes',
        'maximum_swap_growth_bytes','error','cleanup_error','final_sealing_error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
