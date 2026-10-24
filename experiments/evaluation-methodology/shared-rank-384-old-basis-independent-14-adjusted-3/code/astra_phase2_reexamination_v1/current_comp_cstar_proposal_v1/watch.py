"""Separate bounded execution supervision; never certifies native gradients."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
MON = NEW/'current_p1_native_objective_p2_monitor_v2'
P1 = NEW/'current_d50_native_objective_p1_v1'
AP = '/Users/matt/anaconda3/bin/python'
SCHEMA = 'current_comp_cstar_proposal_watch_v1'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
    139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
    204,219,224,227,233,234)
TRAIN = tuple(r for r in range(35,235) if r not in VALID)
BLAS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
MON_PINS = {
    MON/'monitor.py':'d4b3af53b716593670d8e0c7f0fc6f2133b57f1088d9966c06635d3a23f39873',
    MON/'monitor_tests.py':'1438ca2f4b12b53628cfbb2be4f997eea8949bdf5a70a8c44c74738e83529fb9',
    MON/'PROTOCOL.md':'968910dc63ba1c090665322c945b4e7db3b2978b1a059baa10bab13ea2dc3af7',
    P1/'watch.py':'4e8177d6ff943cdee37e48ccca750bf9d552b32cec344aa2b37e6304cbd55fb3',
}
MOCK_PINS = {
    MON/'monitor_tests_v1/result.json':'c5a3f5b8a3dc99ca7e3550fb9ccb474b76968a49b96e00ed7e5606597476bf0e',
    MON/'monitor_tests_v1/completion.json':'bfb69013c603f0578a56c5e268e3654221b1ab50e00a8bd10159ef074b9e846f',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path, tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts and
        not (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),
        'Supervisor never opens target arrays')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size < 16*1024**2,
        'Bounded regular metadata')
    return json.loads(path.read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def check_roster(path, stage):
    value = read(path)
    require(value['schema'] == 'current_comp_cstar_proposal_roster_v1'
        and value['stage'] == stage and value['seed'] == 9309099,'Fixed proposal roster owner')
    if stage == 'qualify':
        expected = [dict(graph_index=i,rank=r,kind=k,update=None) for i,(r,k) in enumerate(
            ((35,'parent_forward'),(100,'parent_forward'),(35,'synthetic_backward'),
             (100,'synthetic_backward'),(35,'synthetic_backward')))]
    else:
        order = value['train_order']
        require(value['generator'] == 'PCG64DXSM' and len(order) == 160
            and all(type(r) is int for r in order) and set(order) == set(TRAIN)
            and value['validation_order'] == list(VALID),'One complete fixed TRAIN order and original VALID40')
        expected = [dict(graph_index=i,rank=r,kind='train_backward',update=i//4) for i,r in enumerate(order)]
        expected += [dict(graph_index=160+i,rank=r,kind='validation_forward',update=None) for i,r in enumerate(VALID)]
    require(value['entries'] == expected,'Exact ordered model-call identities, not count alone')
    return expected


def transport():
    for path,digest in {**MON_PINS,**MOCK_PINS}.items():
        require(sha(path) == digest,'Exact qualified monitor source/evidence')
    record,end = read(MON/'monitor_tests_v1/result.json'),read(MON/'monitor_tests_v1/completion.json')
    require(all(record[k] is True for k in ('passed','completed','hashes_unchanged','sources_unchanged'))
        and end['passed'] and end['completed'] and end['result_sha256'] == MOCK_PINS[MON/'monitor_tests_v1/result.json']
        and not any(record[k] for k in ('numerical_imports','arrays_accessed','targets_accessed','model_accessed'))
        and record['real_subprocesses_started'] == record['real_signals_sent'] == 0,
        'Only the passed target-free command transport certificate')
    spec = importlib.util.spec_from_file_location('cprop_owned_monitor',MON/'monitor.py')
    mon = importlib.util.module_from_spec(spec); spec.loader.exec_module(mon)
    nodes = [x for x in ast.parse((P1/'watch.py').read_text()).body
        if isinstance(x,ast.FunctionDef) and x.name == 'other_workers']
    require(len(nodes) == 1,'Exact original process classifier')
    env = dict(Path=Path,re=re,shlex=shlex,P2=P2,NEW=NEW)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(P1/'watch.py')+':classifier-only','exec'),env)
    return mon,env['other_workers']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--stage',choices=('qualify','train'))
    parser.add_argument('--mps-slot-clear',action='store_true')
    for key in ('source','worker','common','model','protocol','watch-spec',
                'qualification','qualification-watch'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = previous = last_progress = time.monotonic()
    require(args.stage is not None and args.mps_slot_clear,'Explicit stage and free sole numerical slot')
    limit,expected = (300.,5) if args.stage == 'qualify' else (1800.,200)
    require(os.getpriority(os.PRIO_PROCESS,0) >= 10 and sys.flags.optimize == 0
        and sys.dont_write_bytecode and all(os.environ.get(k) == '1' for k in BLAS),
        'nice10, assertions, bytecodeoff and BLAS1 supervisor')
    own = {HERE/'watch.py':args.source_sha256,HERE/'worker.py':args.worker_sha256,
        HERE/'common.py':args.common_sha256,HERE/'model.py':args.model_sha256,
        HERE/'PROTOCOL.md':args.protocol_sha256,HERE/'WATCH_SPEC.md':args.watch_spec_sha256}
    for path,digest in own.items():
        require(re.fullmatch('[0-9a-f]{64}',digest or '') and sha(path) == digest,'Explicit reviewed source '+str(path))
    require(bool(args.qualification_sha256) == bool(args.qualification_watch_sha256) == (args.stage == 'train'),
        'Only training receives new qualification authorities')
    mon,classifier = transport()
    folder = HERE/('qualification_v1' if args.stage == 'qualify' else 'training_v1')
    out = HERE/('qualification_watch_v1' if args.stage == 'qualify' else 'training_watch_v1')
    require(not folder.exists() and not out.exists() and not HERE.is_symlink(),'Exclusive fresh worker/watch paths')
    out.mkdir(exist_ok=False)
    result = dict(schema=SCHEMA,stage=args.stage,passed=False,completed=False,
        source_bindings={str(p):h for p,h in {**own,**MON_PINS}.items()},
        input_bindings={str(p):h for p,h in MOCK_PINS.items()},artifacts={},
        native_gradient_qualified=False,no_native_gradient_claim=True,native_score_measured=False,
        numerical_imports=False,mps_slot_clear_attested=True,whole_cap_s=limit,
        maximum_combined_rss_bytes=0,maximum_interval_s=0.,maximum_swap_growth_bytes=0.,
        retired_calls=0,completed_graph_count=0,completed_sample_intervals_s=[],
        graph_markers=[],whole_owned_tree_retired=False,sources_unchanged=False,
        hashes_unchanged=False,child_execution_completed=False,
        monitor_mock_result_sha256=MOCK_PINS[MON/'monitor_tests_v1/result.json'])
    child=log=reader=stream=None
    initial_swap=None; pending=''; samples=0; roster=None; roster_digest=None
    def deadline():
        now = time.monotonic()
        require(now-began < limit and now-last_progress < 120.,'Original whole/progress cap through final I/O')
    def interrupted(*unused):
        raise RuntimeError('Owned proposal stage interrupted')
    def observe():
        nonlocal previous,last_progress,initial_swap,pending,samples,roster,roster_digest
        absolute = min(previous+1.,began+limit,last_progress+120.)
        sample = mon.collect(absolute)
        scope = mon.scope(sample['rows'],os.getpid(),None if child is None else child.pid)
        require(not classifier(sample['rows'],scope.pop('allowed')),'No competing numerical worker')
        if initial_swap is None:
            initial_swap = sample['swap']['used_bytes']
        growth = sample['swap']['used_bytes']-initial_swap
        require(growth <= 0 and scope['combined_rss_bytes'] < 12_000_000_000,'Zero added swap/combined host12GB')
        if reader is not None:
            pending += reader.read()
            require(len(pending) < 4*1024**2,'Bounded worker log tail')
            while '\n' in pending:
                line,pending = pending.split('\n',1)
                try:
                    item = json.loads(line)
                except (ValueError,TypeError):
                    continue
                if isinstance(item,dict) and item.get('event') == 'retired':
                    deadline()
                    index = result['retired_calls']
                    require(item['call'] == index and index < expected,'Ordered complete model-call marker')
                    if roster is None:
                        roster_path = folder/'model_call_roster.json'
                        roster_digest = sha(roster_path)
                        roster = check_roster(roster_path,args.stage)
                        result['model_call_roster_sha256'] = roster_digest
                    path = folder/f'graph_{index:04d}_complete.json'
                    record = read(path); digest = sha(path)
                    require(digest == item['complete_sha256'] and record['graph_index'] == index
                        and record['rank'] == item['rank'] and record['kind'] == item['kind']
                        and all(record.get(k) is True for k in ('passed','completed','graph_retired')),
                        'Actual released model-call record, not heartbeat progress')
                    require(all(record[k] == v for k,v in roster[index].items()),
                        'Every observed completion matches its prospectively fixed roster')
                    for key,cap in (('mps_driver_bytes',16_000_000_000),('host_peak_rss_bytes',12_000_000_000)):
                        value = item[key]
                        require(isinstance(value,(int,float)) and math.isfinite(value) and 0 <= value < cap,
                            'Retired call reports actual bounded '+key)
                    result['graph_markers'].append(dict(call=index,path=str(path),sha256=digest,
                        rank=item['rank'],kind=item['kind'],observed_elapsed_s=time.monotonic()-began))
                    result['retired_calls'] += 1
                    result['completed_graph_count'] = result['retired_calls']
                    last_progress = time.monotonic()
        deadline()
        result['maximum_combined_rss_bytes'] = max(result['maximum_combined_rss_bytes'],scope['combined_rss_bytes'])
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        item = dict(elapsed_s=time.monotonic()-began,**scope,swap_bytes=sample['swap']['used_bytes'],
            retired_calls=result['retired_calls'],observational_groups_retired=sample['commands_reaped'],
            transport=sample['command_records'],uncached_single_process_table=True)
        stream.write(json.dumps(item,sort_keys=True)+'\n'); stream.flush()
        previous,interval = mon.complete(previous,absolute)
        result['completed_sample_intervals_s'].append(interval)
        result['maximum_interval_s'] = max(result['maximum_interval_s'],interval)
        samples += 1
    def tick():
        deadline()
        if time.monotonic()-previous >= .2:
            observe()
    old_handlers = {s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        stream = (out/'samples.jsonl').open('x')
        observe()
        child_env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_LOW_WATERMARK_RATIO='0.1')
        child_env.update({k:'1' for k in BLAS})
        argv = [AP,'-B','-u',str(HERE/'worker.py'),'--run','--stage',args.stage,
            '--source-sha256',args.worker_sha256,'--common-sha256',args.common_sha256,
            '--model-sha256',args.model_sha256,'--protocol-sha256',args.protocol_sha256,
            '--watcher-sha256',args.source_sha256,'--watch-spec-sha256',args.watch_spec_sha256]
        if args.stage == 'train':
            argv += ['--qualification-sha256',args.qualification_sha256,
                '--qualification-watch-sha256',args.qualification_watch_sha256]
        log = (out/'worker.log').open('x'); reader = (out/'worker.log').open('r')
        child = subprocess.Popen(argv,cwd=NEW,env=child_env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result['child_pid'],result['command'] = child.pid,argv
        write(out/'started.json',result)
        while child.poll() is None:
            observe(); time.sleep(.15)
        observe()
        require(child.wait(timeout=1) == 0,'Worker exit')
        mon.cleanup([child],time.monotonic()+.5)
        files = [folder/name for name in ('result.json','completion.json','finalization.json')]
        digests = [sha(p,tick) for p in files]
        record,end,final = map(read,files)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (record,end,final))
            and end['result_sha256'] == final['result_sha256'] == digests[0]
            and final['completion_sha256'] == digests[1] and final['elapsed_s'] < limit,
            'All three new worker seals, not old failed qualification')
        require(record['stage'] == args.stage and record['graphs_completed'] == expected
            and result['retired_calls'] == expected and record['native_gradient_qualified'] is False
            and record['no_native_gradient_claim'] and record['validation_targets_accessed'] is False,
            'Exact new execution/proposal scope')
        require(record['execution_integrity_passed'] if args.stage == 'qualify' else record['proposal_complete'],
            'Entire selected stage completed')
        require(roster is not None and sha(folder/'model_call_roster.json',tick) == roster_digest
            and check_roster(folder/'model_call_roster.json',args.stage) == roster,
            'Unchanged roster through completed stage and post-retirement verification')
        union = {}
        for group in ('source_bindings','input_bindings','weight_bindings','artifacts'):
            require(isinstance(record[group],dict) and record[group],'Complete child inventory: '+group)
            for path,digest in record[group].items():
                require(path not in union or union[path] == digest,'Consistent child immutable digest')
                union[path] = digest
        require(all(record['source_bindings'].get(str(p)) == h for p,h in own.items()),'Same reviewed six owners')
        for path,digest in union.items():
            require(sha(path,tick) == digest,'Independent fresh child nontarget/artifact hash: '+path)
        for path,digest in result['source_bindings'].items():
            require(sha(path,tick) == digest,'Supervisor sources unchanged')
        require([sha(p,tick) for p in files] == digests,'Child seals immutable through verification')
        result.update(passed=True,completed=True,child_execution_completed=True,whole_owned_tree_retired=True,
            hashes_unchanged=True,sources_unchanged=True,child_result_sha256=digests[0],
            child_completion_sha256=digests[1],child_finalization_sha256=digests[2],
            verified_child_non_target_files=len(union),actual_completed_samples=True,
            foreign_numerical_workers_checked=True)
        observe()
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        try:
            if child is not None:
                mon.cleanup([child],time.monotonic()+.5)
            mon.retire_pending(time.monotonic()+.5)
            result.update(cleanup_passed=True,whole_owned_tree_retired=True,observational_groups_retired=True)
        except BaseException:
            result.update(passed=False,completed=False,cleanup_error=traceback.format_exc())
        for handle in (log,reader,stream):
            if handle is not None:
                handle.close()
        result.update(elapsed_s=time.monotonic()-began,observation_samples=samples,initial_swap_bytes=initial_swap)
        if result['passed']:
            try:
                deadline()
            except BaseException:
                result.update(passed=False,completed=False,error=traceback.format_exc())
        for name in ('started.json','worker.log','samples.jsonl'):
            path = out/name
            if path.exists():
                result['artifacts'][str(path)] = sha(path)
        write(out/'result.json',result)
        result_sha = sha(out/'result.json')
        ok = bool(result['passed'] and time.monotonic()-began < limit and time.monotonic()-last_progress < 120.)
        write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=result_sha))
        completion_sha = sha(out/'completion.json')
        elapsed = time.monotonic()-began
        ok = bool(ok and elapsed < limit and time.monotonic()-last_progress < 120.)
        write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=result_sha,
            completion_sha256=completion_sha,elapsed_s=elapsed))
        sha(out/'finalization.json')
        ok = bool(ok and time.monotonic()-began < limit and time.monotonic()-last_progress < 120.)
        for sig,handler in old_handlers.items():
            signal.signal(sig,handler)
    print(json.dumps(dict(stage=args.stage,passed=ok,retired_calls=result['retired_calls'],
        elapsed_s=time.monotonic()-began,error=result.get('error')),allow_nan=False),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
