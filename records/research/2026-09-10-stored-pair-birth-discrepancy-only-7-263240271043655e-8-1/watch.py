"""One unchanged COMP birth-gauge size call; independent full-tree watch."""
import argparse
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
OFFPY = P2/'data_gate/.venv/bin/python'
TRANSPORT = NEW/'current_comp_cstar_proposal_v1/watch.py'
TRANSPORT_SHA = 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
CANDIDATE = NEW/'current_p1_node_composition_v1/build_v1/terminal_estimator.py'
CANDIDATE_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
BLAS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
CAP, RSS = 180., 8_000_000_000
DIAGNOSTIC_CAP = 2_044_723_200


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def transport():
    require(hashlib.sha256(TRANSPORT.read_bytes()).hexdigest() == TRANSPORT_SHA,
            'Frozen qualified target-free observation transport')
    spec = importlib.util.spec_from_file_location('gauge_size_transport',TRANSPORT)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    monitor,classifier = helper.transport()
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),
            'Supervisor is stdlib-only')
    return helper,monitor,classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--numerical-slot-clear',action='store_true')
    keys = ('source','worker','gauge-source','protocol','spec')
    for key in keys:
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = previous = time.monotonic()
    require(args.numerical_slot_clear and os.getpriority(os.PRIO_PROCESS,0) >= 10
        and sys.flags.optimize == 0 and sys.dont_write_bytecode
        and all(os.environ.get(k) == '1' for k in BLAS),
        'Exclusive nice10/BLAS1/bytecodeoff/assertions')
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k.replace('-','_')+'_sha256') or '')
        for k in keys),'Explicit reviewed source owners')
    h,mon,classifier = transport()
    own = {Path(__file__).resolve():args.source_sha256,HERE/'worker.py':args.worker_sha256,
        HERE/'gauge_fnp.py':args.gauge_source_sha256,HERE/'PROTOCOL.md':args.protocol_sha256,
        HERE/'WATCH_SPEC.md':args.spec_sha256,TRANSPORT:TRANSPORT_SHA,
        CANDIDATE:CANDIDATE_SHA,**h.MON_PINS}
    inputs = dict(h.MOCK_PINS)
    folder,out = HERE/'native_v1',HERE/'watch_v1'
    require(not folder.exists() and not out.exists() and not HERE.is_symlink(),
            'Exclusive one-call diagnostic, no overwrite or resume')
    out.mkdir(exist_ok=False)
    result = dict(schema='current_comp_birth_gauge_size_watch_v1',passed=False,completed=False,
        source_bindings={str(p):v for p,v in own.items()},
        input_bindings={str(p):v for p,v in inputs.items()},artifacts={},
        numerical_imports=False,truth_arrays_accessed=False,scientific_score_measured=False,
        full_model_calls=0,retired_call_count=0,candidate_sha256=CANDIDATE_SHA,
        maximum_combined_rss_bytes=0,maximum_interval_s=0.,maximum_swap_growth_bytes=0.,
        completed_sample_intervals_s=[],sources_unchanged=False,hashes_unchanged=False,
        whole_owned_tree_retired=False,child_execution_completed=False,
        whole_cap_s=CAP,combined_rss_limit_bytes=RSS,numerical_slot_clear_attested=True,
        uploaded=False,automatic_next_stage=False)
    child=log=stream=None;swap0=None;samples=0

    def deadline():
        require(time.monotonic()-began < CAP,'Whole180s through admission/reap/final publication')

    def observe():
        nonlocal previous,swap0,samples
        absolute = min(previous+1.,began+CAP)
        snapshot = mon.collect(absolute)
        scope = mon.scope(snapshot['rows'],os.getpid(),None if child is None else child.pid)
        allowed = scope.pop('allowed')
        require(not classifier(snapshot['rows'],allowed),'No other numerical worker')
        for row in snapshot['rows']:
            if row['pid'] not in allowed:
                try:
                    argv = shlex.split(row['command'])
                except ValueError:
                    argv = row['command'].split()
                require('flopscope_server' not in argv,'No foreign numerical server')
        if swap0 is None:
            swap0 = snapshot['swap']['used_bytes']
        growth = snapshot['swap']['used_bytes']-swap0
        require(growth <= 0 and 0 < scope['combined_rss_bytes'] < RSS,
                'Zero added system swap and full owned tree plus watch<8GB')
        require(snapshot['commands_reaped'] and not mon.PENDING,'Observations reaped')
        deadline()
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began,**scope,
            swap_bytes=snapshot['swap']['used_bytes'],swap_growth_bytes=growth,
            transport=snapshot['command_records'],uncached_single_process_table=True),allow_nan=False)+'\n')
        stream.flush()
        previous,interval = mon.complete(previous,absolute)
        result['maximum_combined_rss_bytes'] = max(result['maximum_combined_rss_bytes'],scope['combined_rss_bytes'])
        result['maximum_interval_s'] = max(result['maximum_interval_s'],interval)
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        result['completed_sample_intervals_s'].append(interval);samples += 1

    def tick():
        deadline()
        if time.monotonic()-previous >= .2:
            observe()

    def interrupted(*unused):
        raise RuntimeError('Owned single-call gauge diagnostic interrupted')

    handlers = {sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        stream = (out/'samples.jsonl').open('x');observe()
        for path,digest in {**own,**inputs}.items():
            require(h.sha(path,tick) == digest,'Reviewed source or mocked observation boundary')
        forwarded = dict(source=args.worker_sha256,gauge_source=args.gauge_source_sha256,
            protocol=args.protocol_sha256,watcher=args.source_sha256,watch_spec=args.spec_sha256)
        argv = [str(OFFPY),'-B','-u',str(HERE/'worker.py'),'--run']
        for key,value in forwarded.items():
            argv += ['--'+key.replace('_','-')+'-sha256',value]
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            PYTORCH_ENABLE_MPS_FALLBACK='0',FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
        env.update({k:'1' for k in BLAS})
        log = (out/'worker.log').open('x')
        child = subprocess.Popen(argv,env=env,cwd=NEW,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result.update(child_pid=child.pid,command=argv)
        h.write(out/'started.json',result)
        while child.poll() is None:
            observe();time.sleep(.15)
        observe();require(child.wait(timeout=1) == 0,'Complete one-call diagnostic exit')
        mon.cleanup([child],time.monotonic()+.5)
        paths = [folder/name for name in ('result.json','completion.json','finalization.json')]
        digests = [h.sha(p,tick) for p in paths];r,e,f = map(h.read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == digests[0]
            and f['completion_sha256'] == digests[1] and f['elapsed_s'] < CAP,
            'Actual completed child three-seal chain')
        require(r['schema'] == 'current_comp_birth_gauge_size_v1'
            and r['candidate_sha256'] == CANDIDATE_SHA
            and r['full_model_calls'] == r['retired_call_count'] == 1
            and r['truth_accessed'] is False and r['scientific_score_measured'] is False,
            'Only one unchanged-COMP target-free size diagnostic')
        for key in ('hashes_unchanged','sources_unchanged','baseline_output_byte_exact',
                    'domains_byte_exact','selectors_byte_exact','own_complete_bills'):
            require(r.get(key) is True,'One-call integrity: '+key)
        require(type(r['diagnostic_flops']) is int and 0 < r['diagnostic_flops'] <= DIAGNOSTIC_CAP
            and r['model_flops_after_diagnostic_removal'] == 471970647528
            and r['model_operation_count_after_diagnostic_removal'] == 31324
            and 0 < r['peak_rss_bytes'] < RSS,'Actual paid diagnostic and unchanged model counters')
        required = {HERE/'worker.py':args.worker_sha256,HERE/'gauge_fnp.py':args.gauge_source_sha256,
            HERE/'PROTOCOL.md':args.protocol_sha256,Path(__file__).resolve():args.source_sha256,
            HERE/'WATCH_SPEC.md':args.spec_sha256}
        require(all(r['source_bindings'].get(str(p)) == v for p,v in required.items()),
                'Exactly reviewed numerical and observer source owners')
        union = {}
        for group in ('source_bindings','input_bindings','artifacts'):
            require(isinstance(r[group],dict) and r[group],'Complete child nontarget inventory')
            for path,digest in r[group].items():
                require(path not in union or union[path] == digest,'Consistent child inventory')
                union[path] = digest
        for path,digest in {**union,**{str(p):v for p,v in own.items()},
                            **{str(p):v for p,v in inputs.items()}}.items():
            require(h.sha(path,tick) == digest,'Fresh independent complete source/input/output hash')
        require([h.sha(p,tick) for p in paths] == digests,'Immutable child seals after verification')
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            child_execution_completed=True,whole_owned_tree_retired=True,
            child_result_sha256=digests[0],child_completion_sha256=digests[1],child_finalization_sha256=digests[2],
            full_model_calls=1,retired_call_count=1,diagnostic_flops=r['diagnostic_flops'],
            verified_child_nontarget_files=len(union),actual_completed_samples=True,
            foreign_numerical_workers_checked=True)
        for path,digest in zip(paths,digests):
            result['input_bindings'][str(path)] = digest
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
        for handle in (log,stream):
            if handle is not None:
                handle.close()
        result.update(elapsed_s=time.monotonic()-began,observation_samples=samples,initial_swap_bytes=swap0)
        if result['elapsed_s'] >= CAP:
            result.update(passed=False,completed=False,error='Whole180s including cleanup')
        for name in ('started.json','worker.log','samples.jsonl'):
            path = out/name
            if path.exists():
                result['artifacts'][str(path)] = h.sha(path)
        h.write(out/'result.json',result);rh=h.sha(out/'result.json')
        ok=bool(result['passed'] and time.monotonic()-began < CAP)
        h.write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh))
        ch=h.sha(out/'completion.json');elapsed=time.monotonic()-began
        ok=bool(ok and elapsed < CAP)
        h.write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,
            completion_sha256=ch,elapsed_s=elapsed))
        h.sha(out/'finalization.json');ok=bool(ok and time.monotonic()-began < CAP)
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s','full_model_calls',
        'maximum_combined_rss_bytes','maximum_interval_s','maximum_swap_growth_bytes','error')},allow_nan=False),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
