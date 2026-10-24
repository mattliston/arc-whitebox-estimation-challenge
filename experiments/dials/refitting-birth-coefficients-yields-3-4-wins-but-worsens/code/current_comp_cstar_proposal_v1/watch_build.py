"""One separately bounded literal build; target-free external supervision."""
import argparse
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
P2 = HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
TRANSPORT = HERE/'watch.py'
TRANSPORT_SHA = 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
SCHEMA = 'current_comp_cstar_proposal_build_watch_v1'
BLAS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def transport():
    require(hashlib.sha256(TRANSPORT.read_bytes()).hexdigest() == TRANSPORT_SHA,
        'Frozen proposal transport-only helpers')
    spec = importlib.util.spec_from_file_location('cprop_build_transport_only',TRANSPORT)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monitor,classifier = helper.transport()
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),
        'Pure stdlib source and mocked observation transport')
    return helper,monitor,classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--numerical-slot-clear',action='store_true')
    keys = ('source','builder','protocol','spec','qualification-result','qualification-watch',
        'training-result','training-watch')
    for key in keys:
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = previous = time.monotonic()
    require(args.numerical_slot_clear and os.getpriority(os.PRIO_PROCESS,0) >= 10
        and sys.flags.optimize == 0 and sys.dont_write_bytecode
        and all(os.environ.get(k) == '1' for k in BLAS),'Exclusive nice10, BLAS1, bytecodeoff build stage')
    for key in keys:
        require(re.fullmatch('[0-9a-f]{64}',getattr(args,key.replace('-','_')+'_sha256') or ''),
            'Explicit reviewed owner/actual result: '+key)
    h,mon,classifier = transport()
    own = {Path(__file__).resolve():args.source_sha256,HERE/'build.py':args.builder_sha256,
        HERE/'BUILD_PROTOCOL.md':args.protocol_sha256,HERE/'BUILD_WATCH_SPEC.md':args.spec_sha256,
        TRANSPORT:TRANSPORT_SHA,**h.MON_PINS}
    inputs = dict(h.MOCK_PINS)
    for folder,digest in (('qualification_v1',args.qualification_result_sha256),
            ('qualification_watch_v1',args.qualification_watch_sha256),
            ('training_v1',args.training_result_sha256),('training_watch_v1',args.training_watch_sha256)):
        inputs[HERE/folder/'result.json'] = digest
    for path,digest in {**own,**inputs}.items():
        require(h.sha(path) == digest,'Explicit prospective source/result: '+str(path))
    folder,out = HERE/'build_v1',HERE/'build_watch_v1'
    require(not folder.exists() and not out.exists() and not HERE.is_symlink(),'Exclusive sole terminal build')
    out.mkdir(exist_ok=False)
    result = dict(schema=SCHEMA,role='terminal',passed=False,completed=False,
        source_bindings={str(p):v for p,v in own.items()},input_bindings={str(p):v for p,v in inputs.items()},
        artifacts={},numerical_imports=False,truth_arrays_accessed=False,full_model_calls=0,
        maximum_combined_rss_bytes=0,maximum_interval_s=0.,maximum_swap_growth_bytes=0.,
        completed_sample_intervals_s=[],sources_unchanged=False,hashes_unchanged=False,
        whole_owned_tree_retired=False,child_execution_completed=False,
        whole_cap_s=120.,combined_rss_limit_bytes=2_000_000_000,numerical_slot_clear_attested=True)
    child=log=stream=None;swap0=None;samples=0
    def deadline():
        require(time.monotonic()-began < 120.,'Whole build120s through admission/reap/final publication')
    def observe():
        nonlocal previous,swap0,samples
        absolute = min(previous+1.,began+120.)
        snapshot = mon.collect(absolute)
        scope = mon.scope(snapshot['rows'],os.getpid(),None if child is None else child.pid)
        require(not classifier(snapshot['rows'],scope.pop('allowed')),'No other numerical worker')
        if swap0 is None:
            swap0 = snapshot['swap']['used_bytes']
        growth = snapshot['swap']['used_bytes']-swap0
        require(growth <= 0 and scope['combined_rss_bytes'] < 2_000_000_000,'Zero added swap and combined2GB')
        require(snapshot['commands_reaped'] and not mon.PENDING,'Observation groups reaped')
        deadline()
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began,**scope,
            swap_bytes=snapshot['swap']['used_bytes'],swap_growth_bytes=growth,
            transport=snapshot['command_records'],uncached_single_process_table=True),allow_nan=False)+'\n')
        stream.flush()
        previous,interval = mon.complete(previous,absolute)
        result['maximum_combined_rss_bytes'] = max(result['maximum_combined_rss_bytes'],scope['combined_rss_bytes'])
        result['maximum_interval_s'] = max(result['maximum_interval_s'],interval)
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        result['completed_sample_intervals_s'].append(interval)
        samples += 1
    def tick():
        deadline()
        if time.monotonic()-previous >= .2:
            observe()
    def interrupted(*unused):
        raise RuntimeError('Owned literal-build stage interrupted')
    handlers = {sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        stream = (out/'samples.jsonl').open('x'); observe()
        forwarded = dict(source=args.builder_sha256,protocol=args.protocol_sha256,
            watcher=args.source_sha256,watch_spec=args.spec_sha256,
            qualification_result=args.qualification_result_sha256,qualification_watch=args.qualification_watch_sha256,
            training_result=args.training_result_sha256,training_watch=args.training_watch_sha256)
        argv = [str(OFFPY),'-B','-u',str(HERE/'build.py'),'--build']
        for key,value in forwarded.items():
            argv += ['--'+key.replace('_','-')+'-sha256',value]
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
        env.update({k:'1' for k in BLAS})
        log = (out/'worker.log').open('x')
        child = subprocess.Popen(argv,env=env,cwd=HERE.parent,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result.update(child_pid=child.pid,command=argv)
        h.write(out/'started.json',result)
        while child.poll() is None:
            observe(); time.sleep(.15)
        observe();require(child.wait(timeout=1) == 0,'Complete builder exit')
        mon.cleanup([child],time.monotonic()+.5)
        paths = [folder/name for name in ('result.json','completion.json','finalization.json')]
        digests = [h.sha(p,tick) for p in paths]
        r,e,f = map(h.read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == digests[0]
            and f['completion_sha256'] == digests[1] and f['elapsed_s'] < 120.,'Complete actual new three-seal build')
        require(r['schema'] == 'current_comp_cstar_proposal_build_v1'
            and r['hashes_unchanged'] and r['sources_unchanged'] and r['parameter_updates'] == 0
            and r['estimator_executed'] is False and r['truth_arrays_accessed'] is False
            and r['peak_rss_bytes'] < 2_000_000_000,'Sole bounded literal construction, no model or fitting')
        required_owners = {HERE/'build.py':args.builder_sha256,HERE/'BUILD_PROTOCOL.md':args.protocol_sha256,
            Path(__file__).resolve():args.source_sha256,HERE/'BUILD_WATCH_SPEC.md':args.spec_sha256}
        require(all(r['source_bindings'].get(str(p)) == v for p,v in required_owners.items()),'Exact four build owners')
        union = {}
        for group in ('source_bindings','input_bindings','artifacts'):
            require(isinstance(r[group],dict) and r[group],'Complete child nontarget inventory')
            for path,digest in r[group].items():
                require(path not in union or union[path] == digest,'No conflicting child binding')
                union[path] = digest
        for path,digest in union.items():
            require(h.sha(path,tick) == digest,'Fresh independent child source/input/artifact hash')
        for path,digest in {**own,**inputs}.items():
            require(h.sha(path,tick) == digest,'Own source and supplied input unchanged')
        require([h.sha(p,tick) for p in paths] == digests,'Immutable three child seals after verification')
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            child_execution_completed=True,whole_owned_tree_retired=True,
            child_result_sha256=digests[0],child_completion_sha256=digests[1],child_finalization_sha256=digests[2],
            candidate=r['candidate'],verified_child_nontarget_files=len(union),
            actual_completed_samples=True,foreign_numerical_workers_checked=True)
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
        if result['elapsed_s'] >= 120.:
            result.update(passed=False,completed=False,error='Whole120s including cleanup')
        for name in ('started.json','worker.log','samples.jsonl'):
            path = out/name
            if path.exists():
                result['artifacts'][str(path)] = h.sha(path)
        h.write(out/'result.json',result);rh=h.sha(out/'result.json')
        ok=bool(result['passed'] and time.monotonic()-began < 120.)
        h.write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh))
        ch=h.sha(out/'completion.json');elapsed=time.monotonic()-began
        ok=bool(ok and elapsed < 120.)
        h.write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,
            completion_sha256=ch,elapsed_s=elapsed))
        h.sha(out/'finalization.json')
        ok=bool(ok and time.monotonic()-began < 120.)
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s','maximum_combined_rss_bytes',
        'maximum_interval_s','candidate','error')},allow_nan=False),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
