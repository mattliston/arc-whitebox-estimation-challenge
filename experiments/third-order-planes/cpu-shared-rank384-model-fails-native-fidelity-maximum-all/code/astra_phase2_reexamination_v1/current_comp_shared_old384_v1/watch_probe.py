"""External single-worker300s/8GB/no-swap watch for fixed native37/repeat37."""
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
NEW, P2 = HERE.parent,HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
TRANSPORT = NEW/'current_comp_cstar_proposal_v1/watch.py'
TRANSPORT_SHA = 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
CAP, RSS, PROGRESS = 300.,8_000_000_000,120.
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
KEYS = ('source','worker','protocol','spec','compose','fnp','design')


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def sha(path,tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and 'truthpost' not in path.parts and not
        (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()), 'No supervisor target access')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size < 16*1024**2,'Bounded metadata')
    return json.loads(path.read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def load(path,name,digest):
    require(sha(path) == digest,'Reviewed source')
    spec = importlib.util.spec_from_file_location(name,path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--numerical-slot-clear',action='store_true')
    for key in KEYS:
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = previous = last_progress = time.monotonic()
    require(args.numerical_slot_clear and Path(sys.executable).resolve() == OFFPY.resolve()
        and os.getpriority(os.PRIO_PROCESS,0) >= 10 and sys.flags.optimize == 0 and sys.dont_write_bytecode
        and all(os.environ.get(k) == '1' for k in BLAS)
        and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
        and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0','Official exclusive nice10/BLAS1/no fallback')
    own = {HERE/'watch_probe.py':args.source_sha256,HERE/'probe.py':args.worker_sha256,
        HERE/'PROBE_PROTOCOL.md':args.protocol_sha256,HERE/'PROBE_WATCH_SPEC.md':args.spec_sha256,
        HERE/'compose.py':args.compose_sha256,HERE/'fnp.py':args.fnp_sha256,
        HERE/'PROTOCOL.md':args.design_sha256,TRANSPORT:TRANSPORT_SHA}
    require(all(re.fullmatch('[0-9a-f]{64}',v or '') and sha(p) == v for p,v in own.items()),
            'Every reviewed new source explicit')
    h = load(TRANSPORT,'shared384_watch_transport',TRANSPORT_SHA)
    mon,classifier = h.transport()
    child_folder,out = HERE/'probe_v1',HERE/'probe_watch_v1'
    require(not child_folder.exists() and not out.exists() and not HERE.is_symlink(),'Fresh exclusive stages')
    out.mkdir()
    result = dict(schema='current_comp_shared_old384_native37_watch_v1',passed=False,completed=False,
        source_bindings={str(p):v for p,v in {**own,**h.MON_PINS}.items()},
        input_bindings={str(p):v for p,v in h.MOCK_PINS.items()},artifacts={},
        whole_cap_s=CAP,progress_cap_s=PROGRESS,combined_rss_limit_bytes=RSS,
        numerical_imports=False,truth_accessed=False,score_measured=False,uploaded=False,
        maximum_combined_rss_bytes=0,maximum_swap_growth_bytes=0,maximum_interval_s=0,
        retired_calls=0,ranks=[37,37],completed_sample_intervals_s=[],markers=[])
    child=log=reader=stream=None
    initial_swap=None; pending=''; samples=0
    def deadline():
        now = time.monotonic()
        require(now-began < CAP and now-last_progress < PROGRESS,'Whole300s/next retired call120s')
    def interrupted(*unused):
        raise RuntimeError('Owned fixed shared-basis stage interrupted')
    def observe():
        nonlocal previous,last_progress,initial_swap,pending,samples
        absolute = min(previous+1.,began+CAP,last_progress+PROGRESS)
        sample = mon.collect(absolute)
        scope = mon.scope(sample['rows'],os.getpid(),None if child is None else child.pid)
        require(not classifier(sample['rows'],scope.pop('allowed')),'No foreign numerical worker')
        if initial_swap is None:
            initial_swap = sample['swap']['used_bytes']
        growth = sample['swap']['used_bytes']-initial_swap
        require(growth <= 0 and scope['combined_rss_bytes'] < RSS,'Zero added swap/full owned tree8GB')
        if reader is not None:
            pending += reader.read()
            require(len(pending) < 4*1024**2,'Bounded live log')
            while '\n' in pending:
                line,pending = pending.split('\n',1)
                try:
                    item = json.loads(line)
                except (ValueError,TypeError):
                    continue
                if isinstance(item,dict) and item.get('event') == 'retired':
                    deadline(); index = result['retired_calls']
                    require(index < 2 and item['call'] == index and item['rank'] == 37,'Exact ordered37/37')
                    path = child_folder/f'call_{index:02d}_complete.json'
                    record,digest = read(path),sha(path)
                    require(digest == item['complete_sha256'] and record['passed'] and record['completed']
                        and record['call'] == index and record['rank'] == 37 and record['context_retired'],
                        'Actual completed released context, not heartbeat')
                    for suffix,key in (('bill','bill_sha256'),('lifecycle','lifecycle_sha256'),
                            ('diagnostics','diagnostic_sha256')):
                        require(sha(child_folder/f'call_{index:02d}_{suffix}.json') == record[key],
                                'Retired marker owns saved numerical records')
                    result['markers'].append(dict(path=str(path),sha256=digest,call=index,rank=37))
                    result['retired_calls'] += 1
                    last_progress = time.monotonic()
        deadline()
        result['maximum_combined_rss_bytes'] = max(result['maximum_combined_rss_bytes'],scope['combined_rss_bytes'])
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began,**scope,
            swap_bytes=sample['swap']['used_bytes'],retired_calls=result['retired_calls'],
            observational_groups_retired=sample['commands_reaped'],transport=sample['command_records']),sort_keys=True)+'\n')
        stream.flush()
        previous,interval = mon.complete(previous,absolute)
        result['completed_sample_intervals_s'].append(interval)
        result['maximum_interval_s'] = max(result['maximum_interval_s'],interval); samples += 1
    def tick():
        deadline()
        if time.monotonic()-previous >= .2:
            observe()
    handlers = {s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        stream = (out/'samples.jsonl').open('x'); observe()
        probe = load(HERE/'probe.py','shared384_passive_probe_api',args.worker_sha256)
        probe.environment()
        composition = load(HERE/'compose.py','shared384_passive_composition',args.compose_sha256)
        pins = composition.authenticate(tick)
        for path,digest in pins.items():
            require(sha(path,tick) == digest,'Complete composition input pins')
            result['source_bindings'][str(path)] = digest
        text,proof = composition.derive_source(tick)
        result['candidate_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        result['composition_proof'] = proof
        del text
        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),'No supervisor numerical imports')
        argv = [str(OFFPY),'-B','-u',str(HERE/'probe.py'),'--run',
            '--source-sha256',args.worker_sha256,'--watch-sha256',args.source_sha256,
            '--protocol-sha256',args.protocol_sha256,'--watch-spec-sha256',args.spec_sha256,
            '--compose-sha256',args.compose_sha256,'--fnp-sha256',args.fnp_sha256,'--design-sha256',args.design_sha256]
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
        env.update({k:'1' for k in BLAS})
        log = (out/'worker.log').open('x'); reader = (out/'worker.log').open('r')
        child = subprocess.Popen(argv,cwd=NEW,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result['child_pid'],result['command'] = child.pid,argv
        write(out/'started.json',result)
        while child.poll() is None:
            observe(); time.sleep(.15)
        observe()
        require(child.wait(timeout=1) == 0 and result['retired_calls'] == 2,'Two complete calls and exit0')
        mon.cleanup([child],time.monotonic()+.5)
        files = [child_folder/name for name in ('result.json','completion.json','finalization.json')]
        digests = [sha(p,tick) for p in files]
        record,end,final = [read(p) for p in files]
        require(all(x.get('passed') is True and x.get('completed') is True for x in (record,end,final))
            and end['result_sha256'] == final['result_sha256'] == digests[0]
            and final['completion_sha256'] == digests[1] and final['elapsed_s'] < CAP,'Actual worker three seals')
        require(record['candidate_sha256'] == result['candidate_sha256']
            and record['composition_proof'] == result['composition_proof']
            and record['full_model_calls'] == record['retired_call_count'] == 2
            and record['ranks'] == [37,37] and record['all16_domains_qualified']
            and record['all47_selectors_qualified'] and record['repeat_byte_exact'] and record['own_complete_bills']
            and record['unaffected_prefix_checks'] == [dict(call=i,layers=5,byte_exact=True) for i in range(2)]
            and record['truth_accessed'] is False and record['score_measured'] is False,
            'Actual target-blind same-source own-cost native subject')
        for group in ('source_bindings','input_bindings','artifacts'):
            require(isinstance(record[group],dict) and record[group],'Complete actual inventory')
            for path,digest in record[group].items():
                require(sha(path,tick) == digest,'Fresh complete child inventory: '+path)
                result[group][path] = digest
        require(all(record['source_bindings'].get(str(p)) == d for p,d in own.items() if p != TRANSPORT),
                'Same reviewed worker/watch/design owners')
        for path,digest in result['source_bindings'].items():
            require(sha(path,tick) == digest,'All supervisor sources unchanged')
        for path,digest in result['input_bindings'].items():
            require(sha(path,tick) == digest,'All merged input/monitor evidence unchanged')
        require([sha(p,tick) for p in files] == digests,'Seals immutable through verification')
        result.update(passed=True,completed=True,child_execution_completed=True,whole_owned_tree_retired=True,
            child_result_sha256=digests[0],child_completion_sha256=digests[1],child_finalization_sha256=digests[2],
            hashes_unchanged=True,sources_unchanged=True,actual_completed_samples=True,
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
        result.update(elapsed_s=time.monotonic()-began,observation_samples=samples,initial_swap_bytes=initial_swap,
            last_completed_observation_elapsed_s=previous-began,post_observation_memory_sample_claimed=False)
        for name in ('started.json','worker.log','samples.jsonl'):
            if (out/name).exists():
                result['artifacts'][str(out/name)] = sha(out/name)
        def final_ok():
            now = time.monotonic()
            return bool(result['passed'] and now-began < CAP and now-last_progress < PROGRESS and now-previous <= 1)
        if not final_ok():
            result.update(passed=False,completed=False)
        write(out/'result.json',result); rh = sha(out/'result.json'); ok = final_ok()
        write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh)); ch = sha(out/'completion.json')
        elapsed = time.monotonic()-began; ok = bool(ok and final_ok())
        write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,
            elapsed_s=elapsed,terminal_publication_interval_s=time.monotonic()-previous))
        sha(out/'finalization.json'); ok = bool(ok and final_ok())
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
    print(json.dumps(dict(passed=ok,completed=ok,retired_calls=result['retired_calls'],
        elapsed_s=time.monotonic()-began,maximum_combined_rss_bytes=result['maximum_combined_rss_bytes'],
        maximum_interval_s=result['maximum_interval_s'],maximum_swap_growth_bytes=result['maximum_swap_growth_bytes'],
        error=result.get('error'))),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
