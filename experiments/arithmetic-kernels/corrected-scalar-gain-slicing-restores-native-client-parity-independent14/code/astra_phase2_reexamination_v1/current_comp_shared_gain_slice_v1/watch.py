"""One source build, one native35/35, one real-client35/35, strictly serial."""
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
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
OPENPY = NEW/'openblas_runtime_env_v1/bin/python'
CLIENTPY = P2/'data_gate/.client-venv/bin/python'
TRANSPORT = NEW/'current_comp_cstar_proposal_v1/watch.py'
TRANSPORT_SHA = 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
CAP, RSS, PROGRESS = 600., 8_000_000_000, 120.
TAG = 'current_comp_shared_gain_slice_pair_v1'
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
SOURCE_FILES = dict(source='watch.py', spec='PAIR_WATCH_SPEC.md',
    builder='build.py', build_protocol='BUILD_PROTOCOL.md', native='native.py',
    native_protocol='NATIVE_PROTOCOL.md', client='client.py', checks='pair_checks.py')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def no_numerics():
    require(not any(n.split('.')[0] in ('numpy','scipy','torch','flopscope',
            'flopscope_server','threadpoolctl') for n in sys.modules),
            'Entire supervisor is stdlib-only')


def sha(path, tick=None):
    path = Path(path)
    require(not path.is_symlink() and path.resolve().is_relative_to(P2)
            and 'truthpost' not in path.parts and 'truth' not in path.name.lower(),
            'Regular in-scope non-target source, input or output')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    require(not Path(path).is_symlink() and Path(path).stat().st_size < 16*1024**2,
            'Bounded actual metadata')
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def load(path, digest, name):
    require(sha(path) == digest, 'Exact reviewed inert source')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    no_numerics()
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--numerical-slot-clear',action='store_true')
    for key in SOURCE_FILES:
        parser.add_argument('--'+key.replace('_','-')+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = previous = last_progress = time.monotonic()
    require(args.numerical_slot_clear and Path(sys.prefix).resolve() == OFFPY.parent.parent.resolve()
        and os.getpriority(os.PRIO_PROCESS,0) >= 10 and sys.flags.optimize == 0 and sys.dont_write_bytecode
        and all(os.environ.get(k) == '1' for k in BLAS)
        and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
        and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
        and not os.environ.get('PYTHONPATH') and not os.environ.get('FLOPSCOPE_SERVER_URL'),
        'Exclusive nice10/BLAS1/no-fallback/assertions source-bound supervisor')
    own = {HERE/name:getattr(args,key+'_sha256') for key,name in SOURCE_FILES.items()}
    own[TRANSPORT] = TRANSPORT_SHA
    require(all(re.fullmatch('[0-9a-f]{64}',d or '') and sha(p) == d for p,d in own.items()),
            'All prospective source owners supplied and exact')
    h = load(TRANSPORT,TRANSPORT_SHA,'gain_pair_transport')
    mon,classifier = h.transport()
    out, native_folder, build_folder = HERE/'pair_watch_v1', HERE/'native_v1', HERE/'build_v1'
    series = NEW/(TAG+'.json')
    remote_paths = [NEW/f'{TAG}_{i}_r35.f32' for i in range(2)]
    require(not HERE.is_symlink() and not any(p.exists() for p in
        (out,native_folder,build_folder,series,*remote_paths)), 'Fresh stages, no retries or overwrites')
    out.mkdir()
    result = dict(schema='current_comp_shared_gain_slice_pair_watch_v1',passed=False,completed=False,
        source_bindings={str(p):d for p,d in {**own,**h.MON_PINS}.items()},
        input_bindings={str(p):d for p,d in h.MOCK_PINS.items()},artifacts={},commands=[],
        numerical_imports=False,truth_accessed=False,raw_error_measured=False,adjusted_score_measured=False,
        parameter_updates=0,full_five_network_fidelity_qualified=False,submission_authorized=False,
        maximum_combined_rss_bytes=0,maximum_swap_growth_bytes=0,maximum_interval_s=0,
        completed_sample_intervals_s=[],native_markers=[],remote_records=[],
        whole_cap_s=CAP,progress_cap_s=PROGRESS,combined_rss_limit_bytes=RSS)
    children,logs,readers = [],[],[]
    phase,reader,pending = 'admission',None,''
    phase_started = began
    initial_swap = None
    stream = (out/'samples.jsonl').open('x')

    def deadline():
        now = time.monotonic()
        require(now < began+CAP and now < last_progress+PROGRESS,
                'Whole600s and <=120s since last fully retired context/phase')
        phase_cap = {'build':60.,'native':300.,'remote':180.}.get(phase,CAP)
        require(now < phase_started+phase_cap, 'Independent current phase wall cap')

    def observe():
        nonlocal previous,last_progress,initial_swap,pending
        phase_cap = {'build':60.,'native':300.,'remote':180.}.get(phase,CAP)
        absolute = min(previous+1.,began+CAP,last_progress+PROGRESS,phase_started+phase_cap)
        sample = mon.collect(absolute)
        scopes = [mon.scope(sample['rows'],os.getpid(),c.pid) for c in children]
        scopes.append(mon.scope(sample['rows'],os.getpid(),None))
        allowed = set().union(*(s['allowed'] for s in scopes))
        owned = mon.descendants({os.getpid()},sample['rows'])
        owned.update(pid for s in scopes for pid in s['child_pids'])
        combined = sum(r['rss_bytes'] for r in sample['rows'] if r['pid'] in owned)
        require(not classifier(sample['rows'],allowed), 'No foreign numerical worker')
        if initial_swap is None:
            initial_swap = sample['swap']['used_bytes']
        growth = sample['swap']['used_bytes']-initial_swap
        require(growth <= 0 and combined < RSS, 'Whole owned tree<8GB and no added swap')
        require(phase != 'build' or combined < 2_000_000_000,
                'Independent source-only build whole tree<2GB')
        if reader is not None:
            pending += reader.read()
            require(len(pending) < 4*1024**2, 'Bounded live worker log')
            while '\n' in pending:
                line,pending = pending.split('\n',1)
                try:
                    item = json.loads(line)
                except (ValueError,TypeError):
                    continue
                if not isinstance(item,dict):
                    continue
                if phase == 'native' and item.get('event') == 'retired':
                    deadline(); index = len(result['native_markers'])
                    require(index < 2 and item['call'] == index and item['rank'] == 35,
                            'Exact native35/35 retired order')
                    path = native_folder/f'call_{index:02d}_complete.json'
                    record,digest = read(path),sha(path)
                    require(digest == item['complete_sha256'] and record['passed'] and record['completed']
                        and record['call'] == index and record['rank'] == 35 and record['context_retired'],
                        'Actual native context retired, not heartbeat')
                    for suffix,key in (('bill','bill_sha256'),('lifecycle','lifecycle_sha256'),
                                       ('diagnostics','diagnostic_sha256')):
                        require(sha(native_folder/f'call_{index:02d}_{suffix}.json') == record[key],
                                'Retired event owns exact saved bill/lifecycle/domain records')
                    result['native_markers'].append(dict(path=str(path),sha256=digest,call=index,rank=35))
                    last_progress = time.monotonic()
                elif phase == 'remote' and 'index' in item and 'flops' in item:
                    deadline(); index = len(result['remote_records'])
                    checks.check_bill(item,index)
                    raw,_ = checks.raw_f32(remote_paths[index])
                    require(hashlib.sha256(raw).hexdigest() == item['output_sha256'],
                            'Actual saved output after complete closed production context')
                    result['remote_records'].append(item)
                    last_progress = time.monotonic()
        deadline()
        result['maximum_combined_rss_bytes'] = max(result['maximum_combined_rss_bytes'],combined)
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began,phase=phase,
            owned_pids=sorted(owned),combined_rss_bytes=combined,swap_bytes=sample['swap']['used_bytes'],
            native_retired=len(result['native_markers']),remote_retired=len(result['remote_records']),
            observational_groups_retired=sample['commands_reaped'],transport=sample['command_records']))+'\n')
        stream.flush()
        previous,interval = mon.complete(previous,absolute)
        result['maximum_interval_s'] = max(result['maximum_interval_s'],interval)
        result['completed_sample_intervals_s'].append(interval)

    def tick():
        deadline()
        if time.monotonic()-previous >= .15:
            observe()

    def spawn(argv,name,env,progress=False):
        nonlocal reader,pending,phase,phase_started
        log = (out/(name+'.log')).open('x'); logs.append(log)
        if name in ('build','native','remote'):
            phase,phase_started = name,time.monotonic()
        if progress:
            reader = (out/(name+'.log')).open('r'); readers.append(reader)
            pending,phase = '',name
        child = subprocess.Popen(argv,cwd=NEW,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child)
        result['commands'].append(dict(argv=argv,pid=child.pid,group=child.pid))
        observe()
        return child

    def finish(child,server=None):
        while child.poll() is None:
            if server is not None:
                require(server.poll() is None, 'Sole IPC server remains alive')
            observe(); time.sleep(.10)
        observe()
        require(child.wait(timeout=.1) == 0, 'Actual child exit0, no retry')
        mon.cleanup([child],time.monotonic()+.5); observe()

    def merge(record):
        for group in ('source_bindings','input_bindings','artifacts'):
            for path,digest in record.get(group,{}).items():
                require(sha(path,tick) == digest, 'Actual complete child inventory unchanged')
                require(path not in result[group] or result[group][path] == digest, 'No conflicting owner')
                result[group][path] = digest

    def chain(folder,cap):
        paths = [folder/n for n in ('result.json','completion.json','finalization.json')]
        ds = [sha(p,tick) for p in paths]; r,e,f = map(read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == ds[0] and f['completion_sha256'] == ds[1]
            and 0 <= r['elapsed_s'] <= f['elapsed_s'] < cap, 'Actual positive bounded three-seal child')
        result['artifacts'].update({str(p):d for p,d in zip(paths,ds)})
        merge(r)
        return r,dict(zip(('result_sha256','completion_sha256','finalization_sha256'),ds))

    def interrupted(*unused):
        raise RuntimeError('Gain-slice pair interrupted')

    handlers = {s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        observe()
        checks = load(HERE/'pair_checks.py',args.checks_sha256,'gain_pair_checks')
        builder = load(HERE/'build.py',args.builder_sha256,'gain_pair_builder')
        native_api = load(HERE/'native.py',args.native_sha256,'gain_pair_native_metadata')
        client_api = load(HERE/'client.py',args.client_sha256,'gain_pair_client_metadata')
        for path,digest in client_api.PINS.items():
            require(sha(path,tick) == digest, 'Exact original client and canonical physical fixture')
            group = 'source_bindings' if Path(path).suffix == '.py' else 'input_bindings'
            key = str(path)
            require(key not in result[group] or result[group][key] == digest, 'Consistent client owner')
            result[group][key] = digest
        client_source,client_proof = client_api.source_and_proof()
        compile(client_source,'gain-client-source-only','exec')
        result['client_source_proof'] = client_proof
        del client_source
        no_numerics(); write(out/'started.json',result); tick()
        env = dict(os.environ,PYTHONUNBUFFERED='1')
        phase = 'build'
        child = spawn([str(OFFPY),'-B','-u',str(HERE/'build.py'),'--run',
            '--source-sha256',args.builder_sha256,'--protocol-sha256',args.build_protocol_sha256],
            'build',env)
        finish(child)
        build,bs = chain(build_folder,60.)
        candidate_sha = build['candidate_sha256']
        builder.verify_build(bs['result_sha256'],candidate_sha,args.builder_sha256,args.build_protocol_sha256,tick)
        result.update(candidate_sha256=candidate_sha,build_seals=bs)
        last_progress = time.monotonic()
        native_argv = native_api.command_args(args.native_sha256,args.source_sha256,
            args.native_protocol_sha256,args.spec_sha256,args.builder_sha256,args.build_protocol_sha256,
            bs['result_sha256'],candidate_sha)
        child = spawn([str(OFFPY),'-B','-u',str(HERE/'native.py'),*native_argv],'native',env,True)
        finish(child)
        require(len(result['native_markers']) == 2, 'Exactly two actually retired native calls')
        native,ns = chain(native_folder,300.)
        native_api.verify_native(native,tick)
        require(native['candidate_sha256'] == candidate_sha, 'Same exact new candidate in native calls')
        result['native_seals'] = ns
        reader,phase = None,'between-pairs'
        last_progress = time.monotonic()
        scratch = Path(tempfile.mkdtemp(prefix='astra_gain_slice_pair_',dir='/private/tmp'))
        result['ipc_directory'] = str(scratch)
        url = 'ipc://'+str(scratch/'flops.sock')
        remote_env = dict(env,FLOPSCOPE_SERVER_URL=url)
        server = spawn([str(OPENPY),'-B','-u','-m','flopscope_server','--url',url,'--timeout','600'],
            'server',remote_env)
        server_start = time.monotonic()
        while not (scratch/'flops.sock').exists():
            require(server.poll() is None and time.monotonic()-server_start < 5, 'Bounded owned IPC startup')
            observe(); time.sleep(.05)
        candidate_relative = str((HERE/'build_v1/estimator.py').relative_to(NEW))
        child = spawn([str(CLIENTPY),'-B','-u',str(HERE/'client.py'),'--run',
            '--candidate',candidate_relative,'--tag',TAG],'remote',remote_env,True)
        finish(child,server)
        mon.cleanup([server],time.monotonic()+.5); observe()
        require(len(result['remote_records']) == 2, 'Two actually completed production contexts')
        measured = read(series)
        require(measured['status'] == 'completed' and measured['candidate'] == candidate_relative
            and measured['source_sha256'] == candidate_sha and measured['records'] == result['remote_records']
            and measured['repeated_output_byte_identical'] is True, 'Same source and actual persistent-client pair')
        for path in (series,*remote_paths):
            result['artifacts'][str(path)] = sha(path,tick)
        result['comparison'] = checks.compare(native,result['remote_records'],remote_paths,tick)
        require(result['comparison']['all16_vector_gate_passed'], 'Original all16 vector tolerance<=1e-5')
        require(result['comparison']['native_initialization_flops'] ==
                result['comparison']['production_initialization_flops'] == 1944970298,
                'Both own complete initialization deltas')
        for group in ('source_bindings','input_bindings','artifacts'):
            for path,digest in result[group].items():
                require(sha(path,tick) == digest, 'Fresh final complete union')
        no_numerics()
        result.update(passed=True,completed=True,pair_vector_gate_passed=True,
            pair_timing_gate_passed=True,own_complete_bills=True,full_model_calls=4,
            actual_completed_samples=True,foreign_numerical_workers_checked=True,
            sources_unchanged=True,hashes_unchanged=True,
            original_negative_fidelity_preserved=True,fresh_objective_required=True)
        observe()
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        try:
            mon.cleanup(children,time.monotonic()+.5); mon.retire_pending(time.monotonic()+.5)
            result.update(whole_owned_tree_retired=True,cleanup_passed=True,observational_groups_retired=True)
        except BaseException:
            result.update(passed=False,completed=False,cleanup_error=traceback.format_exc())
        for handle in (*logs,*readers):
            handle.close()
        try:
            if result['passed']:
                reader = None; observe()
        except BaseException:
            result.update(passed=False,completed=False,final_observation_error=traceback.format_exc())
        stream.close()
        phase_cap = {'build':60.,'native':300.,'remote':180.}.get(phase,CAP)
        publication_deadline = min(previous+1.,began+CAP,last_progress+PROGRESS,phase_started+phase_cap)
        for path in (out/'started.json',out/'samples.jsonl',*(out/(n+'.log') for n in ('build','native','server','remote'))):
            if path.exists():
                result['artifacts'][str(path)] = sha(path)
        if result['passed'] and time.monotonic() >= publication_deadline:
            result.update(passed=False,completed=False,publication_deadline_failed=True)
        result.update(elapsed_s=time.monotonic()-began,no_post_publication_observation_claimed=True,
                      publication_deadline_monotonic_s=publication_deadline)
        write(out/'result.json',result); rh = sha(out/'result.json')
        ok = bool(result['passed'] and time.monotonic() < publication_deadline)
        write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh))
        ch = sha(out/'completion.json')
        write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,
            completion_sha256=ch,elapsed_s=time.monotonic()-began))
        if ok:
            require(time.monotonic() < publication_deadline, 'Final write inside actual1s/whole600s tail')
            deadline()
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','pair_vector_gate_passed',
        'elapsed_s','maximum_combined_rss_bytes','maximum_interval_s','maximum_swap_growth_bytes','error')}),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
