"""Bounded one-local/one-remote scalar diagnostic; no numerical supervisor imports."""
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
NEW, P2 = HERE.parent,HERE.parent.parent
TRANSPORT = NEW/'current_comp_cstar_proposal_v1/watch.py'
TRANSPORT_SHA = 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
OFFPY = P2/'data_gate/.venv/bin/python'
OPENPY = NEW/'openblas_runtime_env_v1/bin/python'
CLIENTPY = P2/'data_gate/.client-venv/bin/python'
CAP, RSS = 60.,2_000_000_000
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def sha(path,tick=None):
    path = Path(path)
    require(not path.is_symlink() and path.is_relative_to(P2),'Regular in-scope source or metadata')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    require(Path(path).stat().st_size < 3*1024**2,'Bounded scalar metadata')
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def load(path,digest,name):
    require(sha(path) == digest,'Reviewed inert source owner')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true')
    ap.add_argument('--numerical-slot-clear',action='store_true')
    for key in ('source','worker','protocol','spec'):
        ap.add_argument('--'+key+'-sha256')
    args = ap.parse_args()
    if not args.run:
        ap.print_help(); return 0
    began = previous = time.monotonic()
    require(args.numerical_slot_clear and Path(sys.prefix).resolve() == OFFPY.parent.parent.resolve()
        and os.getpriority(os.PRIO_PROCESS,0) >= 10 and sys.flags.optimize == 0 and sys.dont_write_bytecode
        and all(os.environ.get(k) == '1' for k in BLAS)
        and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
        and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
        and not os.environ.get('PYTHONPATH') and not os.environ.get('FLOPSCOPE_SERVER_URL'),
        'Exact exclusive nice10/BLAS1 local supervisor environment')
    worker = HERE/'scalar_semantics_probe.py'
    own = {Path(__file__).resolve():args.source_sha256,worker:args.worker_sha256,
        HERE/'SCALAR_SEMANTICS_PROTOCOL.md':args.protocol_sha256,
        HERE/'SCALAR_SEMANTICS_WATCH_SPEC.md':args.spec_sha256,TRANSPORT:TRANSPORT_SHA}
    require(all(re.fullmatch('[0-9a-f]{64}',d or '') and sha(p) == d for p,d in own.items()),
            'Every prospective source hash supplied and exact')
    h = load(TRANSPORT,TRANSPORT_SHA,'scalar_watch_transport')
    mon,classifier = h.transport()
    out,root = HERE/'scalar_semantics_watch_v1',HERE/'scalar_semantics_v1'
    require(not out.exists() and not root.exists() and not HERE.is_symlink(),'Exclusive never-resumed outputs')
    out.mkdir()
    result = dict(schema='current_comp_shared_old384_scalar_semantics_watch_v1',passed=False,completed=False,
        source_bindings={str(p):d for p,d in {**own,**h.MON_PINS}.items()},
        input_bindings={str(p):d for p,d in h.MOCK_PINS.items()},artifacts={},commands=[],
        numerical_imports=False,model_calls=0,truth_accessed=False,candidate_changed=False,
        full_estimator_fidelity_established=False,maximum_combined_rss_bytes=0,
        maximum_swap_growth_bytes=0,maximum_interval_s=0,completed_sample_intervals_s=[])
    children,logs = [],[]
    initial_swap = None
    stream = (out/'samples.jsonl').open('x')
    def deadline():
        require(time.monotonic()-began < CAP,'Whole diagnostic60s, including final publication')
    def observe():
        nonlocal previous,initial_swap
        absolute = min(previous+1.,began+CAP)
        sample = mon.collect(absolute)
        scopes = [mon.scope(sample['rows'],os.getpid(),c.pid) for c in children]
        scopes.append(mon.scope(sample['rows'],os.getpid(),None))
        allowed = set().union(*(s['allowed'] for s in scopes))
        owned = mon.descendants({os.getpid()},sample['rows'])
        owned.update(pid for s in scopes for pid in s['child_pids'])
        combined = sum(r['rss_bytes'] for r in sample['rows'] if r['pid'] in owned)
        require(not classifier(sample['rows'],allowed),'No foreign numerical worker')
        if initial_swap is None:
            initial_swap = sample['swap']['used_bytes']
        growth = sample['swap']['used_bytes']-initial_swap
        require(growth <= 0 and combined < RSS,'Full owned tree2GB/no added swap')
        deadline()
        result['maximum_combined_rss_bytes'] = max(result['maximum_combined_rss_bytes'],combined)
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began,combined_rss_bytes=combined,
            owned_pids=sorted(owned),swap_bytes=sample['swap']['used_bytes'],
            observational_groups_retired=sample['commands_reaped'],transport=sample['command_records']))+'\n')
        stream.flush()
        previous,interval = mon.complete(previous,absolute)
        result['maximum_interval_s'] = max(result['maximum_interval_s'],interval)
        result['completed_sample_intervals_s'].append(interval)
    def tick():
        deadline()
        if time.monotonic()-previous >= .15:
            observe()
    def spawn(argv,name,env):
        log = (out/(name+'.log')).open('x'); logs.append(log)
        child = subprocess.Popen(argv,cwd=NEW,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child)
        result['commands'].append(dict(argv=argv,pid=child.pid,group=child.pid))
        observe()
        return child
    def finish(child,server=None):
        while child.poll() is None:
            if server is not None:
                require(server.poll() is None,'The sole IPC server remains live')
            observe(); time.sleep(.10)
        observe()
        require(child.wait(timeout=.1) == 0,'Mode completed without retry')
        mon.cleanup([child],time.monotonic()+.5)
        observe()
    def verify(mode):
        paths = [root/mode/n for n in ('result.json','completion.json','finalization.json')]
        ds = [sha(p,tick) for p in paths]
        r,e,f = map(read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == ds[0] and f['completion_sha256'] == ds[1]
            and r['mode'] == mode and r['schema'] == 'current_comp_shared_old384_scalar_semantics_v1'
            and r['own_complete_bill'] and r['thirty_expressions_completed'] and r['source_hashes_unchanged']
            and r['model_calls'] == 0 and r['truth_accessed'] is False and r['candidate_changed'] is False
            and r['full_estimator_fidelity_established'] is False and len(r['measurement']['rows']) == 30
            and 0 < r['measurement']['complete_flops'] < 1_000_000
            and 0 < r['peak_rss_bytes'] < 1_000_000_000 and 0 <= r['elapsed_s'] < 60
            and 0 <= f['elapsed_s'] < 60,'Actual complete positive mode, not a score')
        for path,digest in r['source_bindings'].items():
            require(sha(Path(path),tick) == digest,'Every actual numerical owner remains frozen')
            require(path not in result['source_bindings'] or result['source_bindings'][path] == digest,
                    'No conflicting owner')
            result['source_bindings'][path] = digest
        result['artifacts'].update({str(p):d for p,d in zip(paths,ds)})
        return r
    def interrupted(*unused):
        raise RuntimeError('Scalar diagnostic interrupted')
    handlers = {s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        observe()
        api = load(worker,args.worker_sha256,'scalar_probe_metadata_api')
        for path,digest in api.PINS.items():
            require(sha(path,tick) == digest,'Exact primitive numerical source pin')
            result['source_bindings'][str(path)] = digest
        require(not api.numerical_modules(),'Supervisor remains numerical-import free')
        write(out/'started.json',result); tick()
        argv_tail = ['-B','-u',str(worker),'--run','--source-sha256',args.worker_sha256,
            '--protocol-sha256',args.protocol_sha256]
        env = dict(os.environ,PYTHONUNBUFFERED='1')
        local_child = spawn([str(OPENPY),*argv_tail,'--mode','local'],'local',env)
        finish(local_child); local = verify('local')
        scratch = Path(tempfile.mkdtemp(prefix='astra_scalar_semantics_',dir='/private/tmp'))
        result['ipc_directory'] = str(scratch)
        url = 'ipc://'+str(scratch/'flops.sock')
        remote_env = dict(env,FLOPSCOPE_SERVER_URL=url)
        server = spawn([str(OPENPY),'-B','-u','-m','flopscope_server','--url',url,'--timeout','60'],
            'server',remote_env)
        server_start = time.monotonic()
        while not (scratch/'flops.sock').exists():
            require(server.poll() is None and time.monotonic()-server_start < 5,'Bounded IPC startup')
            observe(); time.sleep(.05)
        client = spawn([str(CLIENTPY),*argv_tail,'--mode','remote','--server-url',url],'remote',remote_env)
        finish(client,server)
        mon.cleanup([server],time.monotonic()+.5); observe()
        remote = verify('remote')
        result['comparison'] = api.compare(local,remote); tick()
        for group in ('source_bindings','input_bindings','artifacts'):
            for path,digest in result[group].items():
                require(sha(Path(path),tick) == digest,'Fresh complete final inventory')
        require(not api.numerical_modules(),'No numerical supervisor imports through comparison')
        result.update(passed=True,completed=True,operational_passed=True,
            scalar_hypothesis_supported=result['comparison']['all_predicted_expression_tariffs_observed']
                and result['comparison']['slice_control_all_intermediate_bytes_exact'],
            sources_unchanged=True,hashes_unchanged=True,actual_completed_samples=True)
        observe()
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        try:
            mon.cleanup(children,time.monotonic()+.5)
            mon.retire_pending(time.monotonic()+.5)
            result.update(whole_owned_tree_retired=True,cleanup_passed=True,observational_groups_retired=True)
        except BaseException:
            result.update(passed=False,completed=False,cleanup_error=traceback.format_exc())
        for handle in logs:
            handle.close()
        if result['passed']:
            observe()
        result['elapsed_s'] = time.monotonic()-began
        if result['elapsed_s'] >= CAP:
            result.update(passed=False,completed=False)
        write(out/'result.json',result); rh = sha(out/'result.json')
        ok = bool(result['passed'] and time.monotonic()-began < CAP)
        write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh))
        ch = sha(out/'completion.json')
        write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,
            completion_sha256=ch,elapsed_s=time.monotonic()-began))
        if ok:
            observe(); deadline()
        stream.close()
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','scalar_hypothesis_supported',
        'elapsed_s','maximum_combined_rss_bytes','maximum_interval_s','maximum_swap_growth_bytes','error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
