"""Fixed paired full-card RPC runtime experiment; no numerical imports here."""
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
import struct
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
BASE = NEW/'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
BASE_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
CLIENT = NEW/'current_comp_padded_push_d6_v1/client.py'
TRANSPORT = NEW/'current_comp_cstar_proposal_v1/watch.py'
PINS = {BASE: BASE_SHA,
    CLIENT: '46979737953192110b4c70f04d4ea9a8cd106ace217c1d9d4e8c8e6cb29e0de3',
    NEW/'check_remote_series_v1.py': 'f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55',
    TRANSPORT: 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'}
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
FIXTURE = {STORE/'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    STORE/'rows/observer_fit/0037/row_manifest.json': '900e3cc6929bcc81ea301479708fe8bdfb642447f6c8bb1a10981bffacc15571',
    STORE/'rows/observer_fit/0037/weights.npy': '967d989cf0ddf4770f199d50c97b26c70b1129d2d739929e82c8e229b7fd43d4'}
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
CAP, RSS, PROGRESS = 300., 8_000_000_000, 120.
OUT = HERE/'rpc37_v1'
TAGS = ('current_comp_push_d4_runtime_base37_v1','current_comp_push_d4_runtime_new37_v1')
BASE_F = (406399683439,404454713141)
BASE_OUTPUT = 'd9c03845f71fba89763d357314e00b324b5f8446528212fa418002c15cc0c970'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def no_numerics():
    require(not any(k.split('.')[0] in ('numpy','torch','scipy','flopscope',
        'flopscope_server','threadpoolctl') for k in sys.modules), 'Stdlib-only controller')


def sha(path, tick=None):
    path = Path(path)
    require(not path.is_symlink() and path.resolve().is_relative_to(P2)
        and 'truthpost' not in path.parts and 'truth' not in path.name.lower(), 'Regular non-target artifact')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
            if tick:
                tick()
    return h.hexdigest()


def read(path):
    require(not Path(path).is_symlink() and Path(path).stat().st_size < 2_000_000, 'Bounded metadata')
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def load(path, digest, name):
    require(sha(path) == digest, 'Exact reviewed inert helper')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    no_numerics()
    return module


def values(path):
    require(not path.is_symlink() and path.stat().st_size == 65536, 'Exact served all16 FP32 output')
    raw = path.read_bytes(); v = struct.unpack('<16384f',raw)
    require(all(math.isfinite(x) and x >= 0 for x in v), 'All16 finite and nonnegative')
    return raw,v


def foreign_servers(rows, allowed):
    """Add the underscore server-module spelling to the original classifier."""
    found = []
    for row in rows:
        if row['pid'] in allowed:
            continue
        try:
            argv = shlex.split(row['command'])
        except ValueError:
            argv = row['command'].split()
        if argv and re.fullmatch(r'(?:python|pypy)[0-9.]*',Path(argv[0]).name.lower()) and any(
            x == 'flopscope_server' or x.startswith('flopscope_server.') for x in argv[1:]):
            found.append(row)
    return found


def check_row(row, index):
    require(row['index'] == index and row['rank'] == 37 and row['output_shape'] == [16,1024]
        and row['output_dtype'] == 'float32', 'Fixed actual call identity')
    require(type(row['flops']) is int and 0 < row['flops'] <= 2**41
        and row['util'] == row['flops']/2**41, 'Complete own paid bill')
    t = row['timing']
    require(t and all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in t.values())
        and 0 < t['wall_time_s'] < 60 and t['residual_wall_time_s'] < .4, 'Context wall60/residual.4 bounds')
    require(re.fullmatch('[0-9a-f]{64}',row['output_sha256']), 'Actual output hash')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true'); p.add_argument('--numerical-slot-clear',action='store_true')
    for key in ('source','composer','protocol'):
        p.add_argument('--'+key+'-sha256')
    args = p.parse_args()
    if not args.run:
        p.print_help(); return 0
    began = previous = last_progress = time.monotonic()
    require(args.numerical_slot_clear and Path(sys.prefix).resolve() == OFFPY.parent.parent.resolve()
        and os.getpriority(os.PRIO_PROCESS,0) >= 10 and sys.dont_write_bytecode and not sys.flags.optimize
        and all(os.environ.get(k) == '1' for k in BLAS)
        and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
        and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
        and not any(os.environ.get(k) for k in ('PYTHONPATH','PYTHONOPTIMIZE','FLOPSCOPE_SERVER_URL')),
        'Exclusive nice10/assertions/BLAS1/bytecodeoff/fallback0 environment')
    own = {**PINS,HERE/'watch_rpc37.py':args.source_sha256,HERE/'compose.py':args.composer_sha256,
        HERE/'RPC37_PROTOCOL.md':args.protocol_sha256}
    for path,digest in own.items():
        require(re.fullmatch('[0-9a-f]{64}',digest or '') and sha(path) == digest, 'Exact prospective source '+str(path))
    no_numerics()
    h = load(TRANSPORT,PINS[TRANSPORT],'push_runtime_transport'); mon,classifier = h.transport()
    paths = [NEW/f'{tag}{suffix}' for tag in TAGS for suffix in ('.json','_0_r37.f32','_1_r37.f32')]
    require(not OUT.exists() and not any(x.exists() for x in paths) and not HERE.is_symlink(), 'Fresh exclusive stage')
    OUT.mkdir()
    result = dict(schema='current_comp_push_d4_rpc37_v1',passed=False,completed=False,
        source_bindings={str(k):v for k,v in {**own,**h.MON_PINS}.items()},
        input_bindings={str(k):v for k,v in {**FIXTURE,**h.MOCK_PINS}.items()},artifacts={},
        numerical_imports=False,truth_accessed=False,raw_error_measured=False,adjusted_score_measured=False,
        full_domain_selector_qualified=False,native_rpc_candidate_parity_qualified=False,
        submission_authorized=False,parameter_updates=0,information_advances=False,
        whole_cap_s=CAP,progress_cap_s=PROGRESS,combined_rss_limit_bytes=RSS,
        maximum_combined_rss_bytes=0,maximum_swap_growth_bytes=0,maximum_interval_s=0,
        completed_sample_intervals_s=[],commands=[],records=[[],[]],retired_calls=0)
    children,logs,readers = [],[],[]
    reader = None; pending = ''; active = None; initial_swap = None
    stream = (OUT/'samples.jsonl').open('x')

    def deadline():
        now = time.monotonic()
        require(now-began < CAP and now-last_progress < PROGRESS, 'Whole300s/actual-retired120s cap')

    def observe():
        nonlocal previous,last_progress,initial_swap,pending
        absolute = min(previous+1,began+CAP,last_progress+PROGRESS)
        sample = mon.collect(absolute)
        scopes = [mon.scope(sample['rows'],os.getpid(),c.pid) for c in children]
        scopes.append(mon.scope(sample['rows'],os.getpid(),None))
        allowed = set().union(*(s['allowed'] for s in scopes))
        owned = mon.descendants({os.getpid()},sample['rows'])
        owned.update(pid for s in scopes for pid in s['child_pids'])
        combined = sum(r['rss_bytes'] for r in sample['rows'] if r['pid'] in owned)
        require(not classifier(sample['rows'],allowed) and not foreign_servers(sample['rows'],allowed),
            'No foreign numerical worker, including underscore server modules')
        if initial_swap is None:
            initial_swap = sample['swap']['used_bytes']
        growth = sample['swap']['used_bytes']-initial_swap
        require(growth <= 0 and combined < RSS, 'Full owned tree8GB and zero added swap')
        if reader is not None:
            pending += reader.read(); require(len(pending) < 4*1024**2, 'Bounded worker log')
            while '\n' in pending:
                line,pending = pending.split('\n',1)
                try:
                    item = json.loads(line)
                except (ValueError,TypeError):
                    continue
                if isinstance(item,dict) and 'index' in item and 'flops' in item:
                    deadline(); index = len(result['records'][active]); require(index < 2,'Exactly two calls per role')
                    check_row(item,index)
                    raw,_ = values(NEW/f'{TAGS[active]}_{index}_r37.f32')
                    require(hashlib.sha256(raw).hexdigest() == item['output_sha256'], 'Saved output after closed context')
                    result['records'][active].append(item); result['retired_calls'] += 1
                    last_progress = time.monotonic()
                    print(json.dumps(dict(event='retired',role=active,**item)),flush=True)
        deadline()
        result['maximum_combined_rss_bytes'] = max(result['maximum_combined_rss_bytes'],combined)
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began,owned_pids=sorted(owned),
            combined_rss_bytes=combined,swap_bytes=sample['swap']['used_bytes'],retired_calls=result['retired_calls'],
            observational_groups_retired=sample['commands_reaped'],transport=sample['command_records']))+'\n')
        stream.flush(); previous,interval = mon.complete(previous,absolute)
        result['completed_sample_intervals_s'].append(interval)
        result['maximum_interval_s'] = max(result['maximum_interval_s'],interval)

    def tick():
        deadline()
        if time.monotonic()-previous >= .15:
            observe()

    def spawn(argv,name,env,progress=False):
        nonlocal reader,pending
        log = (OUT/(name+'.log')).open('x'); logs.append(log)
        if progress:
            reader = (OUT/(name+'.log')).open('r'); readers.append(reader); pending = ''
        c = subprocess.Popen(argv,cwd=NEW,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(c); result['commands'].append(dict(argv=argv,pid=c.pid,group=c.pid))
        observe(); return c

    def interrupted(*unused):
        raise RuntimeError('Owned fixed runtime component interrupted')

    handlers = {s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        observe()
        for path,digest in FIXTURE.items():
            require(sha(path,tick) == digest,'Same canonical physical37 fixture')
        client_api = load(CLIENT,PINS[CLIENT],'push_runtime_fixed_client')
        client_source,client_proof = client_api.source_and_proof()
        compile(client_source,'<unchanged-fixed-rpc-client>','exec')
        result['client_source_proof'] = client_proof
        composer = load(HERE/'compose.py',args.composer_sha256,'push_runtime_composer')
        text,proof = composer.derive_source(tick)
        require(proof['full_source_inverse_verified'] and proof['parent_sha256'] == BASE_SHA,
            'Exact reviewed D4-only source inverse')
        for path,digest in proof['source_bindings'].items():
            require(sha(path,tick) == digest, 'Actual complete composer source authority')
            result['source_bindings'][str(path)] = digest
        candidate = OUT/'estimator.py'
        with candidate.open('x') as f:
            f.write(text)
        result['candidate_sha256'] = sha(candidate,tick)
        require(result['candidate_sha256'] == proof['candidate_sha256'],'Actual composed candidate bytes')
        result['composition_proof'] = proof; result['artifacts'][str(candidate)] = result['candidate_sha256']
        del text,client_source
        no_numerics(); write(OUT/'started.json',result); tick()
        result['ipc_directories'] = []
        for role,source in enumerate((BASE,candidate)):
            active = role
            scratch = Path(tempfile.mkdtemp(prefix='astra_push_rpc37_',dir='/private/tmp'))
            result['ipc_directories'].append(str(scratch))
            url = 'ipc://'+str(scratch/'flops.sock')
            env = dict(os.environ,PYTHONUNBUFFERED='1',FLOPSCOPE_MAX_ARRAY_BYTES='4294967296',FLOPSCOPE_SERVER_URL=url)
            server = spawn([str(OPENPY),'-B','-u','-m','flopscope_server','--url',url,'--timeout','300'],
                'server_base' if role == 0 else 'server_new',env)
            startup = time.monotonic()
            while not (scratch/'flops.sock').exists():
                require(server.poll() is None and time.monotonic()-startup < 5,'Bounded original IPC startup')
                observe(); time.sleep(.05)
            rel = str(source.relative_to(NEW))
            child = spawn([str(CLIENTPY),'-B','-u',str(CLIENT),'--run','--candidate',rel,'--tag',TAGS[role]],
                'base' if role == 0 else 'new',env,True)
            while child.poll() is None:
                require(server.poll() is None,'Original sole server alive')
                observe(); time.sleep(.10)
            observe(); require(child.wait(timeout=.1) == 0,'Actual client exit0, no retries')
            mon.cleanup([child],time.monotonic()+.5); reader = None; observe()
            series = read(NEW/(TAGS[role]+'.json'))
            require(series['status'] == 'completed' and series['candidate'] == rel
                and series['source_sha256'] == sha(source,tick) and series['records'] == result['records'][role]
                and len(series['records']) == 2 and series['repeated_output_byte_identical'], 'Complete persistent same-source pair')
            mon.cleanup([server],time.monotonic()+.5); observe()
        payloads = [[values(NEW/f'{tag}_{i}_r37.f32') for i in range(2)] for tag in TAGS]
        for role in range(2):
            require(payloads[role][0][0] == payloads[role][1][0],'Exact own repeat all16 bytes')
        base,new = result['records']
        require(tuple(x['flops'] for x in base) == BASE_F
            and all(x['output_sha256'] == BASE_OUTPUT for x in base),'Exact current37 native output/F replay')
        init = [rows[0]['flops']-rows[1]['flops'] for rows in result['records']]
        require(init == [1944970298,1383914298],'Actual complete initialization, no subtraction from served F')
        require(all(abs(n['flops']/b['flops']-1) <= .05 for b,n in zip(base,new,strict=True)), 'D4 complete F within5%')
        drift = [max(abs(x-y) for x,y in zip(payloads[0][0][1][l*1024:(l+1)*1024],
            payloads[1][0][1][l*1024:(l+1)*1024],strict=True)) for l in range(16)]
        require(max(drift) <= 1e-5,'All16 D4 arithmetic fidelity tolerance')
        walls = [[x['timing']['wall_time_s'] for x in rows] for rows in result['records']]
        ratio = sum(walls[1])/sum(walls[0]); advances = max(walls[1]) <= 45 and ratio <= .80
        result.update(layer_max_abs_drift=drift,initialization_flops=init,paired_wall_ratio=ratio,
            maximum_new_wall_s=max(walls[1]),information_advances=advances,own_repeat_byte_exact=True,
            current37_exact_replay=True,own_complete_bills=True,all16_vector_gate_passed=True)
        for path in paths:
            result['artifacts'][str(path)] = sha(path,tick)
        for group in ('source_bindings','input_bindings','artifacts'):
            for path,digest in result[group].items():
                require(sha(path,tick) == digest,'Final source/input/artifact unchanged')
        no_numerics(); result.update(passed=True,completed=True,actual_completed_samples=True,
            foreign_numerical_workers_checked=True,hashes_unchanged=True,sources_unchanged=True)
        observe()
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        try:
            mon.cleanup(children,time.monotonic()+.5); mon.retire_pending(time.monotonic()+.5)
            result.update(cleanup_passed=True,whole_owned_tree_retired=True,observational_groups_retired=True)
            reader = None
            if result['passed']:
                observe()
        except BaseException:
            result.update(passed=False,completed=False,cleanup_error=traceback.format_exc())
        for f in (*logs,*readers):
            f.close()
        stream.close()
        publication = min(previous+1,began+CAP,last_progress+PROGRESS)
        for name in ('started.json','samples.jsonl','server_base.log','server_new.log','base.log','new.log'):
            if (OUT/name).exists():
                result['artifacts'][str(OUT/name)] = sha(OUT/name)
        ok = bool(result['passed'] and time.monotonic() < publication)
        result.update(passed=ok,completed=ok,elapsed_s=time.monotonic()-began,initial_swap_bytes=initial_swap,
            no_post_publication_observation_claimed=True)
        write(OUT/'result.json',result); rh = sha(OUT/'result.json')
        ok = bool(ok and time.monotonic() < publication)
        write(OUT/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh)); ch = sha(OUT/'completion.json')
        ok = bool(ok and time.monotonic() < publication)
        write(OUT/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,
            elapsed_s=time.monotonic()-began,terminal_publication_interval_s=time.monotonic()-previous))
        ok = bool(ok and time.monotonic() < publication)
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','information_advances','retired_calls',
        'paired_wall_ratio','maximum_new_wall_s','elapsed_s','maximum_combined_rss_bytes',
        'maximum_interval_s','maximum_swap_growth_bytes','error','cleanup_error')}),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
