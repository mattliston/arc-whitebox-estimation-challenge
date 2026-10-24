"""Fixed push832 probe and independent stdlib supervisor; inert without --run.

The private --measure mode is launched once in a fresh owned process group.
Only it imports NumPy. Its client/server roles run serially in that group.
"""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
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
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
GATE, OUT = P2/'data_gate', HERE/'run_v1'
OLD = NEW/'current_d50_serial_d6_kernel_v1'
CENSUS = NEW/'current_d50_d6_cost_census_v1'
P1 = NEW/'current_d50_native_objective_p1_v1'
SOURCE = NEW/'birth_direction_fit_v1/build_terminal_v1/estimator.py'
KERNEL, LAYOUT = OLD/'kernel.py', NEW/'packed_arena_generic.py'
HELPER = NEW/'current_d50_terminal_linear_v2/watch_fit.py'
KERNEL_SHA = 'a8d1cb1a972b37afd9eaaa67acf820cba3196dabea10ba1058bb3409c0b20ec7'
LAYOUT_SHA = '4e4a3dea533c222f34ba3a08538245414e149dbc0f918fd097267b0f869fb27e'
HELPER_SHA = 'b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd'
SHAPE, SEED, ROLES = (1024,832,8192), 9309094, ('d4','breadth_d6','serial_d6')
RSS_LIMIT, WHOLE_LIMIT, ROLE_LIMIT = 8_000_000_000, 420.0, 120.0
PERIOD, MAX_INTERVAL, EXTRA_BINDINGS = .2, 1.0, 561056000
ENV = dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
    MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='0',
    FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
PINS = {
    SOURCE:'d50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451',
    P1/'build_v1/estimator.py':'0deb237350e753220b575796c0278f86884b8461bfeec4ab53eae9c15522c435',
    KERNEL:KERNEL_SHA,LAYOUT:LAYOUT_SHA,HELPER:HELPER_SHA,
    OLD/'client.py':'d1cb99f30149e5b4350db4df81983fdbea0fb0099be88a9015de3fb267b27899',
    OLD/'run.py':'961ab14418c109b0e85ffb665795fb83f50dbce21a90c3cc63dadde4d6c7aabc',
    OLD/'PROTOCOL.md':'888674b8ebfb4fe3f0e1caa071d9528d69c45c880967ffbbb827167536212477',
    OLD/'RESULT.md':'d87c5d29e8eb37a8192f19902c9e92298458435da5d0816e396febdfa0814125',
}
INPUT_PINS = {
    OLD/'run_v1/result.json':'86ce7e8b1ea3d9dcc81c42bad6cb78828b520f38aed5f4918acf6052a6cf8660',
    OLD/'run_v1/completion.json':'caafbeb845cd7277e9c408e4ef824759a8bc521831dd7150c356882e03cf169e',
    CENSUS/'price.json':'995bd57f77d526cee191ec407473946c1d3116ed7390754ba635485458c76c72',
    CENSUS/'native/result.json':'d3d080115b8b27ad5af26196d045c3324db2c5c44496357b2ab8f4ec658666c2',
    CENSUS/'native/completion.json':'a1898f0b068ec74cd16250e71b20bb10bcdbc9bd6e53bf759ec277cd9f253496',
    CENSUS/'native/call_00_tiles.json':'42bf244f40c1e9964b9a722253f06d951240c1e06c483d709a146119ff257a79',
    P1/'build_v1/result.json':'877990bc84cb85431ff98f83e1e7012517c8b7914a1e0b39b29bf7d0b776b0c1',
    P1/'build_v1/completion.json':'cdcb0a934224f242cad404dec3bae7eba60a712d8bd430ace0c7ab7111faea5c',
}


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def module(path,name,digest):
    require(sha(path)==digest,'Frozen source changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def verify(mapping,observe=None):
    for path,digest in mapping.items():
        require(Path(path).is_absolute() and sha(path)==digest,'Unchanged bound file: '+path)
        if observe is not None:
            observe()


def frame():
    require(Path(sys.prefix).resolve()==(GATE/'.venv').resolve() and sys.flags.optimize==0,
            'Fixed OFFPY/assertions frame')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'Inherited nice10 required')
    require(all(os.environ.get(key)==value for key,value in ENV.items()),'Explicit fixed launch environment')


def authenticate(args):
    """Source/receipt metadata only; no numerical libraries or array files."""
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')),'Authenticate before numerical imports')
    for key in ('source_sha256','client_sha256','protocol_sha256'):
        require(re.fullmatch('[0-9a-f]{64}',getattr(args,key) or ''),'Reviewed hash required: '+key)
    sources = {str(path):digest for path,digest in PINS.items()}
    sources.update({str(Path(__file__).resolve()):args.source_sha256,
        str(HERE/'client.py'):args.client_sha256,str(HERE/'PROTOCOL.md'):args.protocol_sha256})
    inputs = {str(path):digest for path,digest in INPUT_PINS.items()}
    verify(sources); verify(inputs)
    prior,completion = read(OLD/'run_v1/result.json'),read(OLD/'run_v1/completion.json')
    require(prior['passed'] is True and prior['completed'] is True
            and prior['single_product_qualification_passed'] is False and prior['timing_gate_passed'] is False
            and completion['passed'] is True and completion['result_sha256']==inputs[str(OLD/'run_v1/result.json')],
            'Preserve old completed, nonadvancing read test')
    price = read(CENSUS/'price.json')
    require(price['passed'] is True and price['completed'] is True,'Sealed actual d50 census price')
    tiles = read(CENSUS/'native/call_00_tiles.json')
    require(sum(tuple(t['shape'])==SHAPE for t in tiles)==6
            and not any(tuple(t['shape'])==(1024,8192,1024) for t in tiles),'New occurring-workload premise')
    group = [t for t in price['records'][0]['tiles'] if tuple(t['shape'])==SHAPE]
    require(len(group)==6 and all(t['flops']==8417538304 and t['hypothetical_d6_flops']==7062260240
            and t['additional_input_padding_flops']==0 for t in group),'Exact six-call source-priced group')
    build,end = read(P1/'build_v1/result.json'),read(P1/'build_v1/completion.json')
    require(build['passed'] is True and build['completed'] is True and end['passed'] is True
            and end['result_sha256']==inputs[str(P1/'build_v1/result.json')],'Frozen completed P1 literal build')
    kernel = module(KERNEL,'push832_frozen_kernel_audit',KERNEL_SHA)
    d50,p1,layout_text = SOURCE.read_text(),(P1/'build_v1/estimator.py').read_text(),LAYOUT.read_text()
    audit = kernel.audit_sources(d50,layout_text,KERNEL.read_text())
    def active_functions(text):
        return {node.name:ast.get_source_segment(text,node) for node in ast.parse(text).body
                if isinstance(node,ast.FunctionDef)}
    d50_functions,p1_functions = active_functions(d50),active_functions(p1)
    require(all(d50_functions[name]==p1_functions[name] for name in kernel.HELPER_NAMES),
            'Exact same three active packed helper bodies in P1')
    require(kernel.price(*SHAPE,4)==8417538304 and kernel.price(*SHAPE,6)==7062260240,
            'Frozen exact per-product tariff')
    audit.update(P1_helper_source_byte_exact=True,MLP_executed=False,numerical_execution=False)
    return sources,inputs,audit


@contextmanager
def phase_limit(seconds):
    def expired(*unused):
        raise TimeoutError('Fixed numerical preparation/judgment/role deadline')
    old = signal.signal(signal.SIGALRM,expired)
    began = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        yield
        require(time.monotonic()-began < seconds,'Completed phase exceeded fixed wall')
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old)


def stop_child(process):
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=3)
    else:
        process.wait(timeout=1)


def run_role(role,contract_sha,q):
    """Children deliberately inherit the externally supervised worker group."""
    scratch = Path(tempfile.mkdtemp(prefix='arc_p1_push832_',dir='/private/tmp'))
    url = f'ipc://{scratch}/flops.sock'
    env = dict(os.environ,**ENV,FLOPSCOPE_SERVER_URL=url)
    began = time.monotonic()
    record = dict(role=role,passed=False,completed=False,ipc_directory=str(scratch),
        inherited_process_group=os.getpgrp(),driver_numerically_idle=True,resources_owned_by_external_watch=True)
    server = client = None
    try:
        with phase_limit(ROLE_LIMIT), (OUT/f'{role}_server.log').open('xb') as server_log, \
                (OUT/f'{role}_client.log').open('xb') as client_log:
            server = subprocess.Popen([str(NEW/'openblas_runtime_env_v1/bin/python'),'-B','-m',
                'flopscope_server','--url',url,'--timeout','150'],stdin=subprocess.DEVNULL,
                env=env,stdout=server_log,stderr=subprocess.STDOUT)
            record['server_pid'] = server.pid
            ready = time.monotonic()+5
            while not (scratch/'flops.sock').exists():
                require(server.poll() is None and time.monotonic()<ready,'Server startup failed')
                time.sleep(.05)
            client = subprocess.Popen([str(GATE/'.client-venv/bin/python'),'-B','-u',str(HERE/'client.py'),
                '--role',role,'--contract-sha256',contract_sha],stdin=subprocess.DEVNULL,
                env=env,cwd=HERE,stdout=client_log,stderr=subprocess.STDOUT)
            record['client_pid'] = client.pid
            while client.poll() is None:
                require(server.poll() is None and time.monotonic()-began<ROLE_LIMIT,'Fixed role deadline/server')
                time.sleep(.1)
            record['client_exit_code'] = client.returncode
            require(client.returncode==0,'Product client failed; preserve its result/log')
            stop_child(server)
            rows = q.process_table()
            survivors = q.descendants({server.pid,client.pid},rows)
            require(not any(pid in survivors for pid,parent,group,rss in rows),'Role left descendants alive')
            record.update(passed=True,completed=True,owned_pair_retired=True)
    except BaseException:
        record.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        for process in (client,server):
            try:
                stop_child(process)
            except BaseException:
                record.update(passed=False,completed=False,cleanup_error=traceback.format_exc())
        record['wall_s'] = time.monotonic()-began
        if record['wall_s']>=ROLE_LIMIT:
            record.update(passed=False,completed=False,deadline_failed=True)
        write(OUT/f'{role}_resources.json',record)
    require(record['passed'],'Fixed role failed: '+role)
    return record


def finite_summary(summary):
    require(type(summary['flops_used']) is int and 0<=summary['flops_used']<=2**39,'Complete finite context bill')
    for key in ('wall_time_s','flopscope_backend_time_s','flopscope_overhead_time_s','residual_wall_time_s'):
        require(math.isfinite(summary[key]) and summary[key]>=0,'Finite nonnegative context timing')
    require(summary['wall_time_s']<30,'Original per-context30s bound')


def judge(np,kernel,contract,resources):
    roles = {}
    reference = np.load(OUT/'reference_f64.npy',allow_pickle=False)
    require(reference.dtype==np.float64 and reference.shape==(SHAPE[0],SHAPE[2])
            and np.isfinite(reference).all(),'Synthetic FP64 reference')
    layout = module(LAYOUT,'push832_judge_layout',LAYOUT_SHA).layout
    binding4 = sum(stop-start for shape in ((1024,1024,8192),(1024,8192,1024))
                   for start,stop,dims in layout(*shape,4)[1])
    binding6 = sum(stop-start for shape in ((1024,1024,8192),(1024,8192,1024))
                   for start,stop,dims in layout(*shape,6)[1])
    for role in ROLES:
        folder = OUT/role
        result,end = read(folder/'result.json'),read(folder/'completion.json')
        require(result['passed'] is True and result['completed'] is True and result['hashes_unchanged'] is True
                and end['passed'] is True and end['completed'] is True
                and end['result_sha256']==sha(folder/'result.json'),'Complete role seal')
        require(result['role']==role and result['shape']==list(SHAPE)
                and result['contract_sha256']==sha(OUT/'contract.json') and resources[role]['passed'],
                'Exact same fixture and resource ownership')
        verify(result['artifacts'])
        require(result['source_bindings']==contract['source_bindings']
                and result['input_bindings']==contract['input_bindings'],'Exact original bound fixture/source inventory')
        require([t['call'] for t in result['trials']]==[0,1,2],'One cold/two warm products')
        finite_summary(result['setup'])
        expected_setup = binding4+(0 if role=='d4' else EXTRA_BINDINGS if role=='serial_d6' else binding6)
        require(result['setup']['flops_used']==expected_setup,'Entire measured setup equals all paid bindings')
        expected = (dict(add=28,subtract=32,stack=8,copyto=4,matmul=1) if role=='d4' else
                    dict(add=42,subtract=48,stack=12,copyto=6,matmul=1) if role=='breadth_d6'
                    else kernel.WARM_PRIMITIVE_CALLS)
        depth = 4 if role=='d4' else 6
        for call in result['trials']:
            require(read(folder/f"call_{call['call']:02d}_bill.json")==call,'Own complete saved call bill')
            summary = call['summary']
            finite_summary(summary); finite_summary(call['verification'])
            require(summary['flops_used']==kernel.price(*SHAPE,depth),'Exact cold/warm product tariff')
            require(all(summary['operations'][key]['calls']==count for key,count in expected.items()),
                    'Exact arithmetic/bulk-copy primitive counts')
            require(call['output_dtype']=='float32' and call['output_shape']==[SHAPE[0],SHAPE[2]],
                    'Actual recorded output shape/dtype')
            path = folder/f"call_{call['call']:02d}.f32"
            require(call['output']==str(path) and sha(path)==call['output_sha256']
                    and path.stat().st_size==4*SHAPE[0]*SHAPE[2],'Own exact output bytes')
            array = np.fromfile(path,dtype=np.float32).reshape(SHAPE[0],SHAPE[2])
            require(np.isfinite(array).all(),'Finite FP32 product')
            error = array.astype(np.float64)-reference
            call['fp64_reference_error'] = dict(rms=float(np.sqrt(np.mean(error*error))),
                                               maximum=float(np.max(np.abs(error))))
            require(all(math.isfinite(x) for x in call['fp64_reference_error'].values()),'Finite reference errors')
            del array,error
        require(len({t['output_sha256'] for t in result['trials']})==1,'Exact own product repeats')
        result['complete_probe_flops'] = result['setup']['flops_used']+sum(
            t['summary']['flops_used']+t['verification']['flops_used'] for t in result['trials'])
        roles[role] = result
    require(roles['serial_d6']['trials'][0]['output_sha256']==roles['breadth_d6']['trials'][0]['output_sha256'],
            'Serial/breadthD6 exact bytes')
    extra = roles['serial_d6']['setup']['flops_used']-roles['d4']['setup']['flops_used']
    require(extra==EXTRA_BINDINGS,'Measured both-layout extra cold binding cost')
    d4_warm = min(t['summary']['wall_time_s'] for t in roles['d4']['trials'][1:])
    serial_warm = max(t['summary']['wall_time_s'] for t in roles['serial_d6']['trials'][1:])
    setup_delta = max(0.0,roles['serial_d6']['setup']['wall_time_s']-roles['d4']['setup']['wall_time_s'])
    warm_delta = max(0.0,serial_warm-d4_warm)
    incremental = setup_delta+6*warm_delta
    residual = max(t['summary']['residual_wall_time_s'] for t in roles['serial_d6']['trials'])
    timing = incremental<=8.0 and residual<=.20
    write(OUT/'judged_roles.json',roles)
    return dict(numerical_and_billing_gate_passed=True,serial_breadth_bytes_equal=True,
        timing_gate_passed=timing,partial_shape_test_admitted=timing,
        d4_min_warm_context_wall_s=d4_warm,serial_max_warm_context_wall_s=serial_warm,
        delta_setup_s=setup_delta,delta_warm_s=warm_delta,incremental_six_call_wall_s=incremental,
        incremental_wall_gate_s=8.0,max_serial_residual_s=residual,
        first_cold_frame_serial_minus_d4_s=roles['serial_d6']['trials'][0]['summary']['wall_time_s']-
            roles['d4']['trials'][0]['summary']['wall_time_s'],
        additional_serial_setup_flops=extra,all_setup_and_verification_bills_recorded=True,
        full_P1_test_automatically_launched=False,whole_estimator_qualified=False,raw_MSE_qualified=False,
        expected_P1_mean_saving=None,score_measured=False,old_read_failure_preserved=True,
        old_whole_wall_reference_s=43.692628208,conditional_reference_plus_increment_s=43.692628208+incremental,
        timing_interpretation='Admission heuristic only; not a whole-card or official-worker bound',
        role_results={role:dict(result_sha256=sha(OUT/role/'result.json'),
            complete_probe_flops=value['complete_probe_flops'],
            fp64_reference_error=value['trials'][0]['fp64_reference_error']) for role,value in roles.items()})


def measure(args):
    frame()
    require(not any(key in sys.modules for key in ('numpy','torch','flopscope')),'Private worker admits before numerical imports')
    admission_path = OUT/'admission.json'
    require(sha(admission_path)==args.admission_sha256,'Externally sealed one-shot admission')
    admission = read(admission_path)
    verify(admission['source_bindings']); verify(admission['input_bindings'])
    require(os.getpgrp()==os.getpid() and admission['shape']==list(SHAPE),'Fresh owned measurement group/fixed shape')
    q = module(HELPER,'push832_process_helpers',HELPER_SHA)
    kernel = module(KERNEL,'push832_measure_kernel',KERNEL_SHA)
    record = dict(schema='current_p1_d6_push832_measurement_v1',passed=False,completed=False,
        hashes_unchanged=False,source_bindings=admission['source_bindings'],input_bindings=admission['input_bindings'],
        admission_sha256=args.admission_sha256,shape=list(SHAPE),resources={},artifacts={},
        truth_accessed=False,MLP_executed=False,training_performed=False,source_mutation=False)
    write(OUT/'measurement_started.json',record)
    try:
        with phase_limit(60):
            import numpy as np
            import importlib.metadata
            from threadpoolctl import threadpool_info,threadpool_limits
            require(np.__version__=='2.4.6' and importlib.metadata.version('flopscope')=='0.12.0',
                    'Pinned offline numerical frame')
            limiter = threadpool_limits(limits=1)
            record['driver_threadpools'] = threadpool_info()
            require(all(p['num_threads']==1 for p in record['driver_threadpools'] if p.get('user_api')=='blas'),
                    'One-thread offline reference')
            rng = np.random.default_rng(SEED)
            left = rng.standard_normal((SHAPE[0],SHAPE[1]),dtype=np.float32)/np.float32(math.sqrt(SHAPE[1]))
            right = rng.standard_normal((SHAPE[1],SHAPE[2]),dtype=np.float32)
            reference = left.astype(np.float64)@right.astype(np.float64)
            for name,array in (('left.npy',left),('right.npy',right),('reference_f64.npy',reference)):
                with (OUT/name).open('xb') as stream:
                    np.save(stream,array)
            del left,right,reference,array,rng
            gc.collect()
            fixtures = {str(OUT/name):sha(OUT/name) for name in ('left.npy','right.npy','reference_f64.npy')}
            contract = dict(source_bindings=admission['source_bindings'],
                input_bindings={**admission['input_bindings'],**fixtures},source_audit=admission['source_audit'],
                env=ENV,shape=list(SHAPE),seed=SEED,measurement_process_group=os.getpgrp(),
                truth_accessed=False,MLP_executed=False,old_max_read_shape_used=False)
            write(OUT/'contract.json',contract)
        digest = sha(OUT/'contract.json')
        for role in ROLES:
            record['resources'][role] = run_role(role,digest,q)
        with phase_limit(60):
            record.update(judge(np,kernel,contract,record['resources']))
            verify(contract['source_bindings']); verify(contract['input_bindings'])
            limiter.restore_original_limits()
            paths = [OUT/'measurement_started.json',OUT/'contract.json',OUT/'judged_roles.json',
                     *(OUT/name for name in ('left.npy','right.npy','reference_f64.npy'))]
            for role in ROLES:
                paths.extend(sorted((OUT/role).iterdir()))
                paths.extend(OUT/f'{role}_{suffix}' for suffix in ('server.log','client.log','resources.json'))
            record['artifacts'] = {str(path):sha(path) for path in paths if path.is_file()}
            record.update(passed=True,completed=True,hashes_unchanged=True)
    except BaseException:
        record.update(passed=False,completed=False,hashes_unchanged=False,error=traceback.format_exc())
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write(OUT/'measurement_result.json',record)
    write(OUT/'measurement_completion.json',dict(passed=record['passed'],completed=record['completed'],
                                               result_sha256=sha(OUT/'measurement_result.json')))
    return 0 if record['passed'] else 1


def supervise(args):
    began = time.monotonic()
    frame()
    require(args.exclusive_slot_clear,'Root must check process inventory and attest the sole numerical slot')
    sources,inputs,audit = authenticate(args)
    require(not OUT.exists() and not OUT.is_symlink() and not HERE.is_symlink(),'Exclusive probe output/no resume')
    OUT.mkdir(exist_ok=False)
    admission = dict(schema='current_p1_d6_push832_admission_v1',source_bindings=sources,
        input_bindings=inputs,source_audit=audit,shape=list(SHAPE),seed=SEED,roles=list(ROLES),
        exclusive_slot_clear_attested=True,MLP_executed=False,truth_accessed=False)
    write(OUT/'admission.json',admission)
    q = module(HELPER,'push832_watch_helpers',HELPER_SHA)
    result = dict(schema='current_p1_d6_push832_probe_v1',passed=False,completed=False,
        hashes_unchanged=False,sources_unchanged=False,partial_shape_test_admitted=False,
        source_bindings=sources,input_bindings=inputs,artifacts={},shape=list(SHAPE),
        exclusive_slot_clear_attested=True,whole_wall_limit_s=WHOLE_LIMIT,combined_rss_limit_bytes=RSS_LIMIT,
        swap_growth_limit_bytes=0,maximum_sample_interval_s=0.0,maximum_sampled_rss_bytes=0,
        maximum_swap_growth_bytes=0.0,requested_sample_period_s=PERIOD,
        numerical_driver_in_supervisor=False,truth_accessed=False,MLP_executed=False,
        started_utc=datetime.now(timezone.utc).isoformat(),samples=0,
        resource_scope='Sampled supervisor plus complete owned measurement/client/server group; not a cgroup peak')
    child = None
    handlers = {}
    last = time.monotonic()
    log = None
    def interrupted(signum,unused):
        raise RuntimeError('Supervisor interrupted: '+str(signum))
    def observe():
        nonlocal last
        item = q.memory_snapshot(child.pid if child is not None else os.getpid())
        sampled = time.monotonic()
        interval,last = sampled-last,sampled
        swap = q.swap_state()
        now = time.monotonic()
        growth = swap['used_bytes']-result['before_swap']['used_bytes']
        item.update(elapsed_s=now-began,interval_s=interval,swap=swap,swap_growth_bytes=growth)
        result['samples'] += 1
        result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],item['combined_rss_bytes'])
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        if log is not None:
            log.write(json.dumps(item,allow_nan=False)+'\n'); log.flush()
        require(now-began<WHOLE_LIMIT and interval<=MAX_INTERVAL,'Whole deadline or resource sample gap')
        require(item['combined_rss_bytes']<RSS_LIMIT,'Strict8GB combined full-tree RSS')
        require(growth<=0,'Zero added system swap')
        return item
    try:
        for signum in (signal.SIGINT,signal.SIGTERM):
            handlers[signum] = signal.signal(signum,interrupted)
        result['before_swap'] = q.swap_state()
        result['process_inventory_before_launch'] = q.process_table()
        write(OUT/'started.json',result)
        command = [str(GATE/'.venv/bin/python'),'-B','-u',str(Path(__file__).resolve()),'--measure',
                   '--admission-sha256',sha(OUT/'admission.json')]
        result['command'] = command
        with (OUT/'stdout.log').open('xb') as stdout, (OUT/'stderr.log').open('xb') as stderr, \
                (OUT/'resources.jsonl').open('x') as resources:
            log = resources
            observe()
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                env=dict(os.environ,**ENV),cwd=HERE,start_new_session=True)
            result['child_pid'] = result['child_process_group'] = child.pid
            while True:
                code = child.poll()
                sample = observe()
                if code is not None:
                    result['child_exit_code'] = code
                    require(not sample['child_pids'],'Measurement worker left its owned group alive')
                    break
                require(sample['child_pids'] or child.poll() is not None,'Live numerical group must be observable')
                time.sleep(PERIOD)
            require(code==0,'Measurement failed; preserve failure and stop')
            rp,cp = OUT/'measurement_result.json',OUT/'measurement_completion.json'
            record,end = read(rp),read(cp)
            require(record['passed'] is True and record['completed'] is True and record['hashes_unchanged'] is True
                    and end['passed'] is True and end['completed'] is True and end['result_sha256']==sha(rp),
                    'Complete measurement result seal')
            require(record['source_bindings']==sources and record['input_bindings']==inputs
                    and record['shape']==list(SHAPE) and record['truth_accessed'] is False
                    and list(record['resources'])==sorted(ROLES),'Exact role/source/input scope')
            verify(record['artifacts'],observe)
            verify(sources,observe); verify(inputs,observe)
            result['artifacts'].update(record['artifacts'])
            result.update(measurement_result_sha256=sha(rp),measurement_completion_sha256=sha(cp),
                measurement_completed=True,whole_owned_tree_retired=True,
                numerical_and_billing_gate_passed=record['numerical_and_billing_gate_passed'],
                timing_gate_passed=record['timing_gate_passed'],
                partial_shape_test_admitted=record['partial_shape_test_admitted'],
                delta_setup_s=record['delta_setup_s'],delta_warm_s=record['delta_warm_s'],
                incremental_six_call_wall_s=record['incremental_six_call_wall_s'],
                additional_serial_setup_flops=record['additional_serial_setup_flops'],
                max_serial_residual_s=record['max_serial_residual_s'],role_results=record['role_results'],
                whole_estimator_qualified=False,raw_MSE_qualified=False,expected_P1_mean_saving=None,
                score_measured=False,automatic_next_stage=False)
            observe()
            result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True)
        log = None
    except BaseException:
        log = None
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
                      partial_shape_test_admitted=False,error=traceback.format_exc())
    finally:
        if child is not None:
            try:
                q.stop_owned_group(child)
                result['whole_owned_group_cleanup_complete'] = True
            except BaseException:
                result.update(passed=False,completed=False,partial_shape_test_admitted=False,
                              cleanup_error=traceback.format_exc())
        try:
            for path in (OUT/'admission.json',OUT/'started.json',OUT/'stdout.log',OUT/'stderr.log',
                         OUT/'resources.jsonl',OUT/'measurement_result.json',OUT/'measurement_completion.json'):
                if path.is_file():
                    result['artifacts'][str(path)] = sha(path)
            verify(sources)
            if result['passed']:
                result['final_resource_sample_after_hashing'] = observe()
        except BaseException:
            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
                          partial_shape_test_admitted=False,final_sealing_error=traceback.format_exc())
        for signum,handler in handlers.items():
            signal.signal(signum,handler)
        result.update(elapsed_s=time.monotonic()-began,finished_utc=datetime.now(timezone.utc).isoformat())
        write(OUT/'result.json',result)
        write(OUT/'completion.json',dict(passed=result['passed'],completed=result['completed'],
                                       result_sha256=sha(OUT/'result.json')))
    print(json.dumps({key:result.get(key) for key in ('passed','completed','partial_shape_test_admitted',
        'incremental_six_call_wall_s','maximum_sampled_rss_bytes','maximum_swap_growth_bytes','elapsed_s','error')},
        allow_nan=False),flush=True)
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--run',action='store_true')
    modes.add_argument('--measure',action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--client-sha256')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--admission-sha256')
    parser.add_argument('--exclusive-slot-clear',action='store_true')
    args = parser.parse_args()
    if args.measure:
        return measure(args)
    if not args.run:
        parser.print_help(); return 0
    return supervise(args)


if __name__=='__main__':
    raise SystemExit(main())
