"""DRAFT: fixed synthetic PUSH-kernel fixtures, RPC roles and saved-output judge.

Inert without --run. No MLP, target, head, candidate build, score or submission.
The separately supervised modes never run concurrently with another role.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback
import weakref

HERE=Path(__file__).resolve().parent
NEW,P2=HERE.parent,HERE.parent.parent
ROOT=HERE/'run_v1'
OPENBLAS=NEW/'openblas_runtime_env_v1'
CLIENT=P2/'data_gate/.client-venv'
OSITE=OPENBLAS/'lib/python3.11/site-packages'
CSITE=CLIENT/'lib/python3.11/site-packages'
BACKEND_OWNER=NEW/'current_comp_shared_pair256_v1/tiny_worker.py'
BACKEND_SHA='c6afc327c9acb6a9cd6ba3ed66ef293f8a05d9a22e9031b97ffe9cec49d42e1d'
KERNEL_OWNER=NEW/'current_d50_serial_d6_kernel_v1/kernel.py'
KERNEL_OWNER_SHA='a8d1cb1a972b37afd9eaaa67acf820cba3196dabea10ba1058bb3409c0b20ec7'
CLIENT_OWNER=NEW/'current_d50_serial_d6_kernel_v1/client.py'
CLIENT_OWNER_SHA='d1cb99f30149e5b4350db4df81983fdbea0fb0099be88a9015de3fb267b27899'
SCHEMA='current_comp_push_microbatch3_kernel_v1'
SHAPE=(1024,384,8192)
ROLES=('d4','serial_d6','microbatch3')
SEED=9309094
KEYS=('source','kernel','client','watcher','protocol','watch-spec')
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
         'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
_BACKEND=None


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path,tick=None):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(P2)
        and 'truthpost' not in path.parts and 'weights.npy'!=path.name
        and not (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),
        'Only source/metadata and own synthetic fixtures, no network or truth arrays')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    path=Path(path)
    require(path.suffix=='.json' and not path.is_symlink() and path.stat().st_size<16*1024**2,'Bounded kernel JSON')
    return json.loads(path.read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    return sha(path)


def load(path,digest,name):
    require(sha(path)==digest,'Exact source owner before import')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND=load(BACKEND_OWNER,BACKEND_SHA,'micro3_qualified_backend_metadata')
    return _BACKEND


def numerical_modules():
    return sorted(k for k in sys.modules if k.split('.')[0] in
        ('numpy','scipy','torch','flopscope','flopscope_server','threadpoolctl'))


def owner_map(args):
    b=backend()
    numerical={p:d for p,d in b.PINS.items() if p.is_relative_to(OSITE) or p.is_relative_to(CSITE)}
    return {**numerical,BACKEND_OWNER:BACKEND_SHA,KERNEL_OWNER:KERNEL_OWNER_SHA,CLIENT_OWNER:CLIENT_OWNER_SHA,
        b.PARENT:b.PARENT_SHA,HERE/'worker.py':args.source_sha256,HERE/'kernel.py':args.kernel_sha256,
        HERE/'client.py':args.client_sha256,HERE/'watch.py':args.watcher_sha256,
        HERE/'PROTOCOL.md':args.protocol_sha256,HERE/'WATCH_SPEC.md':args.watch_spec_sha256}


def admit(args,tick=None):
    require(not numerical_modules(),'Source/metadata admission before numerical imports')
    owners=owner_map(args)
    for path,digest in owners.items():
        require(re.fullmatch('[0-9a-f]{64}',digest or '') and sha(path,tick)==digest,'Exact kernel qualification source')
    kernel=load(HERE/'kernel.py',args.kernel_sha256,'micro3_inert_kernel')
    bundle,literals,proof=kernel.source_bundle();changed=kernel.child_source(bundle)
    proof.update(batch_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        original_complete_child_and_top_inverses=True,physical_terms_or_heads_changed=False,
        no_depth_or_shape_policy_search=True,original_READ_kernel_unchanged=True)
    old=load(KERNEL_OWNER,KERNEL_OWNER_SHA,'micro3_original_kernel_price')
    expected={r:kernel.price(r) for r in ROLES}
    require(expected=={r:old.price(*SHAPE,4 if r=='d4' else 6) for r in ROLES},'Independent old exact complete tariff')
    require(tuple(kernel.SHAPE)==SHAPE and tuple(kernel.ROLES)==ROLES,'One geometry and fixed reference roles')
    require(not numerical_modules(),'No model or numerical work through metadata API')
    return kernel,literals,proof,{str(p):d for p,d in owners.items()}


def bind(mapping,tick=None):
    for path,digest in mapping.items():
        require(sha(path,tick)==digest,'Immutable complete own input/source/artifact')


def chain(folder,expected=None,tick=None):
    paths=[folder/n for n in ('result.json','completion.json','finalization.json')]
    ds=[sha(p,tick) for p in paths];r,e,f=map(read,paths)
    require((expected is None or ds[0]==expected)
        and all(x.get(k) is True for x in (r,e,f) for k in ('passed','completed'))
        and e['result_sha256']==f['result_sha256']==ds[0] and f['completion_sha256']==ds[1]
        and 0<=r['elapsed_s']<=f['elapsed_s']<120 and 0<r['peak_rss_bytes']<8_000_000_000,
        'Actual three positive bounded mode seals')
    return r,dict(zip(('result_sha256','completion_sha256','finalization_sha256'),ds))


def fixture_contract(args,owners,tick):
    path=ROOT/'prepare/contract.json'
    require(re.fullmatch('[0-9a-f]{64}',args.contract_sha256 or '') and sha(path,tick)==args.contract_sha256,
        'Actual separately generated synthetic contract, not an invented future digest')
    c=read(path)
    require(c['schema']==SCHEMA and c['shape']==list(SHAPE) and c['seed']==SEED and c['generator']=='PCG64'
        and c['roles']==list(ROLES) and c['source_bindings']==owners
        and c['model_calls']==0 and c['truth_accessed'] is False
        and set(c['fixtures'])=={'left.npy','right.npy','reference_f64.npy'},'Exact model-independent fixture ownership')
    expected={str(ROOT/'prepare'/name):row['sha256'] for name,row in c['fixtures'].items()}
    require(c['input_bindings']==expected,'No additional physical or target input')
    bind(expected,tick)
    prep,seals=chain(ROOT/'prepare',tick=tick)
    require(prep['contract_sha256']==args.contract_sha256 and prep['admitted_sources']==owners
        and all(prep['source_bindings'].get(p)==d for p,d in owners.items())
        and all(prep['artifacts'].get(p)==d for p,d in expected.items()),'Actual prepared files/seals')
    bind(prep['source_bindings'],tick);bind(prep['artifacts'],tick)
    return c,seals


def receipt_bindings(folder,seals):
    return {str(folder/name):seals[key] for name,key in (
        ('result.json','result_sha256'),('completion.json','completion_sha256'),
        ('finalization.json','finalization_sha256'))}


def prepare(args,kernel,literals,proof,result,out,guard):
    import numpy as np
    import flopscope
    import flopscope.numpy
    result['backend_attestation']=backend().backend_attestation('local',result['source_bindings'],guard)
    rng=np.random.Generator(np.random.PCG64(SEED))
    left=rng.standard_normal(SHAPE[:2],dtype=np.float32)/np.float32(math.sqrt(SHAPE[1]))
    right=rng.standard_normal(SHAPE[1:],dtype=np.float32)
    reference=left.astype(np.float64)@right.astype(np.float64)
    fixtures={}
    for name,value in (('left.npy',left),('right.npy',right),('reference_f64.npy',reference)):
        require(np.isfinite(value).all(),'Finite synthetic fixture and independent dense FP64 reference')
        path=out/name
        with path.open('xb') as stream:
            np.save(stream,value,allow_pickle=False)
        digest=sha(path,guard);result['artifacts'][str(path)]=digest
        fixtures[name]=dict(path=str(path),sha256=digest,dtype=str(value.dtype),shape=list(value.shape),
            raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())
    del left,right,reference,value,rng;gc.collect();guard()
    c=dict(schema=SCHEMA,shape=list(SHAPE),seed=SEED,generator='PCG64',roles=list(ROLES),
        source_bindings=result['admitted_sources'],input_bindings={v['path']:v['sha256'] for v in fixtures.values()},
        fixtures=fixtures,primitive_proof=proof,expected_kernel_flops={r:kernel.price(r) for r in ROLES},
        expected_setup_flops={r:kernel.setup_price(literals,r) for r in ROLES},
        expected_warm_primitive_counts=kernel.COUNTS,metric=kernel.micro_metadata(literals),
        model_calls=0,truth_accessed=False,reference_is_synthetic_FP64=True,
        fixture_generation_is_not_candidate_inference=True)
    result['contract_sha256']=write(out/'contract.json',c)
    result['artifacts'][str(out/'contract.json')]=result['contract_sha256']


@contextmanager
def paid_context(flops,guard):
    """Original client.12 owns no wall constructor: actual SIGALRM60 guard."""
    guard();began=time.monotonic();signal.setitimer(signal.ITIMER_REAL,60.)
    try:
        with flops.BudgetContext(flop_budget=2**39,quiet=True) as ctx:
            yield ctx
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    require(time.monotonic()-began<60.,'Actual per-context RPC60s');guard()


def bill_gate(bill):
    keys=('wall_time_s','residual_wall_time_s','flopscope_backend_time_s','flopscope_overhead_time_s')
    require(all(math.isfinite(bill[k]) and bill[k]>=0 for k in keys)
        and bill['wall_time_s']<60 and bill['residual_wall_time_s']<=.4
        and type(bill['flops_used']) is int and 0<=bill['flops_used']<2**39
        and sum(x['flop_cost'] for x in bill['operations'].values())==bill['flops_used'],
        'Complete RPC bill60s/residual.4, actual complete operation tariff sum')


def product_bill_gate(bill,role,index,kernel):
    bill_gate(bill)
    require(bill['flops_used']==kernel.price(role),'Exact full warm/cold arithmetic tariff')
    expected=dict(kernel.COUNTS[role])
    if index==0:
        expected['unstack']={'d4':16,'serial_d6':24,'microbatch3':44}[role]
    require({k:x['calls'] for k,x in bill['operations'].items()}==expected
        and (index!=0 or bill['operations']['unstack']['flop_cost']==0),
        'Exact numerical and zero-F frame-discovery calls, no unbilled arithmetic')


def setup_bill_gate(bill,role,kernel,literals):
    bill_gate(bill)
    require(bill['flops_used']==kernel.setup_price(literals,role)
        and {k:x['calls'] for k,x in bill['operations'].items()}
            ==dict(empty=2,reshape=48 if role=='d4' else 116)
        and bill['operations']['empty']['flop_cost']==0
        and bill['operations']['reshape']['flop_cost']==bill['flops_used'],
        'Exact two zero-F allocations and all paid original/replacement view bindings')


def fetch_bill_gate(bill):
    bill_gate(bill)
    require(bill['flops_used']==0 and bill['operations']=={},
        'Actual pinned output-fetch F0, with its complete wall/overhead still reported')


def role_run(args,kernel,literals,proof,result,out,guard):
    c,ps=fixture_contract(args,result['admitted_sources'],guard)
    result['input_bindings'].update(c['input_bindings'])
    result['input_bindings'].update(receipt_bindings(ROOT/'prepare',ps))
    result['input_bindings'][str(ROOT/'prepare/contract.json')]=args.contract_sha256
    result['preparation_seals']=ps
    import flopscope as flops
    import flopscope.numpy as fnp
    result['backend_attestation']=backend().backend_attestation('remote',result['source_bindings'],guard)
    ns,actual_literals,actual_proof=kernel.make_namespace(fnp)
    require(actual_literals==literals and actual_proof==proof==c['primitive_proof'],'Same exact audited kernel after import')
    with paid_context(flops,guard) as setup:
        left=fnp.load(str(ROOT/'prepare/left.npy'));right=fnp.load(str(ROOT/'prepare/right.npy'))
        state=kernel.setup(fnp,ns,literals,args.role,left,right)
    setup_bill=setup.summary_dict();del setup
    setup_bill_gate(setup_bill,args.role,kernel,literals)
    require(setup_bill['flops_used']==c['expected_setup_flops'][args.role],
        'All input loads/empty allocations and BOTH original plus actual replacement view bindings paid')
    setup_path=out/'setup.json'
    setup_record=dict(schema=SCHEMA,role=args.role,kind='setup',passed=True,completed=True,
        context_retired=True,bill=setup_bill,backing_arena_bytes=kernel.ARENA_BYTES,
        complete_flops=setup_bill['flops_used'],persistent_kernel_arrays_intentionally_retained=True)
    result['artifacts'][str(setup_path)]=write(setup_path,setup_record)
    print(json.dumps(dict(event='retired',role=args.role,kind='setup',call=None,path=str(setup_path),
        complete_sha256=result['artifacts'][str(setup_path)])),flush=True)
    result['setup']=setup_record;result['calls']=[];first=None
    for index in range(3):
        began=time.monotonic()
        with paid_context(flops,guard) as ctx:
            require(kernel.product(ns,args.role,left,right,state) is state['output'],'Returned original caller output')
        core=ctx.summary_dict();del ctx
        product_bill_gate(core,args.role,index,kernel)
        with paid_context(flops,guard) as verification:
            raw,shape,dtype=state['output']._fetch_data()
        egress=verification.summary_dict();del verification
        fetch_bill_gate(egress)
        require(tuple(shape)==(SHAPE[0],SHAPE[2]) and str(dtype)=='float32'
            and len(raw)==4*SHAPE[0]*SHAPE[2],'Exact actual NumPy-free output wire shape/dtype')
        digest=hashlib.sha256(raw).hexdigest()
        require(first is None or first==digest,'All own product repeats byte-exact')
        first=digest
        path=out/f'call_{index:02d}.f32'
        with path.open('xb') as stream:
            stream.write(raw)
        require(sha(path,guard)==digest,'Actual saved full output, not only a reported digest')
        result['artifacts'][str(path)]=digest
        record=dict(schema=SCHEMA,role=args.role,kind='product',call=index,passed=True,completed=True,
            context_retired=True,core_bill=core,verification_bill=egress,
            complete_flops=core['flops_used']+egress['flops_used'],output=str(path),output_sha256=digest,
            dtype='float32',shape=[SHAPE[0],SHAPE[2]],outer_wall_s=time.monotonic()-began,
            persistent_kernel_arrays_intentionally_retained=True,model_calls=0,truth_accessed=False)
        receipt=out/f'call_{index:02d}.json';rh=write(receipt,record)
        result['artifacts'][str(receipt)]=rh;result['calls'].append(dict(path=str(receipt),sha256=rh))
        print(json.dumps(dict(event='retired',role=args.role,kind='product',call=index,path=str(receipt),complete_sha256=rh)),flush=True)
        del raw,record;guard()
    refs=[weakref.ref(x) for x in (left,right,state['arena'],state['output'])]
    ns['_PACKED_FRAME_CACHE'].clear()
    if state['state'] is not None:
        state['state'].frames.clear();state['state'].groups.clear()
    del state,left,right,ns;gc.collect()
    require(all(ref() is None for ref in refs),'All persistent per-role input/output/arena owners actually retired')
    require(not any(k.split('.')[0] in ('numpy','scipy','torch') for k in sys.modules),'Actual client remained NumPy-free')
    result['backend_attestation']=backend().backend_attestation('remote',result['source_bindings'],guard)
    result.update(three_products_completed=True,own_repeat_byte_exact=True,all_role_array_owners_retired=True,
        all_contexts_retired=True,backing_arena_bytes=kernel.ARENA_BYTES)


def verify_role(role,contract,tick=None):
    out=ROOT/role/'client';r,seals=chain(out,tick=tick)
    require(r['schema']==SCHEMA and r['mode']=='remote' and r['role']==role and r['contract_sha256']==contract
        and all(r.get(k) is True for k in ('three_products_completed','own_repeat_byte_exact',
            'all_role_array_owners_retired','all_contexts_retired','source_hashes_unchanged'))
        and r['backend_attestation']['numpy_free'] and r['backing_arena_bytes']==1168900096
        and r['model_calls']==0 and r['truth_accessed'] is False and len(r['calls'])==3,'Actual fixed role qualification')
    bind(r['source_bindings'],tick);bind(r['input_bindings'],tick);bind(r['artifacts'],tick)
    records=[]
    for index,meta in enumerate(r['calls']):
        path=out/f'call_{index:02d}.json'
        require(meta==dict(path=str(path),sha256=r['artifacts'][str(path)]),'Exact ordered role receipt')
        q=read(path)
        require(q['role']==role and q['call']==index and q['context_retired']
            and q['output']==str(out/f'call_{index:02d}.f32')
            and r['artifacts'][q['output']]==q['output_sha256'],'Actual complete product artifact links')
        bill_gate(q['core_bill']);fetch_bill_gate(q['verification_bill'])
        require(q['complete_flops']==q['core_bill']['flops_used']+q['verification_bill']['flops_used'],
            'Both actual product and materialization contexts included')
        records.append(q)
    return r,seals,records


def judge(args,kernel,literals,proof,result,out,guard):
    c,ps=fixture_contract(args,result['admitted_sources'],guard)
    collected={r:verify_role(r,args.contract_sha256,guard) for r in ROLES}
    result['preparation_seals']=ps;result['role_seals']={r:x[1] for r,x in collected.items()}
    result['input_bindings'].update(c['input_bindings'])
    result['input_bindings'].update(receipt_bindings(ROOT/'prepare',ps))
    result['input_bindings'][str(ROOT/'prepare/contract.json')]=args.contract_sha256
    for role,(record,seals,calls) in collected.items():
        result['input_bindings'].update(record['artifacts'])
        result['input_bindings'].update(record['input_bindings'])
        result['input_bindings'].update(receipt_bindings(ROOT/role/'client',seals))
        for path,digest in record['source_bindings'].items():
            require(path not in result['source_bindings'] or result['source_bindings'][path]==digest,
                'Consistent actual imported numerical owner across fresh roles')
            result['source_bindings'][path]=digest
        require(record['primitive_proof']==proof and record['setup']['bill']['flops_used']==kernel.setup_price(literals,role),
            'Actual fixed setup and immutable source proof')
        require(len({q['output_sha256'] for q in calls})==1,'Independent own three-repeat hashes')
        setup_bill_gate(record['setup']['bill'],role,kernel,literals)
        for index,q in enumerate(calls):
            product_bill_gate(q['core_bill'],role,index,kernel)
    serial=collected['serial_d6'][2];micro=collected['microbatch3'][2]
    require(all(a['output_sha256']==b['output_sha256'] for a,b in zip(serial,micro,strict=True)),
        'Every original serialD6/microbatch3 output byte-exact, not a relaxed approximation')
    require(collected['microbatch3'][0]['setup']['bill']['flops_used']
        -collected['serial_d6'][0]['setup']['bill']['flops_used']==482160128,'Actual complete replacement cold-binding delta')
    import numpy as np
    import flopscope
    import flopscope.numpy
    result['backend_attestation']=backend().backend_attestation('local',result['source_bindings'],guard)
    reference=np.load(ROOT/'prepare/reference_f64.npy',allow_pickle=False)
    require(reference.dtype==np.float64 and reference.shape==(SHAPE[0],SHAPE[2]) and np.isfinite(reference).all(),
        'Exact independent synthetic FP64 reference frame')
    errors={}
    for role,(record,seals,calls) in collected.items():
        value=np.fromfile(calls[0]['output'],dtype=np.float32).reshape(SHAPE[0],SHAPE[2])
        require(np.isfinite(value).all(),'All actual signed kernel outputs finite')
        diff=value.astype(np.float64)-reference
        errors[role]=dict(maximum=float(np.max(np.abs(diff))),rms=float(np.sqrt(np.mean(diff*diff))))
        del value,diff;guard()
    del reference;gc.collect()
    times={r:max(q['core_bill']['wall_time_s'] for q in x[2][1:]) for r,x in collected.items()}
    ratio=times['microbatch3']/times['serial_d6']
    require(math.isfinite(ratio) and ratio>0,'Actual same-geometry measured warm timing ratio')
    result.update(numerical_and_billing_gate_passed=True,serial_microbatch_bytes_exact=True,
        complete_setup_and_warm_costs_verified=True,all_original_repetitions_passed=True,
        fp64_reference_errors=errors,maximum_warm_wall_s=times,microbatch_serial_warm_wall_ratio=ratio,
        timing_gate_passed=ratio<=.90,component_promising=ratio<=.90,prospective_timing_ratio=.90,
        additional_cold_binding_flops=482160128,warm_kernel_flops_unchanged=True,
        complete_estimator_qualified=False,official_timeout_repair_established=False)


def main(forced_mode=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--mode',choices=('prepare','server','remote','judge'))
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--contract-sha256');parser.add_argument('--server-url')
    for key in KEYS:
        parser.add_argument('--'+key+'-sha256')
    args=parser.parse_args()
    if not args.run:
        parser.print_help();return 0
    require(args.mode in (('remote',) if forced_mode=='remote' else ('prepare','server','judge')),'Actual worker/client entry')
    require((args.role in ROLES)==(args.mode in ('server','remote')),'Fixed role only for the actual RPC pair')
    began=time.monotonic()
    def guard():
        require(time.monotonic()-began<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8_000_000_000,
            'Entire mode120s/self8GB, externally supervised complete tree')
    def alarm(*unused):
        raise TimeoutError('Actual fixed RPC context or whole-mode deadline')
    require(sys.platform=='darwin' and sys.byteorder=='little' and sys.flags.optimize==0 and sys.dont_write_bytecode
        and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1' and all(os.environ.get(k)=='1' for k in THREADS)
        and not os.environ.get('PYTHONPATH') and not os.environ.get('PYTHONOPTIMIZE')
        and os.environ.get('FLOPSCOPE_MAX_ARRAY_BYTES')=='4294967296'
        and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0','Exact fixed runtime and one-thread environment')
    os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
    require(Path(sys.prefix).resolve()==(CLIENT if args.mode=='remote' else OPENBLAS).resolve(),'Actual pinned interpreter')
    if args.mode in ('remote','server'):
        require(args.server_url and args.server_url.startswith('ipc:///private/tmp/') and args.server_url.endswith('/flops.sock')
            and os.environ.get('FLOPSCOPE_SERVER_URL')==args.server_url,'Only owned local IPC, no external endpoint')
    else:
        require(not args.server_url and not os.environ.get('FLOPSCOPE_SERVER_URL'),'Offline synthetic prepare/judge')
    versions={'flopscope-client':'0.12.0'} if args.mode=='remote' else {
        'numpy':'2.4.6','flopscope':'0.12.0','flopscope-server':'0.12.0','threadpoolctl':'3.6.0'}
    require(all(importlib.metadata.version(k)==v for k,v in versions.items()),'Exact installed package versions')
    if args.mode=='remote':
        require(importlib.util.find_spec('numpy') is None,'NumPy-free actual client before imports')
    kernel,literals,proof,owners=admit(args,guard)
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,120.)
    if args.mode=='server':
        import flopscope
        import flopscope.numpy
        from flopscope_server.__main__ import main as server_main
        sources=dict(owners);attestation=backend().backend_attestation('server',sources,guard)
        write(ROOT/args.role/'server_attestation.json',dict(schema=SCHEMA,role=args.role,passed=True,
            source_bindings=sources,admitted_sources=owners,backend_attestation=attestation,
            pid=os.getpid(),url=args.server_url,no_fixture_arrays_created=True,model_calls=0,truth_accessed=False))
        sys.argv=['flopscope_server','--url',args.server_url,'--timeout','120'];server_main();return 0
    out=ROOT/args.role/'client' if args.mode=='remote' else ROOT/args.mode
    require(not out.exists() and not HERE.is_symlink(),'Exclusive new mode, no retry or resume')
    out.mkdir(parents=True,exist_ok=False)
    result=dict(schema=SCHEMA,mode=args.mode,role=args.role,passed=False,completed=False,source_bindings=dict(owners),
        admitted_sources=owners,input_bindings={},artifacts={},primitive_proof=proof,contract_sha256=args.contract_sha256,
        model_calls=0,truth_accessed=False,score_measured=False,candidate_changed=False,uploaded=False,
        submission_authorized=False,automatic_next_stage=False,package_versions=versions)
    result['artifacts'][str(out/'started.json')]=write(out/'started.json',result)
    try:
        {'prepare':prepare,'remote':role_run,'judge':judge}[args.mode](args,kernel,literals,proof,result,out,guard)
        if args.mode in ('prepare','judge'):
            result['backend_attestation']=backend().backend_attestation('local',result['source_bindings'],guard)
        bind(result['source_bindings'],guard);bind(result['input_bindings'],guard);bind(result['artifacts'],guard)
        result.update(passed=True,completed=True,source_hashes_unchanged=True,hashes_unchanged=True,sources_unchanged=True)
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    signal.setitimer(signal.ITIMER_REAL,0)
    result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    ok=bool(result['passed'] and result['elapsed_s']<120 and result['peak_rss_bytes']<8_000_000_000)
    result['passed']=result['completed']=ok;rh=write(out/'result.json',result)
    ok=bool(ok and time.monotonic()-began<120)
    ch=write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh))
    ok=bool(ok and time.monotonic()-began<120)
    write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=time.monotonic()-began))
    ok=bool(ok and time.monotonic()-began<120)
    print(json.dumps({k:result.get(k) for k in ('mode','role','passed','completed','component_promising','error')}),flush=True)
    return 0 if ok else 1


if __name__=='__main__':
    raise SystemExit(main())
