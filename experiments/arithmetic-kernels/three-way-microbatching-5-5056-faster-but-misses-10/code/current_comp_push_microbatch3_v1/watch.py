"""DRAFT inert supervisor: three fixed, serial, NumPy-free RPC kernel roles.

The qualified stdlib monitor owns actual process/swap sampling and group reap.
This new small orchestration admits no model/truth or ancestral panel graph.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
NEW,P2=HERE.parent,HERE.parent.parent
ROOT,OUT=HERE/'run_v1',HERE/'watch_v1'
OFFPY=P2/'data_gate/.venv/bin/python'
OPENPY=NEW/'openblas_runtime_env_v1/bin/python'
CLIENTPY=P2/'data_gate/.client-venv/bin/python'
TRANSPORT=NEW/'current_comp_cstar_proposal_v1/watch.py'
TRANSPORT_SHA='d3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
CAP,RSS=300.,8_000_000_000
ROLES=('d4','serial_d6','microbatch3')
BLAS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
      'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
SCHEMA='current_comp_push_microbatch3_kernel_watch_v1'


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def sha(path,tick=None):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(P2)
        and 'truthpost' not in path.parts and path.name!='weights.npy'
        and not (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),
        'Only direct kernel sources, metadata and own synthetic bytes')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def read(path):
    path=Path(path)
    require(path.suffix=='.json' and not path.is_symlink() and path.stat().st_size<16*1024**2,
        'Bounded direct metadata, never numerical decoding')
    return json.loads(path.read_text())


def write(path,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    require(len(raw)<16*1024**2,'Bounded exact JSON publication')
    with Path(path).open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def load(path,digest,name):
    require(sha(path)==digest,'Exact reviewed inert source before import')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def foreign_servers(rows,allowed):
    """Cover python -m flopscope_server, omitted by the old dash-spelled classifier."""
    found=[]
    for row in rows:
        if row['pid'] in allowed:
            continue
        try:
            argv=shlex.split(row['command'])
        except ValueError:
            argv=row['command'].split()
        if argv and re.fullmatch(r'(?:python|pypy)[0-9.]*',Path(argv[0]).name.lower()) and any(
            x=='flopscope_server' or x.startswith('flopscope_server.') for x in argv[1:]):
            found.append(row)
    return found


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--numerical-slot-clear',action='store_true')
    for key in ('source','worker','kernel','client','protocol','spec'):
        parser.add_argument('--'+key+'-sha256')
    args=parser.parse_args()
    if not args.run:
        parser.print_help();return 0
    began=previous=last_progress=time.monotonic()
    require(args.numerical_slot_clear and Path(sys.prefix).resolve()==OFFPY.parent.parent.resolve()
        and sys.platform=='darwin' and sys.flags.optimize==0 and sys.dont_write_bytecode
        and os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in BLAS)
        and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
        and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
        and os.environ.get('FLOPSCOPE_MAX_ARRAY_BYTES')=='4294967296'
        and not os.environ.get('PYTHONPATH') and not os.environ.get('PYTHONOPTIMIZE')
        and not os.environ.get('FLOPSCOPE_SERVER_URL'),
        'One explicit numerical slot, OFFPY nice10, six threads1, bytecodeoff and fixed4GiB array limit')
    own={HERE/'watch.py':args.source_sha256,HERE/'worker.py':args.worker_sha256,
        HERE/'kernel.py':args.kernel_sha256,HERE/'client.py':args.client_sha256,
        HERE/'PROTOCOL.md':args.protocol_sha256,HERE/'WATCH_SPEC.md':args.spec_sha256,
        TRANSPORT:TRANSPORT_SHA}
    require(all(re.fullmatch('[0-9a-f]{64}',d or '') and sha(p)==d for p,d in own.items()),
        'Every actual new owner supplied explicitly before any child')
    h=load(TRANSPORT,TRANSPORT_SHA,'micro3_exact_monitor_transport')
    mon,classifier=h.transport()
    require(not ROOT.exists() and not OUT.exists() and not HERE.is_symlink(),
        'Exclusive future run/watch directories, no retry or appended old result')
    OUT.mkdir();ROOT.mkdir()
    result=dict(schema=SCHEMA,passed=False,completed=False,
        source_bindings={str(p):d for p,d in {**own,**h.MON_PINS}.items()},
        input_bindings={str(p):d for p,d in h.MOCK_PINS.items()},artifacts={},commands=[],
        numerical_imports=False,truth_arrays_accessed=False,model_calls=0,parameter_updates=0,
        candidate_changed=False,uploaded=False,submission_authorized=False,automatic_next_stage=False,
        official_timeout_repair_established=False,complete_estimator_qualified=False,
        roles=list(ROLES),shape=[1024,384,8192],whole_cap_s=CAP,
        maximum_combined_rss_bytes=0,maximum_swap_growth_bytes=0,maximum_interval_s=0,
        completed_sample_intervals_s=[],retired_context_groups=0,context_markers=[],
        positive_role_seals={},preparation_seals=None,judge_seals=None)
    children,logs,readers,pending=[],[],{},{}
    active_children={}
    initial_swap=None;api=kernel=None;admitted=None;contract_sha=None;contract=None
    stream=(OUT/'samples.jsonl').open('x')

    def deadline():
        now=time.monotonic()
        require(now-began<CAP and now-last_progress<120.,'Entire300s and actual retired-context progress120s')
        require(all(now<record['deadline_monotonic_s'] for record in active_children.values()),
            'Every actual child group retires within120s of its spawn, independent of worker timers')

    def observation_deadline():
        return min(previous+1.,began+CAP,last_progress+120.,
            *(record['deadline_monotonic_s'] for record in active_children.values()))

    def accept_markers():
        nonlocal last_progress
        for role,reader in readers.items():
            pending[role]+=reader.read()
            require(len(pending[role])<4*1024**2,'Bounded complete JSON marker log')
            while '\n' in pending[role]:
                line,pending[role]=pending[role].split('\n',1)
                try:
                    item=json.loads(line)
                except (ValueError,TypeError):
                    continue
                if not isinstance(item,dict) or item.get('event')!='retired':
                    continue
                deadline();number=result['retired_context_groups']
                require(number<12 and role==ROLES[number//4] and item['role']==role,
                    'Exactly three fixed serial role lifetimes, not extra timing attempts')
                index=number%4-1;kind='setup' if index<0 else 'product'
                path=ROOT/role/'client'/('setup.json' if index<0 else f'call_{index:02d}.json')
                digest=sha(path);q=read(path)
                require(item['kind']==kind and item['call']==(None if index<0 else index)
                    and item['path']==str(path) and item['complete_sha256']==digest
                    and q['schema']==api.SCHEMA and q['role']==role and q['kind']==kind
                    and all(q.get(k) is True for k in ('passed','completed','context_retired',
                        'persistent_kernel_arrays_intentionally_retained')),
                    'Progress only from actual complete paid-and-retired context receipts')
                if index<0:
                    api.setup_bill_gate(q['bill'],role,kernel,literals)
                    require(q['complete_flops']==q['bill']['flops_used']==contract['expected_setup_flops'][role]
                        and q['backing_arena_bytes']==1168900096,'Exact actual complete setup tariff/arena')
                else:
                    api.product_bill_gate(q['core_bill'],role,index,kernel)
                    api.fetch_bill_gate(q['verification_bill'])
                    require(q['call']==index and q['complete_flops']==q['core_bill']['flops_used']
                        and q['output']==str(ROOT/role/'client'/f'call_{index:02d}.f32')
                        and Path(q['output']).stat().st_size==33554432
                        and re.fullmatch('[0-9a-f]{64}',q['output_sha256']),
                        'Actual complete kernel+egress bill and full-shape output owner')
                result['context_markers'].append(dict(role=role,kind=kind,call=None if index<0 else index,
                    path=str(path),sha256=digest))
                result['retired_context_groups']+=1
                last_progress=time.monotonic()

    def observe():
        nonlocal previous,initial_swap
        absolute=observation_deadline()
        sample=mon.collect(absolute)
        scopes=[mon.scope(sample['rows'],os.getpid(),c.pid) for c in children]
        scopes.append(mon.scope(sample['rows'],os.getpid(),None))
        allowed=set().union(*(s['allowed'] for s in scopes))
        owned=mon.descendants({os.getpid()},sample['rows'])
        owned.update(pid for s in scopes for pid in s['child_pids'])
        combined=sum(r['rss_bytes'] for r in sample['rows'] if r['pid'] in owned)
        require(not classifier(sample['rows'],allowed) and not foreign_servers(sample['rows'],allowed),
            'No other numerical tree, including underscore-spelled server modules')
        if initial_swap is None:
            initial_swap=sample['swap']['used_bytes'];result['initial_swap_bytes']=initial_swap
        growth=sample['swap']['used_bytes']-initial_swap
        require(growth<=0 and combined<RSS,'Actual whole owned tree8GB and zero added system swap')
        accept_markers();deadline()
        result['maximum_combined_rss_bytes']=max(result['maximum_combined_rss_bytes'],combined)
        result['maximum_swap_growth_bytes']=max(result['maximum_swap_growth_bytes'],growth)
        child_pids=sorted(set().union(*(set(s['child_pids']) for s in scopes)))
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began,combined_rss_bytes=combined,
            owned_pids=sorted(owned),child_pids=child_pids,swap_bytes=sample['swap']['used_bytes'],
            retired_context_groups=result['retired_context_groups'],
            observational_groups_retired=sample['commands_reaped'],transport=sample['command_records']))+'\n')
        stream.flush()
        previous,interval=mon.complete(previous,absolute)
        result['maximum_interval_s']=max(result['maximum_interval_s'],interval)
        result['completed_sample_intervals_s'].append(interval)
        result['last_child_pids']=child_pids

    def tick():
        deadline()
        if time.monotonic()-previous>=.15:
            observe()

    def merge(group,mapping):
        for path,digest in mapping.items():
            require(sha(path,tick)==digest,'Fresh actual kernel source/input/artifact')
            require(path not in result[group] or result[group][path]==digest,'No conflicting owner in final union')
            result[group][path]=digest

    def spawn(argv,name,env,role=None):
        log=(OUT/(name+'.log')).open('x');logs.append(log)
        spawned=time.monotonic()
        child=subprocess.Popen(argv,cwd=NEW,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child)
        record=dict(name=name,argv=argv,pid=child.pid,group=child.pid,
            spawn_monotonic_s=spawned,deadline_monotonic_s=spawned+120.)
        result['commands'].append(record);active_children[child.pid]=record
        if role is not None:
            readers[role]=(OUT/(name+'.log')).open('r');pending[role]=''
        observe();return child

    def retire(child):
        deadline()
        mon.cleanup([child],observation_deadline())
        require(child.poll() is not None and not mon.group_alive(child.pid),'Actual whole child group reaped')
        retired=time.monotonic();record=active_children[child.pid]
        require(retired<record['deadline_monotonic_s'],'Actual observed group retirement before its120s deadline')
        record.update(retired_monotonic_s=retired,
            elapsed_through_group_retirement_s=retired-record['spawn_monotonic_s'])
        del active_children[child.pid]
        observe()

    def finish(child,server=None):
        while child.poll() is None:
            if server is not None:
                require(server.poll() is None,'The owned local IPC server remains live')
            observe();time.sleep(.10)
        observe();require(child.wait(timeout=.1)==0,'One actual mode succeeded without retry')
        retire(child)

    def verify_mode(folder,mode):
        r,seals=api.chain(folder,tick=tick)
        require(r['schema']==api.SCHEMA and r['mode']==mode and r['admitted_sources']==admitted
            and r['source_hashes_unchanged'] and r['model_calls']==0 and r['truth_accessed'] is False
            and r['candidate_changed'] is False and r['automatic_next_stage'] is False
            and r['backend_attestation']['actual_origins_qualified']
            and r['backend_attestation']['production_OpenBLAS_qualified'],
            'Actual bounded synthetic mode and production OpenBLAS origins')
        for group in ('source_bindings','input_bindings','artifacts'):
            merge(group,r[group])
        merge('artifacts',api.receipt_bindings(folder,seals))
        return r,seals

    def interrupted(*unused):
        raise RuntimeError('Owned microbatch qualification interrupted')

    handlers={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        observe()
        api=load(HERE/'worker.py',args.worker_sha256,'micro3_supervised_metadata')
        ns=SimpleNamespace(source_sha256=args.worker_sha256,kernel_sha256=args.kernel_sha256,
            client_sha256=args.client_sha256,watcher_sha256=args.source_sha256,
            protocol_sha256=args.protocol_sha256,watch_spec_sha256=args.spec_sha256,contract_sha256=None)
        kernel,literals,proof,admitted=api.admit(ns,tick)
        merge('source_bindings',admitted)
        require(not api.numerical_modules(),'Actual supervisor remains stdlib-only')
        result['primitive_proof']=proof;result['authority']=vars(ns).copy()
        write(OUT/'started.json',result);tick()
        tail=['-B','-u',str(HERE/'worker.py'),'--run','--source-sha256',args.worker_sha256,
            '--kernel-sha256',args.kernel_sha256,'--client-sha256',args.client_sha256,
            '--watcher-sha256',args.source_sha256,'--protocol-sha256',args.protocol_sha256,
            '--watch-spec-sha256',args.spec_sha256]
        env=dict(os.environ,PYTHONUNBUFFERED='1')
        child=spawn([str(OPENPY),*tail,'--mode','prepare'],'prepare',env)
        finish(child);prep,ps=verify_mode(ROOT/'prepare','prepare')
        contract_sha=prep['contract_sha256'];ns.contract_sha256=contract_sha
        contract,actual_ps=api.fixture_contract(ns,admitted,tick)
        require(ps==actual_ps and contract['primitive_proof']==proof,'Actual fixed fixture/source contract')
        result['preparation_seals']=ps;result['contract_sha256']=contract_sha
        for role in ROLES:
            (ROOT/role).mkdir(exist_ok=False)
            scratch=Path(tempfile.mkdtemp(prefix='astra_push_micro3_'+role+'_',dir='/private/tmp'))
            result.setdefault('ipc_directories',[]).append(str(scratch))
            url='ipc://'+str(scratch/'flops.sock');remote_env=dict(env,FLOPSCOPE_SERVER_URL=url)
            role_tail=[*tail,'--role',role,'--contract-sha256',contract_sha,'--server-url',url]
            server=spawn([str(OPENPY),*role_tail,'--mode','server'],role+'_server',remote_env)
            server_start=time.monotonic()
            while not (scratch/'flops.sock').exists():
                require(server.poll() is None and time.monotonic()-server_start<5.,'Bounded actual IPC startup')
                observe();time.sleep(.05)
            client_tail=list(role_tail);client_tail[2]=str(HERE/'client.py')
            client=spawn([str(CLIENTPY),*client_tail,'--mode','remote'],role+'_client',remote_env,role)
            finish(client,server)
            retire(server)
            r,seals,calls=api.verify_role(role,contract_sha,tick)
            require(r['admitted_sources']==admitted and r['preparation_seals']==ps and r['primitive_proof']==proof,
                'Every actual client shares the same complete fixture/source authority')
            result['positive_role_seals'][role]=seals
            for group in ('source_bindings','input_bindings','artifacts'):
                merge(group,r[group])
            merge('artifacts',api.receipt_bindings(ROOT/role/'client',seals))
            path=ROOT/role/'server_attestation.json';att=read(path)
            require(att['schema']==api.SCHEMA and att['passed'] and att['role']==role
                and att['admitted_sources']==admitted and att['pid']==server.pid and att['url']==url
                and att['no_fixture_arrays_created'] and att['model_calls']==0 and att['truth_accessed'] is False
                and att['backend_attestation']['actual_origins_qualified']
                and att['backend_attestation']['production_OpenBLAS_qualified'],
                'Actual owned pinned single-thread OpenBLAS server, not an interpreter label')
            merge('source_bindings',att['source_bindings']);merge('artifacts',{str(path):sha(path,tick)})
            require(result['retired_context_groups']==4*(ROLES.index(role)+1)
                and not mon.group_alive(server.pid) and not mon.group_alive(client.pid),
                'All setup/product groups and whole old role retired before the next role')
        child=spawn([str(OPENPY),*tail,'--mode','judge','--contract-sha256',contract_sha],'judge',env)
        finish(child);judged,js=verify_mode(ROOT/'judge','judge')
        require(judged['contract_sha256']==contract_sha and judged['role_seals']==result['positive_role_seals']
            and judged['preparation_seals']==ps and judged['primitive_proof']==proof
            and all(judged.get(k) is True for k in ('numerical_and_billing_gate_passed',
                'serial_microbatch_bytes_exact','complete_setup_and_warm_costs_verified',
                'all_original_repetitions_passed','warm_kernel_flops_unchanged'))
            and judged['additional_cold_binding_flops']==482160128
            and judged['prospective_timing_ratio']==.90
            and judged['component_promising']==judged['timing_gate_passed']
                ==(judged['microbatch_serial_warm_wall_ratio']<=.90)
            and judged['complete_estimator_qualified'] is False
            and judged['official_timeout_repair_established'] is False,
            'Complete direct comparison and unchanged prospective component-only timing question')
        result['judge_seals']=js
        result['comparison']={k:judged[k] for k in ('fp64_reference_errors','maximum_warm_wall_s',
            'microbatch_serial_warm_wall_ratio','timing_gate_passed','component_promising',
            'additional_cold_binding_flops')}
        for group in ('source_bindings','input_bindings','artifacts'):
            merge(group,dict(result[group]))
        require(not api.numerical_modules(),'No numerical imports or synthetic-array decoding in supervisor')
        result.update(passed=True,completed=True,operational_passed=True,
            component_promising=judged['component_promising'],nine_products_completed=True,
            three_setups_completed=True,budget_contexts_completed=21,
            sources_unchanged=True,hashes_unchanged=True,actual_completed_samples=True)
        observe()
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        try:
            cleanup_deadline=observation_deadline() if result['passed'] else time.monotonic()+.5
            mon.cleanup(children,cleanup_deadline);mon.retire_pending(cleanup_deadline)
            if result['passed']:
                require(not active_children,'Every successful child lifetime already observed and retired within120s')
            result.update(whole_owned_tree_retired=True,cleanup_passed=True,observational_groups_retired=True,
                child_exit_records=[dict(pid=c.pid,returncode=c.poll(),group_retired=not mon.group_alive(c.pid)) for c in children])
        except BaseException:
            result.update(passed=False,completed=False,cleanup_error=traceback.format_exc())
        for handle in logs:
            handle.close()
        try:
            if result['passed']:
                observe();require(result['last_child_pids']==[],'Final actual sample has no owned child group')
        except BaseException:
            result.update(passed=False,completed=False,final_observation_error=traceback.format_exc())
        stream.close()
        for reader in readers.values():
            reader.close()
        publication_deadline=min(previous+1.,began+CAP,last_progress+120.)
        try:
            for path in (OUT/'started.json',OUT/'samples.jsonl',*(OUT/(row['name']+'.log') for row in result['commands'])):
                if path.exists():
                    result['artifacts'][str(path)]=sha(path)
            if result['passed']:
                require(time.monotonic()<publication_deadline,'All final bytes inside actual completed1s tail')
        except BaseException:
            result.update(passed=False,completed=False,publication_error=traceback.format_exc())
        result.update(no_post_publication_observation_claimed=True,
            publication_deadline_monotonic_s=publication_deadline,elapsed_s=time.monotonic()-began)
        if result['elapsed_s']>=CAP or (result['passed'] and time.monotonic()>=publication_deadline):
            result.update(passed=False,completed=False)
        rh=write(OUT/'result.json',result)
        ok=bool(result['passed'] and time.monotonic()<publication_deadline)
        ch=write(OUT/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh))
        ok=bool(ok and time.monotonic()<publication_deadline)
        write(OUT/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,
            completion_sha256=ch,elapsed_s=time.monotonic()-began))
        ok=bool(ok and time.monotonic()<publication_deadline)
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','component_promising','elapsed_s',
        'maximum_combined_rss_bytes','maximum_interval_s','maximum_swap_growth_bytes','error')}),flush=True)
    return 0 if ok and time.monotonic()<publication_deadline else 1


if __name__=='__main__':
    raise SystemExit(main())
