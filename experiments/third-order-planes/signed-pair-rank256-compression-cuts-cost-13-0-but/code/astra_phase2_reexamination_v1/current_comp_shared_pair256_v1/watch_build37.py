"""Single source-only build under the qualified external observation transport.

No model, fit, scoring, submission, or automatic numerical next stage.
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
import time
import traceback

HERE=Path(__file__).resolve().parent
NEW,P2=HERE.parent,HERE.parent.parent
OFFPY=P2/'data_gate/.venv/bin/python'
TRANSPORT=NEW/'current_comp_cstar_proposal_v1/watch.py'
TRANSPORT_SHA='d3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
CAP,RSS=60.,2_000_000_000
BLAS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(ok,message):
    if not ok: raise RuntimeError(message)


def no_numerics():
    require(not any(n.split('.')[0] in ('numpy','scipy','torch','flopscope',
        'flopscope_server','threadpoolctl') for n in sys.modules),'Entire watcher stdlib-only')


def sha(path,tick=None):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.is_relative_to(P2)
        and 'truthpost' not in path.parts and path.name!='key.txt','Regular in-scope target-free file')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
            if tick: tick()
    return h.hexdigest()


def read(path):
    path=Path(path)
    require(path.suffix=='.json' and path.stat().st_size<64*1024**2,'Bounded metadata only')
    return json.loads(path.read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def load(path,digest,name):
    require(sha(path)==digest,'Exact reviewed source owner')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    no_numerics();return module


def foreign_servers(rows,allowed):
    """Cover the underscore module spelling omitted by the historical classifier."""
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
    for key in ('source','worker','protocol','spec'): parser.add_argument('--'+key+'-sha256')
    args=parser.parse_args()
    if not args.run:
        parser.print_help();return 0
    began=previous=time.monotonic()
    require(args.numerical_slot_clear and Path(sys.prefix).resolve()==OFFPY.parent.parent.resolve()
        and os.getpriority(os.PRIO_PROCESS,0)>=10 and sys.flags.optimize==0 and sys.dont_write_bytecode
        and all(os.environ.get(k)=='1' for k in BLAS)
        and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
        and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
        and not os.environ.get('PYTHONPATH') and not os.environ.get('PYTHONOPTIMIZE')
        and not os.environ.get('FLOPSCOPE_SERVER_URL'),'Exact exclusive nice10/BLAS1 source watcher')
    worker=HERE/'build37.py'
    own={Path(__file__).resolve():args.source_sha256,worker:args.worker_sha256,
        HERE/'BUILD37_PROTOCOL.md':args.protocol_sha256,HERE/'BUILD37_WATCH_SPEC.md':args.spec_sha256,
        TRANSPORT:TRANSPORT_SHA}
    require(all(re.fullmatch('[0-9a-f]{64}',d or '') and sha(p)==d for p,d in own.items()),
        'Actual explicit source authority')
    h=load(TRANSPORT,TRANSPORT_SHA,'pair37_source_transport')
    mon,classifier=h.transport()
    out=HERE/'build37_watch_v1'
    require(not out.exists() and not (HERE/'build37_v1').exists() and not HERE.is_symlink(),
        'Never-resumed exclusive source stage')
    out.mkdir()
    result=dict(schema='current_comp_shared_pair256_build37_watch_v1',passed=False,completed=False,
        source_bindings={str(p):d for p,d in {**own,**h.MON_PINS}.items()},
        input_bindings={str(p):d for p,d in h.MOCK_PINS.items()},artifacts={},commands=[],
        numerical_imports=False,model_calls=0,truth_accessed=False,estimator_executed=False,
        parameter_updates=0,automatic_next_stage=False,uploaded=False,
        maximum_combined_rss_bytes=0,maximum_interval_s=0,maximum_swap_growth_bytes=0,
        completed_sample_intervals_s=[])
    children,logs=[],[]
    initial_swap=None
    stream=(out/'samples.jsonl').open('x')
    def deadline(): require(time.monotonic()<began+CAP,'Whole source60s through publication')
    def observe():
        nonlocal previous,initial_swap
        absolute=min(previous+1.,began+CAP)
        sample=mon.collect(absolute)
        scopes=[mon.scope(sample['rows'],os.getpid(),c.pid) for c in children]
        scopes.append(mon.scope(sample['rows'],os.getpid(),None))
        allowed=set().union(*(s['allowed'] for s in scopes))
        owned=mon.descendants({os.getpid()},sample['rows'])
        owned.update(pid for s in scopes for pid in s['child_pids'])
        combined=sum(r['rss_bytes'] for r in sample['rows'] if r['pid'] in owned)
        require(not classifier(sample['rows'],allowed) and not foreign_servers(sample['rows'],allowed),
            'No foreign numerical/Python worker, including underscore server modules')
        if initial_swap is None: initial_swap=sample['swap']['used_bytes']
        growth=sample['swap']['used_bytes']-initial_swap
        require(growth<=0 and combined<RSS,'Full source tree2GB/no added swap')
        deadline()
        result['maximum_combined_rss_bytes']=max(result['maximum_combined_rss_bytes'],combined)
        result['maximum_swap_growth_bytes']=max(result['maximum_swap_growth_bytes'],growth)
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began,combined_rss_bytes=combined,
            owned_pids=sorted(owned),swap_bytes=sample['swap']['used_bytes'],
            observational_groups_retired=sample['commands_reaped'],transport=sample['command_records']))+'\n')
        stream.flush()
        previous,interval=mon.complete(previous,absolute)
        result['maximum_interval_s']=max(result['maximum_interval_s'],interval)
        result['completed_sample_intervals_s'].append(interval)
    def tick():
        deadline()
        if time.monotonic()-previous>=.15: observe()
    def interrupt(*unused): raise RuntimeError('Source build watch interrupted')
    handlers={s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        observe();write(out/'started.json',result)
        command=[str(OFFPY),'-B','-u',str(worker),'--run','--source-sha256',args.worker_sha256,
            '--protocol-sha256',args.protocol_sha256]
        log=(out/'builder.log').open('x');logs.append(log)
        child=subprocess.Popen(command,cwd=NEW,env=dict(os.environ,PYTHONUNBUFFERED='1'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child);result['commands'].append(dict(argv=command,pid=child.pid,group=child.pid))
        observe()
        while child.poll() is None:
            observe();time.sleep(.10)
        observe();require(child.wait(timeout=.1)==0,'Source child completed without retry')
        mon.cleanup([child],time.monotonic()+.5);observe()
        api=load(worker,args.worker_sha256,'pair37_completed_source_builder')
        root=HERE/'build37_v1';paths=[root/name for name in api.NAMES]
        ds=[sha(path,tick) for path in paths]
        admitted=api.verify_build(ds[0],api.CANDIDATE_SHA,args.worker_sha256,args.protocol_sha256,tick)
        for group in ('source_bindings','input_bindings'):
            for path,digest in admitted[group].items():
                require(path not in result[group] or result[group][path]==digest,'Consistent source build ownership')
                result[group][path]=digest
        result['artifacts'].update({str(p):d for p,d in zip(paths,ds)})
        result['worker_seals']=dict(zip(('result_sha256','completion_sha256','finalization_sha256'),ds))
        result['candidate_sha256']=admitted['candidate_sha256']
        result['source_build_proof']=admitted['proof']
        for group in ('source_bindings','input_bindings','artifacts'):
            for path,digest in result[group].items(): require(sha(path,tick)==digest,'Fresh completed build inventory')
        no_numerics();observe()
        result.update(passed=True,completed=True,source_build_qualified=True,sources_unchanged=True,
            hashes_unchanged=True,actual_completed_samples=True)
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        try:
            mon.cleanup(children,time.monotonic()+.5);mon.retire_pending(time.monotonic()+.5)
            result.update(whole_owned_tree_retired=True,cleanup_passed=True,observational_groups_retired=True)
        except BaseException:
            result.update(passed=False,completed=False,cleanup_error=traceback.format_exc())
        for log in logs: log.close()
        try:
            if result['passed']: observe()
        except BaseException:
            result.update(passed=False,completed=False,final_observation_error=traceback.format_exc())
        stream.close()
        publication_deadline=min(previous+1.,began+CAP)
        for path in (out/'started.json',out/'samples.jsonl',out/'builder.log'):
            if path.exists(): result['artifacts'][str(path)]=sha(path)
        result.update(elapsed_s=time.monotonic()-began,no_post_publication_observation_claimed=True,
            publication_deadline_monotonic_s=publication_deadline)
        if result['passed']: require(time.monotonic()<publication_deadline,'Original1s/60s publication tail')
        write(out/'result.json',result);rh=sha(out/'result.json')
        ok=bool(result['passed'] and time.monotonic()<publication_deadline)
        write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh))
        ch=sha(out/'completion.json')
        write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,
            completion_sha256=ch,elapsed_s=time.monotonic()-began))
        if ok:
            require(time.monotonic()<publication_deadline,'Final write within original observation deadline');deadline()
        for sig,handler in handlers.items(): signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s',
        'maximum_combined_rss_bytes','maximum_interval_s','maximum_swap_growth_bytes','error')}),flush=True)
    return 0 if result['passed'] and ok else 1


if __name__=='__main__':
    raise SystemExit(main())
