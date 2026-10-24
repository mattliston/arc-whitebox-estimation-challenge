"""Independent serial admission plus fixed CHECK4 native acquisition watch."""
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
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
NEW,P2=HERE.parent,HERE.parent.parent
OFFPY=P2/'data_gate/.venv/bin/python'
TRANSPORT=NEW/'current_comp_cstar_proposal_v1/watch.py'
TRANSPORT_SHA='d3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
CANDIDATE_SHA='33879ab58f3a5e4e1ad41deb5db21b73ba703438374225d30fd0ebfc7511b89a'
RANKS=(37,39,42,48,37)
CAP,RSS=600.,8_000_000_000
BLAS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
      'MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(ok,message):
    if not ok: raise RuntimeError(message)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--numerical-slot-clear',action='store_true')
    keys=('source','worker','protocol','spec','runtime','build','candidate')
    for key in keys: parser.add_argument('--'+key+'-sha256')
    args=parser.parse_args()
    if not args.run:
        parser.print_help();return 0
    began=previous=last_progress=time.monotonic()
    require(args.numerical_slot_clear and os.getpriority(os.PRIO_PROCESS,0)>=10
        and sys.flags.optimize==0 and sys.dont_write_bytecode
        and Path(sys.prefix).resolve()==OFFPY.parent.parent.resolve()
        and all(os.environ.get(k)=='1' for k in BLAS),'Official serial nice10/BLAS1/assertions/bytecodeoff')
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '') for k in keys)
        and args.candidate_sha256==CANDIDATE_SHA,'Explicit reviewed sources and actual prior receipts')
    require(hashlib.sha256(TRANSPORT.read_bytes()).hexdigest()==TRANSPORT_SHA,'Qualified transport')
    h=load(TRANSPORT,'aligned_native_transport');mon,classifier=h.transport()
    require(not {'numpy','torch','flopscope','threadpoolctl'}&set(sys.modules),'Stdlib-only monitor')
    own={Path(__file__).resolve():args.source_sha256,HERE/'native.py':args.worker_sha256,
        HERE/'NATIVE_PROTOCOL.md':args.protocol_sha256,HERE/'NATIVE_WATCH_SPEC.md':args.spec_sha256,
        HERE/'build_v1/estimator.py':CANDIDATE_SHA,TRANSPORT:TRANSPORT_SHA,**h.MON_PINS}
    inputs={**h.MOCK_PINS,HERE/'runtime_v1/result.json':args.runtime_sha256,
        HERE/'build_v1/result.json':args.build_sha256}
    out=HERE/'native_watch_v1';ad=HERE/'native_terminal_admission_v1';native=HERE/'native_terminal_v1'
    require(not any(p.exists() for p in (out,ad,native)) and not HERE.is_symlink(),'Exclusive never-resumed native stages')
    out.mkdir()
    result=dict(schema='current_comp_aligned_push_d6_native_watch_v1',passed=False,completed=False,
        source_bindings={str(p):v for p,v in own.items()},input_bindings={str(p):v for p,v in inputs.items()},artifacts={},
        candidate_sha256=CANDIDATE_SHA,role='terminal',ranks=list(RANKS),
        numerical_imports=False,truth_arrays_accessed=False,score_measured=False,
        uploaded=False,automatic_next_stage=False,whole_cap_s=CAP,worker_rss_limit_bytes=RSS,
        admission_cap_s=60.,admission_rss_limit_bytes=2_000_000_000,
        maximum_combined_rss_bytes=0,maximum_admission_rss_bytes=0,maximum_worker_rss_bytes=0,
        maximum_swap_growth_bytes=0.,maximum_interval_s=0.,completed_sample_intervals_s=[],
        retired_call_count=0,full_model_calls=0,retired_calls=[],hashes_unchanged=False,sources_unchanged=False,
        whole_owned_tree_retired=False,phases=[])
    child=log=reader=stream=None;swap0=None;samples=0;pending=''
    phase='admission';phase_began=began

    def deadline():
        now=time.monotonic()
        require(now-began<CAP and now-last_progress<120.,'Whole600s/next-retired-context120s')
        if phase=='admission': require(now-phase_began<60.,'Source-only admission through reap/verification<60s')

    def observe():
        nonlocal previous,swap0,samples,pending,last_progress
        absolute=min(previous+1.,began+CAP,last_progress+120.,
            phase_began+60. if phase=='admission' else began+CAP)
        snapshot=mon.collect(absolute)
        scope=mon.scope(snapshot['rows'],os.getpid(),None if child is None else child.pid)
        allowed=scope.pop('allowed')
        require(not classifier(snapshot['rows'],allowed),'No foreign numerical worker')
        for row in snapshot['rows']:
            if row['pid'] not in allowed:
                try: argv=shlex.split(row['command'])
                except ValueError: argv=row['command'].split()
                require('flopscope_server' not in argv,'No foreign numerical server')
        if swap0 is None: swap0=snapshot['swap']['used_bytes']
        growth=snapshot['swap']['used_bytes']-swap0
        limit=2_000_000_000 if phase=='admission' else RSS
        require(growth<=0 and 0<scope['combined_rss_bytes']<limit,'Zero added swap and complete current-phase RSS bound')
        if reader is not None:
            pending+=reader.read();require(len(pending)<2*1024**2,'Bounded worker stdout tail')
            while '\n' in pending:
                line,pending=pending.split('\n',1)
                try: row=json.loads(line)
                except (ValueError,TypeError): continue
                if isinstance(row,dict) and row.get('event')=='retired':
                    index=len(result['retired_calls'])
                    require(phase=='worker' and index<5 and row['call']==index
                        and row.get('rank',RANKS[index])==RANKS[index],'Exactly ordered fully retired native5')
                    deadline();result['retired_calls'].append(dict(call=index,rank=RANKS[index]))
                    result['retired_call_count']=index+1;last_progress=time.monotonic()
        require(snapshot['commands_reaped'] and not mon.PENDING,'Actual observation subprocesses reaped')
        deadline()
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began,phase=phase,**scope,
            swap_bytes=snapshot['swap']['used_bytes'],swap_growth_bytes=growth,
            retired_call_count=result['retired_call_count'],transport=snapshot['command_records'],
            uncached_single_process_table=True),allow_nan=False)+'\n');stream.flush()
        previous,interval=mon.complete(previous,absolute)
        result['maximum_combined_rss_bytes']=max(result['maximum_combined_rss_bytes'],scope['combined_rss_bytes'])
        key='maximum_admission_rss_bytes' if phase=='admission' else 'maximum_worker_rss_bytes'
        result[key]=max(result[key],scope['combined_rss_bytes'])
        result['maximum_swap_growth_bytes']=max(result['maximum_swap_growth_bytes'],growth)
        result['maximum_interval_s']=max(result['maximum_interval_s'],interval)
        result['completed_sample_intervals_s'].append(interval);samples+=1

    def tick():
        deadline()
        if time.monotonic()-previous>=.2: observe()

    def seals(folder):
        paths=[folder/n for n in ('result.json','completion.json','finalization.json')]
        digests=[h.sha(p,tick) for p in paths];r,e,f=map(h.read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256']==f['result_sha256']==digests[0]
            and f['completion_sha256']==digests[1],'Actual complete child three-seal chain')
        for p,digest in zip(paths,digests): result['input_bindings'][str(p)]=digest
        return r,e,dict(zip(('result_sha256','completion_sha256','finalization_sha256'),digests))

    def inventory(record):
        for group in ('source_bindings','input_bindings','artifacts'):
            require(isinstance(record[group],dict) and record[group],'Complete non-target inventory')
            for p,digest in record[group].items():
                require(re.fullmatch('[0-9a-f]{64}',digest or '') and Path(p).is_absolute(),'Actual digest binding')
                require('truthpost' not in Path(p).parts and not
                    (Path(p).suffix in ('.npy','.npz') and 'truth' in Path(p).name.lower()),'No target arrays')
                require(h.sha(p,tick)==digest,'Fresh source/input/artifact binding')
                target='artifacts' if group=='artifacts' else group
                require(p not in result[target] or result[target][p]==digest,'Consistent source inventory')
                result[target][p]=digest

    def interrupted(*unused): raise RuntimeError('Owned fixed CHECK4 native experiment interrupted')
    handlers={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        stream=(out/'samples.jsonl').open('x');observe()
        for p,digest in {**own,**inputs}.items():
            require(h.sha(p,tick)==digest,'Reviewed owner or actual prerequisite')
        worker=load(HERE/'native.py','aligned_push_native_saved_verifier')
        require(not {'numpy','torch','flopscope','threadpoolctl'}&set(sys.modules),'Inert verifier import only')
        common=['--role','terminal','--source-sha256',args.worker_sha256,
            '--protocol-sha256',args.protocol_sha256,'--watcher-sha256',args.source_sha256,
            '--watch-spec-sha256',args.spec_sha256]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            PYTORCH_ENABLE_MPS_FALLBACK='0',FLOPSCOPE_MAX_ARRAY_BYTES='4294967296')
        env.update({k:'1' for k in BLAS})
        argv=[str(OFFPY),'-B','-u',str(HERE/'native.py'),'--admit',*common,
            '--runtime-sha256',args.runtime_sha256,'--build-sha256',args.build_sha256,
            '--candidate-sha256',CANDIDATE_SHA]
        log=(out/'admission.log').open('x');reader=(out/'admission.log').open()
        child=subprocess.Popen(argv,cwd=NEW,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result.update(admission_pid=child.pid,admission_command=argv)
        h.write(out/'started.json',result)
        while child.poll() is None:
            observe();time.sleep(.15)
        observe();require(child.wait(timeout=1)==0,'Admission successfully exited')
        mon.cleanup([child],time.monotonic()+.5)
        ar,ae,aseals=seals(ad)
        inventory(ar)
        context_sha=h.sha(ad/'context.json',tick)
        ns=SimpleNamespace(**{**vars(args),'source_sha256':args.worker_sha256,
            'watcher_sha256':args.source_sha256,'watch_spec_sha256':args.spec_sha256,
            'role':'terminal','context_sha256':context_sha})
        context=worker.read_context(ns,context_sha,tick=tick)
        require(context['candidate_sha256']==CANDIDATE_SHA,'Exact admitted candidate')
        require(not {'numpy','torch','flopscope','threadpoolctl'}&set(sys.modules),'Source-only retired admission')
        result.update(admission_seals=aseals,context_sha256=context_sha,admission_producer_reaped=True)
        result['input_bindings'][str(ad/'context.json')]=context_sha
        result['phases'].append(dict(phase='admission',pid=child.pid,exited=True,
            elapsed_through_verification_s=time.monotonic()-phase_began))
        deadline();observe()
        reader.close();log.close();reader=log=None;pending=''
        phase='worker';phase_began=last_progress=time.monotonic()
        argv=[str(OFFPY),'-B','-u',str(HERE/'native.py'),'--run',*common,'--context-sha256',context_sha]
        log=(out/'worker.log').open('x');reader=(out/'worker.log').open()
        child=subprocess.Popen(argv,cwd=NEW,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result.update(worker_pid=child.pid,worker_command=argv)
        h.write(out/'worker_started.json',dict(command=argv,pid=child.pid,context_sha256=context_sha))
        while child.poll() is None:
            observe();time.sleep(.15)
        observe()
        require(child.wait(timeout=1)==0 and result['retired_call_count']==5,'Native worker exited with all five fully retired')
        mon.cleanup([child],time.monotonic()+.5)
        record,end,nseals=seals(native)
        worker.verify_acquisition(record,end,nseals['result_sha256'],ns,context,tick=tick)
        require(record['schema']=='current_comp_aligned_push_d6_native_v1'
            and record['candidate_sha256']==CANDIDATE_SHA
            and record['full_model_calls']==record['retired_call_count']==5
            and record['initialization_flops']==1944970298
            and record['truth_accessed'] is False,'Actual complete fixed five-call native subject')
        for key in ('own_complete_bills','all16_domains_qualified','all47_selectors_qualified',
                    'repeat_output_domain_byte_exact','initialization_ownership_passed',
                    'same_existing_arena_qualified','additional_bindings_paid_once','dispatch_tariffs_qualified',
                    'sources_unchanged','hashes_unchanged'):
            require(record.get(key) is True,'Original native qualification plus generic D6: '+key)
        inventory(record)
        require(not {'numpy','torch','flopscope','threadpoolctl'}&set(sys.modules),'Final saved-acquisition verification is stdlib-only')
        for p,digest in {**result['source_bindings'],**result['input_bindings'],**result['artifacts']}.items():
            require(h.sha(p,tick)==digest,'Final independently rehashed native non-target inventory')
        result['phases'].append(dict(phase='worker',pid=child.pid,exited=True,
            elapsed_through_verification_s=time.monotonic()-phase_began))
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            child_execution_completed=True,whole_owned_tree_retired=True,full_model_calls=5,
            initialization_flops=1944970298,all5_complete=True,actual_completed_samples=True,
            foreign_numerical_workers_checked=True,child_result_sha256=nseals['result_sha256'],
            child_completion_sha256=nseals['completion_sha256'],child_finalization_sha256=nseals['finalization_sha256'])
        observe()
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        try:
            if child is not None: mon.cleanup([child],time.monotonic()+.5)
            mon.retire_pending(time.monotonic()+.5)
            result.update(cleanup_passed=True,whole_owned_tree_retired=True,observational_groups_retired=True)
        except BaseException:
            result.update(passed=False,completed=False,cleanup_error=traceback.format_exc())
        for handle in (reader,log,stream):
            if handle is not None: handle.close()
        result.update(elapsed_s=time.monotonic()-began,observation_samples=samples,initial_swap_bytes=swap0,
            last_completed_observation_elapsed_s=previous-began,post_observation_memory_sample_claimed=False)
        if result['elapsed_s']>=CAP or time.monotonic()-last_progress>=120.:
            result.update(passed=False,completed=False,error='Whole/progress deadline including cleanup')
        if phase=='admission' and time.monotonic()-phase_began>=60.:
            result.update(passed=False,completed=False,error='Source-only admission exceeded60s')
        for name in ('started.json','worker_started.json','admission.log','worker.log','samples.jsonl'):
            p=out/name
            if p.exists(): result['artifacts'][str(p)]=h.sha(p)
        if time.monotonic()-previous>1.:
            result.update(passed=False,completed=False,error='Terminal cleanup/hash interval>1s')
        h.write(out/'result.json',result);rh=h.sha(out/'result.json')
        ok=bool(result['passed'] and time.monotonic()-began<CAP and time.monotonic()-last_progress<120.
            and time.monotonic()-previous<=1.)
        h.write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh));ch=h.sha(out/'completion.json')
        elapsed=time.monotonic()-began;terminal_interval=time.monotonic()-previous
        ok=bool(ok and elapsed<CAP and time.monotonic()-last_progress<120. and terminal_interval<=1.)
        h.write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,
            elapsed_s=elapsed,terminal_publication_interval_s=terminal_interval))
        h.sha(out/'finalization.json');ok=bool(ok and time.monotonic()-began<CAP
            and time.monotonic()-last_progress<120. and time.monotonic()-previous<=1.)
        for sig,handler in handlers.items(): signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s','full_model_calls',
        'maximum_combined_rss_bytes','maximum_interval_s','maximum_swap_growth_bytes','error')},allow_nan=False),flush=True)
    return 0 if ok else 1


if __name__=='__main__':
    raise SystemExit(main())
