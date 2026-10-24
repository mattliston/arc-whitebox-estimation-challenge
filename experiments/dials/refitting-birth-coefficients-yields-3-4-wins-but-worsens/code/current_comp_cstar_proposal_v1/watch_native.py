"""Retired source admission then one fixed five-call native CHECK4 stage."""
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
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
P2 = HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
TRANSPORT = HERE/'watch.py'
TRANSPORT_SHA = 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
SCHEMA = 'current_comp_cstar_proposal_native_watch_v1'
SEQUENCE = (37,39,42,48,37)
BLAS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
KEYS = ('source','worker','model','protocol','spec','builder','build-protocol','build',
        'build-watch','build-watcher','build-watch-spec','candidate')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def transport():
    require(hashlib.sha256(TRANSPORT.read_bytes()).hexdigest() == TRANSPORT_SHA,
        'Frozen source-only transport helpers')
    spec = importlib.util.spec_from_file_location('cprop_native_transport_only',TRANSPORT)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    mon,classifier = helper.transport()
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),
        'Supervisor imports no numerical modules')
    return helper,mon,classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--role',choices=('terminal',))
    parser.add_argument('--numerical-slot-clear',action='store_true')
    for key in KEYS:
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help();return 0
    began = previous = progress_at = time.monotonic()
    require(args.role == 'terminal' and args.numerical_slot_clear
        and os.getpriority(os.PRIO_PROCESS,0) >= 10 and sys.dont_write_bytecode
        and sys.flags.optimize == 0 and all(os.environ.get(k) == '1' for k in BLAS),
        'One reviewed nice10/BLAS1/bytecodeoff native terminal stage')
    for key in KEYS:
        require(re.fullmatch('[0-9a-f]{64}',getattr(args,key.replace('-','_')+'_sha256') or ''),
            'Explicit source/candidate/build authority: '+key)
    h,mon,classifier = transport()
    own = {Path(__file__).resolve():args.source_sha256,HERE/'native.py':args.worker_sha256,
        HERE/'model.py':args.model_sha256,HERE/'NATIVE_PROTOCOL.md':args.protocol_sha256,
        HERE/'NATIVE_WATCH_SPEC.md':args.spec_sha256,HERE/'build.py':args.builder_sha256,
        HERE/'BUILD_PROTOCOL.md':args.build_protocol_sha256,HERE/'watch_build.py':args.build_watcher_sha256,
        HERE/'BUILD_WATCH_SPEC.md':args.build_watch_spec_sha256,TRANSPORT:TRANSPORT_SHA,**h.MON_PINS}
    inputs = {**h.MOCK_PINS,HERE/'build_v1/result.json':args.build_sha256,
        HERE/'build_watch_v1/result.json':args.build_watch_sha256,
        HERE/'build_v1/estimator.py':args.candidate_sha256}
    for path,digest in {**own,**inputs}.items():
        require(h.sha(path) == digest,'Exact prelaunch owner or actual build')
    admission,folder,out = (HERE/name for name in
        ('native_terminal_admission_v1','native_terminal_v1','native_terminal_watch_v1'))
    require(not any(p.exists() for p in (admission,folder,out)) and not HERE.is_symlink(),
        'Exclusive fixed admission/native/watch outputs')
    out.mkdir(exist_ok=False)
    result = dict(schema=SCHEMA,role='terminal',passed=False,completed=False,
        source_bindings={str(p):v for p,v in own.items()},input_bindings={str(p):v for p,v in inputs.items()},artifacts={},
        candidate_sha256=args.candidate_sha256,build_sha256=args.build_sha256,build_watch_sha256=args.build_watch_sha256,
        numerical_imports=False,truth_arrays_accessed=False,score_measured=False,
        full_model_calls=0,retired_calls=0,completed_bill_count=0,
        maximum_combined_rss_bytes=0,maximum_interval_s=0.,maximum_swap_growth_bytes=0.,
        completed_sample_intervals_s=[],phase_records=[],progress=[],whole_owned_tree_retired=False,
        whole_cap_s=600.,admission_cap_s=60.,per_call_cap_s=120.,progress_cap_s=120.,
        sources_unchanged=False,hashes_unchanged=False,child_execution_completed=False)
    child=log=reader=stream=None;children=[];swap0=None;samples=0;pending=''
    phase='admission';phase_began=began;phase_live=False
    def deadline():
        now=time.monotonic()
        require(now-began < 600. and now-progress_at < 120.,'Whole600 and next fully released native call120')
        require(not phase_live or phase != 'admission' or now-phase_began < 60.,'Separate source admission60s')
    def observe():
        nonlocal previous,progress_at,swap0,samples,pending
        absolute=min(previous+1.,began+600.,progress_at+120.)
        if phase_live and phase == 'admission':
            absolute=min(absolute,phase_began+60.)
        snapshot=mon.collect(absolute)
        scope=mon.scope(snapshot['rows'],os.getpid(),None if child is None else child.pid)
        require(not classifier(snapshot['rows'],scope.pop('allowed')),'No competing numerical worker')
        if swap0 is None:swap0=snapshot['swap']['used_bytes']
        growth=snapshot['swap']['used_bytes']-swap0
        rss_cap=2_000_000_000 if phase == 'admission' else 8_000_000_000
        require(growth <= 0 and scope['combined_rss_bytes'] < rss_cap,'Zero added swap and exact phase RSS cap')
        require(snapshot['commands_reaped'] and not mon.PENDING,'Observational commands retired')
        if reader is not None:
            pending+=reader.read();require(len(pending)<4*1024**2,'Bounded native stdout tail')
            while '\n' in pending:
                line,pending=pending.split('\n',1)
                try:item=json.loads(line)
                except (ValueError,TypeError):continue
                if isinstance(item,dict) and item.get('event') == 'retired':
                    deadline();index=result['retired_calls']
                    require(phase == 'native' and item['call'] == index and index < 5,'Ordered original post-release marker')
                    bp=folder/f'call_{index:02d}_bill.json'
                    dp=folder/f'call_{index:02d}_diagnostics.json'
                    bill,diagnostic=h.read(bp),h.read(dp)
                    require(bill['call'] == diagnostic['call'] == index
                        and bill['rank'] == diagnostic['rank'] == SEQUENCE[index]
                        and 0 < bill['flops'] <= 2**41 and bill['wall_s'] < 120.
                        and diagnostic['offline_after_saved_bill_and_context_release']
                        and diagnostic['truth_accessed'] is False,'Actual saved own bill and post-release diagnostics')
                    result['progress'].append(dict(call=index,rank=SEQUENCE[index],bill_sha256=h.sha(bp),
                        diagnostic_sha256=h.sha(dp),observed_elapsed_s=time.monotonic()-began))
                    result['retired_calls']+=1;result['completed_bill_count']=result['retired_calls']
                    progress_at=time.monotonic()
        deadline()
        stream.write(json.dumps(dict(phase=phase,elapsed_s=time.monotonic()-began,**scope,
            swap_bytes=snapshot['swap']['used_bytes'],swap_growth_bytes=growth,retired_calls=result['retired_calls'],
            transport=snapshot['command_records'],uncached_single_process_table=True),allow_nan=False)+'\n');stream.flush()
        previous,interval=mon.complete(previous,absolute)
        result['maximum_combined_rss_bytes']=max(result['maximum_combined_rss_bytes'],scope['combined_rss_bytes'])
        result['maximum_interval_s']=max(result['maximum_interval_s'],interval)
        result['maximum_swap_growth_bytes']=max(result['maximum_swap_growth_bytes'],growth)
        result['completed_sample_intervals_s'].append(interval);samples+=1
    def tick():
        deadline()
        if time.monotonic()-previous >= .2:observe()
    def chain(path,limit):
        files=[path/name for name in ('result.json','completion.json','finalization.json')]
        digests=[h.sha(p,tick) for p in files];r,e,f=map(h.read,files)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == digests[0]
            and f['completion_sha256'] == digests[1] and f['elapsed_s'] < limit,'Actual complete phase three seals')
        result['input_bindings'].update({str(p):v for p,v in zip(files,digests)})
        return r,e,digests
    def command(mode,context_sha=None):
        forwarded=dict(source=args.worker_sha256,model=args.model_sha256,protocol=args.protocol_sha256,
            watcher=args.source_sha256,watch_spec=args.spec_sha256,builder=args.builder_sha256,
            build_protocol=args.build_protocol_sha256,build=args.build_sha256,build_watch=args.build_watch_sha256,
            build_watcher=args.build_watcher_sha256,build_watch_spec=args.build_watch_spec_sha256,
            candidate=args.candidate_sha256)
        if context_sha is not None:forwarded['context']=context_sha
        argv=[str(OFFPY),'-B','-u',str(HERE/'native.py'),'--'+mode,'--role','terminal']
        for key,value in forwarded.items():argv+=['--'+key.replace('_','-')+'-sha256',value]
        return argv
    def supervise(mode,context_sha=None):
        nonlocal child,log,reader,pending,phase_began,phase_live
        argv=command(mode,context_sha);pending=''
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
        env.update({k:'1' for k in BLAS})
        log=(out/(phase+'.log')).open('x');reader=(out/(phase+'.log')).open('r')
        observe();phase_began=time.monotonic();phase_live=True
        child=subprocess.Popen(argv,env=env,cwd=HERE.parent,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(child)
        result['phase_records'].append(dict(phase=phase,pid=child.pid,command=argv,whole_owned_tree_retired=False))
        while child.poll() is None:observe();time.sleep(.15)
        observe();require(child.wait(timeout=1)==0,'Completed '+phase+' child exit')
        mon.cleanup([child],time.monotonic()+.5);phase_live=False
        result['phase_records'][-1].update(whole_owned_tree_retired=True,elapsed_s=time.monotonic()-phase_began,exit_code=0)
        log.close();reader.close();log=reader=None
    def interrupted(*unused):raise RuntimeError('Owned CHECK4 stage interrupted')
    handlers={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        stream=(out/'samples.jsonl').open('x');observe();h.write(out/'started.json',result)
        supervise('admit')
        ar,ae,adigests=chain(admission,60.)
        context_sha=h.sha(admission/'context.json',tick);context=h.read(admission/'context.json')
        require(ar['context_sha256']==context_sha and context['source_admission_complete']
            and ar['admission_pid']==context['admission_pid']==children[0].pid
            and ar['numerical_imports'] is False and ar['truth_accessed'] is False
            and ar['peak_rss_bytes']<2_000_000_000,'Fully retired target-free source admission')
        result.update(admission_context_sha256=context_sha,admission_result_sha256=adigests[0])
        result['input_bindings'][str(admission/'context.json')]=context_sha
        phase='native'
        supervise('run',context_sha)
        r,e,digests=chain(folder,600.)
        require(r['schema']=='current_comp_cstar_proposal_native_v1' and r['role']=='terminal'
            and r['full_model_calls']==r['retired_call_count']==result['retired_calls']==5
            and r['ranks']==list(SEQUENCE[:4]) and r['candidate_sha256']==args.candidate_sha256
            and r['build_sha256']==args.build_sha256 and r['admission_context_sha256']==context_sha
            and all(r.get(k) is True for k in ('own_complete_bills','all16_domains_qualified',
                'all47_selectors_qualified','initialization_ownership_passed','repeat_output_domain_byte_exact',
                'hashes_unchanged','sources_unchanged','full_admission_in_retired_process'))
            and r['truth_accessed'] is False and r['score_measured'] is False,'Complete actual new five native calls')
        require(h.sha(HERE/'native.py',tick)==args.worker_sha256,'Exact saved-file verifier owner after child retirement')
        spec=importlib.util.spec_from_file_location('cprop_native_final_metadata_verifier',HERE/'native.py')
        verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
        authority=SimpleNamespace(**dict(vars(args),source_sha256=args.worker_sha256,
            watcher_sha256=args.source_sha256,watch_spec_sha256=args.spec_sha256))
        result['complete_artifact_verifier_proof']=verifier.verify_native(r,e,digests[0],authority,context,tick)
        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),
            'Full saved-file semantic verifier remained stdlib-only')
        union={}
        for record in (context,r):
            for group in ('source_bindings','input_bindings','artifacts'):
                for path,digest in record.get(group,{}).items():
                    require(path not in union or union[path]==digest,'Consistent native bindings');union[path]=digest
        require(len(r['artifacts'])==30,'Exactly six original artifacts per complete call')
        for path,digest in union.items():require(h.sha(path,tick)==digest,'Fresh source/input/output/bill/diagnostic binding')
        for path,digest in {**own,**inputs}.items():require(h.sha(path,tick)==digest,'Unchanged prospective source/build owners')
        require(all(r['source_bindings'].get(str(p))==v for p,v in own.items()
            if p not in h.MON_PINS and p!=TRANSPORT),
            'New native binds its reviewed required owners')
        require([h.sha(folder/name,tick) for name in ('result.json','completion.json','finalization.json')]==digests,
            'All native seals immutable after retirement and inventory')
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            child_execution_completed=True,whole_owned_tree_retired=True,full_model_calls=5,
            child_result_sha256=digests[0],child_completion_sha256=digests[1],child_finalization_sha256=digests[2],
            all30_artifacts_rehashed=True,all30_artifacts_semantically_verified=True,
            actual_completed_samples=True,foreign_numerical_workers_checked=True,
            verified_child_nontarget_files=len(union))
        observe()
    except BaseException:result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        try:
            mon.cleanup(children,time.monotonic()+.5);mon.retire_pending(time.monotonic()+.5)
            result.update(cleanup_passed=True,whole_owned_tree_retired=True,observational_groups_retired=True)
        except BaseException:result.update(passed=False,completed=False,cleanup_error=traceback.format_exc())
        for handle in (log,reader,stream):
            if handle is not None:handle.close()
        result.update(elapsed_s=time.monotonic()-began,observation_samples=samples,initial_swap_bytes=swap0)
        if result['elapsed_s']>=600. or time.monotonic()-progress_at>=120.:
            result.update(passed=False,completed=False,error='Whole/progress cap includes final cleanup')
        for name in ('started.json','admission.log','native.log','samples.jsonl'):
            path=out/name
            if path.exists():result['artifacts'][str(path)]=h.sha(path)
        h.write(out/'result.json',result);rh=h.sha(out/'result.json')
        ok=bool(result['passed'] and time.monotonic()-began<600. and time.monotonic()-progress_at<120.)
        h.write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh))
        ch=h.sha(out/'completion.json');elapsed=time.monotonic()-began
        ok=bool(ok and elapsed<600. and time.monotonic()-progress_at<120.)
        h.write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,elapsed_s=elapsed))
        h.sha(out/'finalization.json');ok=bool(ok and time.monotonic()-began<600. and time.monotonic()-progress_at<120.)
        for sig,handler in handlers.items():signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','retired_calls','elapsed_s',
        'maximum_combined_rss_bytes','maximum_interval_s','error')},allow_nan=False),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
