"""Independent bounded watch of the fixed APUSH saved-output CHECK4 score."""
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
CANDIDATE_SHA='33879ab58f3a5e4e1ad41deb5db21b73ba703438374225d30fd0ebfc7511b89a'
CAP,RSS=120.,1_000_000_000
BLAS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
      'MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
FORWARD=('protocol','native-source','native-protocol','native-watch-source','native-watch-spec',
         'native','native-watch','admission','build','runtime','candidate')


def require(ok,message):
    if not ok: raise RuntimeError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--numerical-slot-clear',action='store_true')
    keys=('source','worker','spec')+FORWARD
    for key in keys: parser.add_argument('--'+key+'-sha256')
    args=parser.parse_args()
    if not args.run:
        parser.print_help();return 0
    began=previous=time.monotonic()
    require(args.numerical_slot_clear and os.getpriority(os.PRIO_PROCESS,0)==10
        and sys.flags.optimize==0 and sys.dont_write_bytecode
        and Path(sys.prefix).resolve()==OFFPY.parent.parent.resolve()
        and all(os.environ.get(k)=='1' for k in BLAS),'Official serial nice10/BLAS1/assertions/bytecodeoff')
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k.replace('-','_')+'_sha256') or '') for k in keys)
        and args.candidate_sha256==CANDIDATE_SHA,'Reviewed sources and actual completed prerequisite pins')
    require(hashlib.sha256(TRANSPORT.read_bytes()).hexdigest()==TRANSPORT_SHA,'Qualified transport')
    spec=importlib.util.spec_from_file_location('apush_score_transport',TRANSPORT)
    h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
    mon,classifier=h.transport()
    require(not {'numpy','torch','flopscope','threadpoolctl'}&set(sys.modules),'Stdlib-only independent watch')
    mapping={'protocol':HERE/'SCORE_PROTOCOL_V2.md','native-source':HERE/'native.py',
        'native-protocol':HERE/'NATIVE_PROTOCOL.md','native-watch-source':HERE/'watch_native.py',
        'native-watch-spec':HERE/'NATIVE_WATCH_SPEC.md','native':HERE/'native_terminal_v1/result.json',
        'native-watch':HERE/'native_watch_v1/result.json','admission':HERE/'native_terminal_admission_v1/result.json',
        'build':HERE/'build_v1/result.json','runtime':HERE/'runtime_v1/result.json',
        'candidate':HERE/'build_v1/estimator.py'}
    require(set(mapping)==set(FORWARD),'One actual path per forwarded argument')
    requested=dict(run=True,source_sha256=args.worker_sha256,watcher_sha256=args.source_sha256,
        watch_spec_sha256=args.spec_sha256)
    requested.update({k.replace('-','_')+'_sha256':getattr(args,k.replace('-','_')+'_sha256') for k in FORWARD})
    own={Path(__file__).resolve():args.source_sha256,HERE/'score_v2.py':args.worker_sha256,
         HERE/'SCORE_WATCH_SPEC.md':args.spec_sha256,TRANSPORT:TRANSPORT_SHA,**h.MON_PINS}
    supplied={path:getattr(args,k.replace('-','_')+'_sha256') for k,path in mapping.items()}
    inputs={**h.MOCK_PINS,**supplied}
    out=HERE/'score_watch_v1';scored=HERE/'score_v2'
    require(not out.exists() and not scored.exists() and not HERE.is_symlink(),'Exclusive score and watch; no resumption')
    out.mkdir()
    result=dict(schema='current_comp_aligned_push_d6_score_watch_v1',stage='score',passed=False,completed=False,
        source_bindings={str(p):v for p,v in own.items()},input_bindings={str(p):v for p,v in inputs.items()},artifacts={},
        candidate_sha256=CANDIDATE_SHA,full_model_calls=0,parameter_updates=0,numerical_imports=False,
        truth_arrays_decoded=False,truth_bytes_hashed=False,uploaded=False,automatic_next_stage=False,
        whole_cap_s=CAP,combined_rss_limit_bytes=RSS,maximum_combined_rss_bytes=0,
        maximum_swap_growth_bytes=0.,maximum_interval_s=0.,completed_sample_intervals_s=[],
        hashes_unchanged=False,sources_unchanged=False,whole_owned_tree_retired=False)
    child=log=stream=None;swap0=None;samples=0

    def deadline(): require(time.monotonic()-began<CAP,'Whole120s through admission/score/verification/seals')

    def observe():
        nonlocal previous,swap0,samples
        absolute=min(previous+1.,began+CAP)
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
        require(growth<=0 and 0<scope['combined_rss_bytes']<RSS,'Zero added swap and1GB entire owned tree')
        require(snapshot['commands_reaped'] and not mon.PENDING,'Actual reaped observational commands')
        deadline()
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began,**scope,
            swap_bytes=snapshot['swap']['used_bytes'],swap_growth_bytes=growth,
            transport=snapshot['command_records'],uncached_single_process_table=True),allow_nan=False)+'\n')
        stream.flush();previous,interval=mon.complete(previous,absolute)
        result['maximum_combined_rss_bytes']=max(result['maximum_combined_rss_bytes'],scope['combined_rss_bytes'])
        result['maximum_swap_growth_bytes']=max(result['maximum_swap_growth_bytes'],growth)
        result['maximum_interval_s']=max(result['maximum_interval_s'],interval)
        result['completed_sample_intervals_s'].append(interval);samples+=1

    def tick():
        deadline()
        if time.monotonic()-previous>=.2: observe()

    def bind(mapping,target):
        require(isinstance(mapping,dict),'Flat sealed inventory')
        for path,digest in mapping.items():
            require(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}',digest or ''),'Actual immutable binding')
            if target=='truth_bindings':
                expected={P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{r:04d}.npz'
                    for r in (37,39,42,48)}
                require(Path(path) in expected,'Only four already-scored original truth files')
                value=hashlib.sha256()
                with Path(path).open('rb') as stream:
                    for block in iter(lambda:stream.read(1<<20),b''): value.update(block);tick()
                actual=value.hexdigest()
            else: actual=h.sha(path,tick)
            require(actual==digest,'Independent final file rehash')
            require(path not in result[target] or result[target][path]==digest,'Consistent inventory')
            result[target][path]=digest

    def contract(record):
        require(record['schema']=='current_comp_aligned_push_d6_score_v2' and record['stage']=='score'
            and record['requested_seals']==requested and record['candidate_sha256']==CANDIDATE_SHA,
            'Exact fixed14-argument scorer and candidate')
        flags=('passed','completed','integrity_passed','hashes_unchanged','sources_unchanged',
            'all_predictions_verified_before_truth','exact_CD_score_replay','repeat_excluded',
            'no_native_gradient_claim','complete_APUSH_acquisition_reverified','all_five_new_outputs_verified',
            'all_five_domains_and_selectors_reverified','own_complete_bills','runtime_prerequisite_qualified','no_model_imported',
            'saved_external_watch_revalidated','original_external_watch_still_failed','repaired_metadata_only')
        require(all(record.get(k) is True for k in flags),'Complete authenticated four-score result')
        require(record['full_model_calls']==record['parameter_updates']==0 and all(record.get(k) is False for k in
            ('training_occurred','uploaded','automatic_next_stage','production_five_qualified','new_independent14_qualified',
             'submission_authorized','frozen_geometry_price_used_for_scoring','numerical_acquisition_repeated')),
             'No fitted/estimated/production/release substitution')
        require([r['rank'] for r in record['records']]==[37,39,42,48] and 0<record['elapsed_s']<CAP
            and 0<record['peak_rss_bytes']<RSS,'All four exact completed score rows and original worker bounds')
        for path,digest in {HERE/'score_v2.py':args.worker_sha256,Path(__file__).resolve():args.source_sha256,
            HERE/'SCORE_WATCH_SPEC.md':args.spec_sha256,**supplied}.items():
            require(record['input_bindings'].get(str(path))==digest,'Scorer owns every supplied source and receipt')
        seal=scored/'predictions_and_bills_sealed.json'
        require(h.sha(seal,tick)==record['predictions_and_bills_seal_sha256']==record['artifacts'][str(seal)],
            'Exact pretruth prediction/bill seal')
        value=h.read(seal)
        require(value['truth_accessed'] is False and value['ranks']==[37,39,42,48]
            and all(value.get(k) is True for k in ('all_five_new_outputs_verified','all_native_bills_and_inputs_verified',
                'repeat_excluded','own_complete_bills')),'All five frames sealed before any target decode')
        require(record['truth_accessed'] is True and len(record['truth_bindings'])==4,
            'Exactly four original CHECK4 truths used by child')
        comparison=record['comparisons']['terminal_vs_CD']
        require(record['advances_to_native40'] is comparison['gate_passed']
            and comparison['gate_passed'] is (comparison['mean_adjusted_improved'] and comparison['at_least_three_wins']),
            'Strict adjusted objective gate remains separate from operational PASS')

    def interrupted(*unused): raise RuntimeError('Owned fixed CHECK4 score interrupted')
    handlers={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        stream=(out/'samples.jsonl').open('x');observe()
        for path,digest in {**own,**inputs}.items(): require(h.sha(path,tick)==digest,'Reviewed owner or actual prerequisite')
        argv=[str(OFFPY),'-B','-u',str(HERE/'score_v2.py'),'--run']
        for key,value in requested.items():
            if key!='run': argv.extend(['--'+key.replace('_','-'),value])
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
        env.update({k:'1' for k in BLAS})
        log=(out/'worker.log').open('x')
        child=subprocess.Popen(argv,cwd=NEW,env=env,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        result.update(worker_pid=child.pid,worker_command=argv)
        h.write(out/'started.json',result)
        while child.poll() is None: observe();time.sleep(.15)
        observe();require(child.wait(timeout=1)==0,'Fixed scorer successfully exited')
        mon.cleanup([child],time.monotonic()+.5)
        paths=[scored/n for n in ('result.json','completion.json','finalization.json')]
        digests=[h.sha(p,tick) for p in paths];record,end,final=map(h.read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (record,end,final))
            and end['result_sha256']==final['result_sha256']==digests[0]
            and final['completion_sha256']==digests[1] and record['elapsed_s']<=final['elapsed_s']<CAP,
            'Actual complete successful child three-seal chain')
        for p,digest in zip(paths,digests): result['input_bindings'][str(p)]=digest
        contract(record)
        for group in ('source_bindings','input_bindings','artifacts'): bind(record[group],group)
        # These are raw byte hashes of already-scored truths, not numerical decodes.
        result['truth_bindings']={};bind(record['truth_bindings'],'truth_bindings')
        result['truth_bytes_hashed']=True
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            child_execution_completed=True,whole_owned_tree_retired=True,actual_completed_samples=True,
            foreign_numerical_workers_checked=True,child_result_sha256=digests[0],child_completion_sha256=digests[1],
            child_finalization_sha256=digests[2],scientific_gate_passed=record['advances_to_native40'])
        require(not {'numpy','torch','flopscope','threadpoolctl'}&set(sys.modules),'Final watch remains stdlib-only')
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
        for handle in (log,stream):
            if handle is not None: handle.close()
        result.update(elapsed_s=time.monotonic()-began,observation_samples=samples,initial_swap_bytes=swap0,
            last_completed_observation_elapsed_s=previous-began,post_observation_memory_sample_claimed=False)
        for name in ('started.json','worker.log','samples.jsonl'):
            p=out/name
            if p.exists(): result['artifacts'][str(p)]=h.sha(p)
        if time.monotonic()-began>=CAP or time.monotonic()-previous>1.:
            result.update(passed=False,completed=False,error='Whole deadline or terminal cleanup/hash interval>1s')
        h.write(out/'result.json',result);rh=h.sha(out/'result.json')
        ok=bool(result['passed'] and time.monotonic()-began<CAP and time.monotonic()-previous<=1.)
        h.write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh));ch=h.sha(out/'completion.json')
        elapsed=time.monotonic()-began;terminal_interval=time.monotonic()-previous
        ok=bool(ok and elapsed<CAP and terminal_interval<=1.)
        h.write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,
            elapsed_s=elapsed,terminal_publication_interval_s=terminal_interval))
        h.sha(out/'finalization.json');ok=bool(ok and time.monotonic()-began<CAP and time.monotonic()-previous<=1.)
        for sig,handler in handlers.items(): signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','scientific_gate_passed','elapsed_s',
        'maximum_combined_rss_bytes','maximum_interval_s','maximum_swap_growth_bytes','error')},allow_nan=False),flush=True)
    return 0 if ok else 1


if __name__=='__main__':
    raise SystemExit(main())
