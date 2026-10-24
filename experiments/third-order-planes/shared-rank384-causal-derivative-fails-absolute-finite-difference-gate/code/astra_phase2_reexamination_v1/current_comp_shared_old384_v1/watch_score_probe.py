"""One externally bounded saved-output scorer; no target reads or model imports."""
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

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent,HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
TRANSPORT = NEW/'current_comp_cstar_proposal_v1/watch.py'
TRANSPORT_SHA = 'd3258f391ca501698e6c5f14a9dce21f28ae4e63d05731800ce6b2da8c94f23e'
CAP, RSS = 120.,1_000_000_000
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')
KEYS = ('source','worker','protocol','probe-source','probe-watch-source',
        'probe-protocol','probe-watch-spec','probe','probe-watch')


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path,tick=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(P2)
        and 'truthpost' not in path.parts and not (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),
        'Only regular target-free supervised inputs')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            h.update(block)
            if tick is not None:
                tick()
    return h.hexdigest()


def load(path,digest,name):
    require(sha(path) == digest,'Exact metadata source owner')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Supervisor remains stdlib-only')
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true'); parser.add_argument('--numerical-slot-clear',action='store_true')
    for key in KEYS:
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = previous = time.monotonic()
    require(args.numerical_slot_clear and Path(sys.executable).resolve() == OFFPY.resolve()
        and os.getpriority(os.PRIO_PROCESS,0) >= 10 and sys.flags.optimize == 0 and sys.dont_write_bytecode
        and all(os.environ.get(k) == '1' for k in BLAS), 'Exclusive original OFFPY nice10/sixBLAS1 scorer')
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k.replace('-','_')+'_sha256') or '') for k in KEYS),
        'Explicit reviewed sources and actual acquisition result pins')
    h = load(TRANSPORT,TRANSPORT_SHA,'shared_score_transport_only')
    mon,classifier = h.transport()
    own = {HERE/'watch_score_probe.py':args.source_sha256,HERE/'score_probe.py':args.worker_sha256,
        HERE/'SCORE_PROBE_PROTOCOL.md':args.protocol_sha256,TRANSPORT:TRANSPORT_SHA,**h.MON_PINS}
    inputs = {**h.MOCK_PINS,HERE/'probe_v1/result.json':args.probe_sha256,
        HERE/'probe_watch_v1/result.json':args.probe_watch_sha256,HERE/'probe.py':args.probe_source_sha256,
        HERE/'watch_probe.py':args.probe_watch_source_sha256,HERE/'PROBE_PROTOCOL.md':args.probe_protocol_sha256,
        HERE/'PROBE_WATCH_SPEC.md':args.probe_watch_spec_sha256}
    folder,out = HERE/'score_probe_v1',HERE/'score_probe_watch_v1'
    require(not folder.exists() and not out.exists() and not HERE.is_symlink(), 'Exclusive score and watch; no retry/resume')
    out.mkdir(exist_ok=False)
    result = dict(schema='current_comp_shared_old384_score_probe_watch_v1',passed=False,completed=False,
        source_bindings={str(p):v for p,v in own.items()},input_bindings={str(p):v for p,v in inputs.items()},
        artifacts={},numerical_imports=False,truth_arrays_accessed=False,full_model_calls=0,parameter_updates=0,
        uploaded=False,submission_authorized=False,automatic_next_stage=False,information_promising=False,
        whole_cap_s=CAP,combined_rss_limit_bytes=RSS,maximum_combined_rss_bytes=0,
        maximum_interval_s=0.,maximum_swap_growth_bytes=0.,completed_sample_intervals_s=[],
        whole_owned_tree_retired=False,cleanup_passed=False,observational_groups_retired=False,
        hashes_unchanged=False,sources_unchanged=False,numerical_slot_clear_attested=True)
    child=log=stream=None; swap0=None; samples=0
    def deadline():
        require(time.monotonic()-began < CAP,'Whole120s through source admission and final publication')
    def observe():
        nonlocal previous,swap0,samples
        absolute = min(previous+1.,began+CAP)
        snapshot = mon.collect(absolute)
        scope = mon.scope(snapshot['rows'],os.getpid(),None if child is None else child.pid)
        require(not classifier(snapshot['rows'],scope.pop('allowed')),'No foreign numerical worker')
        if swap0 is None:
            swap0 = snapshot['swap']['used_bytes']
        growth = snapshot['swap']['used_bytes']-swap0
        require(growth <= 0 and scope['combined_rss_bytes'] < RSS,'Zero added swap and combined owned-tree1GB')
        require(snapshot['commands_reaped'] and not mon.PENDING,'Every observation group retired')
        deadline()
        stream.write(json.dumps(dict(elapsed_s=time.monotonic()-began,**scope,
            swap_bytes=snapshot['swap']['used_bytes'],swap_growth_bytes=growth,
            transport=snapshot['command_records'],uncached_single_process_table=True),allow_nan=False)+'\n')
        stream.flush()
        previous,interval = mon.complete(previous,absolute)
        result['maximum_combined_rss_bytes'] = max(result['maximum_combined_rss_bytes'],scope['combined_rss_bytes'])
        result['maximum_interval_s'] = max(result['maximum_interval_s'],interval)
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        result['completed_sample_intervals_s'].append(interval); samples += 1
    def tick():
        deadline()
        if time.monotonic()-previous >= .2:
            observe()
    def interrupted(*unused):
        raise RuntimeError('Owned saved-output score interrupted')
    handlers = {sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        stream = (out/'samples.jsonl').open('x'); observe()
        for path,digest in {**own,**inputs}.items():
            require(sha(path,tick) == digest,'Explicit own source/input pin')
        scorer = load(HERE/'score_probe.py',args.worker_sha256,'shared_score_metadata_only')
        forwarded = {k.replace('-','_'):getattr(args,k.replace('-','_')+'_sha256')
            for k in KEYS if k not in ('source','worker')}
        forwarded.update(source=args.worker_sha256,watcher=args.source_sha256)
        argv = [str(OFFPY),'-B','-u',str(HERE/'score_probe.py'),'--run']
        for key,value in forwarded.items():
            argv += ['--'+key.replace('_','-')+'-sha256',value]
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='0')
        env.update({k:'1' for k in BLAS})
        log = (out/'worker.log').open('x')
        child = subprocess.Popen(argv,env=env,cwd=NEW,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result.update(worker_pid=child.pid,worker_command=argv)
        h.write(out/'started.json',result)
        while child.poll() is None:
            observe(); time.sleep(.15)
        observe(); require(child.wait(timeout=1) == 0,'Scorer completed with exit0')
        mon.cleanup([child],time.monotonic()+.5)
        paths = [folder/n for n in ('result.json','completion.json','finalization.json')]
        digests = [sha(p,tick) for p in paths]; r,e,f = map(h.read,paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
            and e['result_sha256'] == f['result_sha256'] == digests[0] and f['completion_sha256'] == digests[1]
            and r['schema'] == scorer.SCHEMA and 0 <= f['elapsed_s'] < CAP and 0 <= r['elapsed_s'] < CAP
            and 0 < r['peak_rss_bytes'] < RSS,'Actual bounded scorer three-seal chain')
        flags = ('integrity_passed','hashes_unchanged','sources_unchanged','complete_probe_acquisition_reverified',
            'all_four_outputs_verified','all16_domains_and_all47_selectors_reverified','own_complete_bills',
            'all_predictions_verified_before_truth','exact_APUSH_score_replay','repeated_outputs_byte_exact',
            'no_model_imported','metadata_fresh_union_rehashed')
        require(all(r.get(k) is True for k in flags) and r['full_model_calls'] == r['parameter_updates'] == 0
            and r['ranks'] == [37] and r['repeat_excluded'] and r['estimator_executed'] is False
            and r['training_or_refit_performed'] is False and r['submission_authorized'] is False
            and r['automatic_next_stage'] is False and type(r['information_promising']) is bool
            and r['information_promising'] == r['scientific_gate_passed'] == (r['comparison']['adjusted_ratio'] <= .95),
            'One exact cold score; scientific FAIL remains a complete operational result')
        require(r['truth_bindings'] == {str(scorer.TRUTH):scorer.TRUTH_SHA} and r['truth_accessed'] is True,
            'Only the fixed target metadata; supervisor never opens it')
        seal = folder/'predictions_and_bills_sealed.json'; sealed = h.read(seal)
        require(sha(seal,tick) == r['predictions_and_bills_seal_sha256'] == r['artifacts'][str(seal)]
            and sealed['truth_accessed'] is False and sealed['all_four_outputs_verified']
            and sealed['own_complete_bills'] and sealed['lineage'] == r['lineage']
            and sealed['source_bindings'] == r['source_bindings'] and sealed['input_bindings'] == r['input_bindings'],
            'Actual immutable all-output pre-truth seal, not only a result flag')
        require(r['requested_seals'] == {'run':True,**{k+'_sha256':v for k,v in forwarded.items()}},
            'Exact forwarded reviewed sources and acquisition authorities')
        union = {}
        for group in ('source_bindings','input_bindings','artifacts'):
            require(isinstance(r[group],dict) and r[group],'Full actual child non-target inventory')
            for path,digest in r[group].items():
                require(path not in union or union[path] == digest,'No conflicting child inventory')
                union[path] = digest
        require(all(r['source_bindings'].get(str(p)) == v for p,v in own.items()
            if p in (HERE/'score_probe.py',HERE/'watch_score_probe.py',HERE/'SCORE_PROBE_PROTOCOL.md')),
            'Exact three current scorer/watch/protocol source owners')
        for path,digest in union.items():
            require(sha(path,tick) == digest,'Independent fresh complete child non-target union')
        for path,digest in {**own,**inputs}.items():
            require(sha(path,tick) == digest,'Own sources and explicit acquisition inputs unchanged')
        require([sha(p,tick) for p in paths] == digests,'Immutable three child seals')
        result['input_bindings'].update(union)
        result['input_bindings'].update({str(p):v for p,v in zip(paths,digests)})
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            child_execution_completed=True,child_result_sha256=digests[0],child_completion_sha256=digests[1],
            child_finalization_sha256=digests[2],information_promising=r['information_promising'],
            scientific_advancement=r['information_promising'],verified_child_nontarget_files=len(union),
            actual_completed_samples=True,foreign_numerical_workers_checked=True)
        observe()
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
    finally:
        try:
            if child is not None:
                mon.cleanup([child],time.monotonic()+.5)
            mon.retire_pending(time.monotonic()+.5)
            result.update(cleanup_passed=True,whole_owned_tree_retired=True,observational_groups_retired=True)
        except BaseException:
            result.update(passed=False,completed=False,cleanup_error=traceback.format_exc())
        for handle in (log,stream):
            if handle is not None:
                handle.close()
        result.update(elapsed_s=time.monotonic()-began,observation_samples=samples,initial_swap_bytes=swap0,
            last_completed_observation_elapsed_s=previous-began,post_observation_memory_sample_claimed=False)
        def final_ok():
            return bool(result['passed'] and time.monotonic()-began < CAP and time.monotonic()-previous <= 1.)
        for name in ('started.json','worker.log','samples.jsonl'):
            path = out/name
            if path.exists():
                result['artifacts'][str(path)] = sha(path)
        result['passed'] = result['completed'] = final_ok()
        h.write(out/'result.json',result); rh = sha(out/'result.json'); ok = final_ok()
        h.write(out/'completion.json',dict(passed=ok,completed=ok,result_sha256=rh)); ch = sha(out/'completion.json')
        elapsed = time.monotonic()-began; ok = bool(ok and final_ok())
        h.write(out/'finalization.json',dict(passed=ok,completed=ok,result_sha256=rh,completion_sha256=ch,
            elapsed_s=elapsed,terminal_publication_interval_s=time.monotonic()-previous))
        sha(out/'finalization.json'); ok = bool(ok and final_ok())
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
    print(json.dumps({k:result.get(k) for k in ('passed','completed','information_promising','elapsed_s',
        'maximum_combined_rss_bytes','maximum_interval_s','error')},allow_nan=False),flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
