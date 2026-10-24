"""One explicit target-blind information worker; stdlib resource supervision.

Operational PASS is not a variance, raw-MSE, cost, or advancement verdict.
No numerical import, worker, test, or automatic next stage occurs at import.
"""
import argparse
from datetime import datetime, timezone
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
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
AP = '/Users/matt/anaconda3/bin/python'
R2 = NEW/'current_p1_native_objective_p2_monitor_v2'
CLASSIFIER = NEW/'current_d50_native_objective_p1_v1/watch.py'
MONITOR_SHA = 'd4b3af53b716593670d8e0c7f0fc6f2133b57f1088d9966c06635d3a23f39873'
CLASSIFIER_SHA = '4e8177d6ff943cdee37e48ccca750bf9d552b32cec344aa2b37e6304cbd55fb3'
MOCK_SHA = 'c5a3f5b8a3dc99ca7e3550fb9ccb474b76968a49b96e00ed7e5606597476bf0e'
MOCK_END_SHA = 'bfb69013c603f0578a56c5e268e3654221b1ab50e00a8bd10159ef074b9e846f'
MOCK_SOURCES = {
    str(R2/'monitor.py'): MONITOR_SHA,
    str(CLASSIFIER): CLASSIFIER_SHA,
    str(R2/'monitor_tests.py'): '1438ca2f4b12b53628cfbb2be4f997eea8949bdf5a70a8c44c74738e83529fb9',
    str(R2/'PROTOCOL.md'): '968910dc63ba1c090665322c945b4e7db3b2978b1a059baa10bab13ea2dc3af7',
}
SCHEMA = 'current_first_layer_quadratic_rqmc_watch_v1'
CHILD_SCHEMA = 'current_first_layer_quadratic_rqmc_information_v1'
RANKS = (35,36,38,100)
ROSTER = [dict(rule_index=16*i+j,rank=r,replicate=j,seed=9309100000+100*r+j,
    tail_seed=(9309100000+100*r+j)^0xA17C9E52D643B80F) for i,r in enumerate(RANKS) for j in range(16)]
LIMIT, PROGRESS, INTERVAL, PERIOD = 600.,120.,1.,.20
RSS_LIMIT, DRIVER_LIMIT = 8_000_000_000,8_000_000_000
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value,message):
    if not value: raise RuntimeError(message)


def digest_string(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def sha(path,tick=None):
    path = Path(path).resolve()
    require(path.is_relative_to(P2) and not (path.suffix in ('.npy','.npz') and 'truth' in str(path).lower()),
        'Supervisor hashes only Phase2 sources, weights and outputs, never truth arrays')
    before = path.stat(); result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            result.update(block)
            if tick: tick()
    after = path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'File changed during hashing')
    return result.hexdigest()


def read(path):
    path = Path(path); require(path.stat().st_size<=16*1024**2,'Bounded JSON metadata')
    return json.loads(path.read_text())


def write(path,value):
    path = Path(path); temporary = path.with_name(path.name+'.publishing')
    with temporary.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')
    os.link(temporary,path); temporary.unlink()


def load(path,name,expected):
    require(sha(path)==expected,'Reviewed stdlib helper source')
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bindings(value,tick=None):
    require(isinstance(value,dict),'Flat absolute-path hash bindings')
    for path,expected in value.items():
        require(isinstance(path,str) and Path(path).is_absolute() and digest_string(expected)
            and sha(path,tick)==expected,'Immutable binding: '+str(path))


def marker(value,entry,folder):
    require(all(type(value.get(k)) is int and value[k]==v for k,v in entry.items()),'Exact next rule identity/seed')
    require(all(value.get(k) is True for k in ('passed','completed','rule_retired','path_state_released',
        'all16_domains_qualified','exact_antipodes')) and value.get('truth_accessed') is False,
        'Only checked, target-blind and fully retired rule markers advance progress')
    r = value['resources']
    require(type(r['peak_rss_bytes']) is int and 0<r['peak_rss_bytes']<RSS_LIMIT
        and type(r['mps_driver_bytes']) is int and 0<=r['mps_driver_bytes']<DRIVER_LIMIT
        and all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0
                for x in (value['elapsed_s'],r['elapsed_s'])),'Cooperative host/driver/time sample')
    info = value['arrays']
    require(Path(info['path']).resolve()==folder/f"rule_{entry['rule_index']:02d}_means.npz"
        and digest_string(info['sha256']) and digest_string(value['coefficient_sha256']),
        'Exact rule-output and frozen-coefficient ownership')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true'); parser.add_argument('--mps-slot-clear',action='store_true')
    for key in ('source','worker','protocol'): parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run: parser.print_help(); return 0
    began = last_progress = last_sample = time.monotonic()
    require(args.mps_slot_clear and all(digest_string(getattr(args,k+'_sha256'))
        for k in ('source','worker','protocol')),'Explicit reviewed hashes and exclusive-slot attestation')
    folder,child_folder = HERE/'information_watch_v1',HERE/'information_v1'
    require(not folder.exists() and not child_folder.exists(),'Exclusive fresh watcher and worker outputs; no resume')
    folder.mkdir(exist_ok=False)
    record = dict(schema=SCHEMA,stage='information',passed=False,completed=False,hashes_unchanged=False,
        sources_unchanged=False,child_execution_completed=False,whole_owned_tree_retired=False,
        source_bindings=MOCK_SOURCES.copy(),input_bindings={},artifacts={},completed_rule_count=0,
        progress=[],samples=0,maximum_sampled_rss_bytes=0,maximum_swap_growth_bytes=0,
        maximum_sample_interval_s=0.,completed_sample_intervals_s=[],truth_accessed=False,
        numerical_imports=False,operational_verdict_only=True,scientific_advancement=False,
        monitor_mock_result_sha256=MOCK_SHA,monitor_scope='Target-free command transport and process classification only',
        limits=dict(whole_s=LIMIT,no_retired_rule_s=PROGRESS,combined_rss_bytes=RSS_LIMIT,
            driver_bytes=DRIVER_LIMIT,added_swap_bytes=0,completed_sample_interval_s=INTERVAL),
        started_utc=datetime.now(timezone.utc).isoformat())
    child = monitor = classifier = None
    resources = stdout = stderr = None
    before = None; roster_digest = None; observed = {}; handlers = {}

    def time_guard():
        now = time.monotonic()
        require(now-began<LIMIT and now-last_progress<PROGRESS,'Whole600s and120s without a retired rule, including sealing')

    def progress():
        nonlocal last_progress,roster_digest
        rp = child_folder/'rule_roster.json'
        if rp.exists():
            current = sha(rp)
            require(roster_digest is None or current==roster_digest,'Immutable predeclared rule roster')
            require(read(rp)==dict(schema=CHILD_SCHEMA,entries=ROSTER),'One fixed four-network/64-rule seed roster')
            roster_digest = current
        found = {p.name for p in child_folder.glob('rule_*_complete.json')} if child_folder.exists() else set()
        valid = {f'rule_{i:02d}_complete.json' for i in range(64)}
        require(found<=valid,'Only the fixed64 completed-rule marker names')
        count = record['completed_rule_count']
        while count<64 and f'rule_{count:02d}_complete.json' in found:
            now = time.monotonic()
            require(now-last_progress<PROGRESS,'Reject a late completed rule BEFORE resetting its deadline')
            require(roster_digest is not None,'Roster published before the first completed rule')
            path = child_folder/f'rule_{count:02d}_complete.json'
            value = read(path); marker(value,ROSTER[count],child_folder)
            digest = sha(path); observed[str(path)] = digest
            record['progress'].append(dict(**ROSTER[count],path=str(path),sha256=digest,
                observed_elapsed_s=now-began,preceding_progress_interval_s=now-last_progress))
            last_progress = now; count += 1; record['completed_rule_count'] = count
        require(not any(f'rule_{i:02d}_complete.json' in found for i in range(count,64)),
            'No out-of-order rule completion')

    def observe():
        nonlocal last_sample
        absolute = min(last_sample+INTERVAL,began+LIMIT,last_progress+PROGRESS)
        snapshot = monitor.collect(absolute)
        memory = monitor.scope(snapshot['rows'],os.getpid(),child.pid if child is not None else None)
        foreign = classifier.other_workers(snapshot['rows'],memory.pop('allowed'))
        growth = snapshot['swap']['used_bytes']-before['used_bytes']
        progress(); now = time.monotonic()
        item = dict(elapsed_s=now-began,**memory,swap=snapshot['swap'],swap_growth_bytes=growth,
            completed_rule_count=record['completed_rule_count'],no_retired_rule_s=now-last_progress,
            foreign_numerical_workers=foreign,transport=snapshot['command_records'],
            one_shared_process_table=True,observational_commands_reaped=snapshot['commands_reaped'],
            cached_observation=False,observation_retry_count=0,
            log_timestamp_scope='Pre-flush; authoritative completed intervals are in result.json')
        record['samples'] += 1
        record['maximum_sampled_rss_bytes'] = max(record['maximum_sampled_rss_bytes'],memory['combined_rss_bytes'])
        record['maximum_swap_growth_bytes'] = max(record['maximum_swap_growth_bytes'],growth)
        if resources is not None:
            resources.write(json.dumps(item,allow_nan=False)+'\n'); resources.flush()
        require(not foreign and memory['combined_rss_bytes']<RSS_LIMIT and growth<=0,
            'Exclusive numerical slot, combined8GB, zero added system swap')
        require(snapshot['commands_reaped'] and not monitor.PENDING,'Every observational group retired')
        time_guard()
        last_sample,interval = monitor.complete(last_sample,absolute)
        record['maximum_sample_interval_s'] = max(record['maximum_sample_interval_s'],interval)
        record['completed_sample_intervals_s'].append(interval)
        return dict(item,completed_elapsed_s=last_sample-began,completed_interval_s=interval)

    def tick():
        time_guard()
        if monitor is not None and before is not None and time.monotonic()-last_sample>=PERIOD: observe()

    try:
        require(not {'numpy','torch','scipy','flopscope','threadpoolctl'} & set(sys.modules),
            'Supervisor is stdlib-only')
        for path,expected in ((Path(__file__).resolve(),args.source_sha256),(HERE/'prototype.py',args.worker_sha256),
                             (HERE/'PROTOCOL.md',args.protocol_sha256)):
            require(sha(path)==expected,'Exact reviewed launch source'); record['source_bindings'][str(path)] = expected
        bindings(MOCK_SOURCES)
        mock_path,end_path = R2/'monitor_tests_v1/result.json',R2/'monitor_tests_v1/completion.json'
        require(sha(mock_path)==MOCK_SHA and sha(end_path)==MOCK_END_SHA,'Immutable target-free monitor mock seals')
        mock,end = read(mock_path),read(end_path)
        require(mock['schema']=='current_p1_native_objective_p2_monitor_tests_v2'
            and all(mock.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged'))
            and all(mock.get(k) is False for k in ('targets_accessed','arrays_accessed','model_accessed','numerical_imports'))
            and mock['source_bindings']==MOCK_SOURCES and end['passed'] is True and end['completed'] is True
            and end['result_sha256']==MOCK_SHA,'Only qualified source/mock transport is inherited')
        record['input_bindings'].update({str(mock_path):MOCK_SHA,str(end_path):MOCK_END_SHA})
        monitor = load(R2/'monitor.py','quadratic_rqmc_monitor',MONITOR_SHA)
        classifier = load(CLASSIFIER,'quadratic_rqmc_worker_classifier',CLASSIFIER_SHA)
        require(not {'numpy','torch','scipy','flopscope','threadpoolctl'} & set(sys.modules),'No transitive numerical imports')
        os.setpriority(os.PRIO_PROCESS,0,max(10,os.getpriority(os.PRIO_PROCESS,0)))
        initial = monitor.collect(min(last_sample+INTERVAL,began+LIMIT,last_progress+PROGRESS))
        scope = monitor.scope(initial['rows'],os.getpid(),None)
        require(not classifier.other_workers(initial['rows'],scope['allowed']) and scope['combined_rss_bytes']<RSS_LIMIT
            and initial['commands_reaped'] and not monitor.PENDING,'Prelaunch free slot and complete observation')
        before = initial['swap']; record['before_swap'] = before
        last_sample,interval = monitor.complete(last_sample,min(last_sample+INTERVAL,began+LIMIT,last_progress+PROGRESS))
        record['completed_sample_intervals_s'].append(interval); record['maximum_sample_interval_s'] = interval
        env = os.environ.copy(); env.update(dict.fromkeys(BLAS,'1'))
        env.update(PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_LOW_WATERMARK_RATIO='0.1')
        argv = [AP,'-u',str(HERE/'prototype.py'),'--run','--source-sha256',args.worker_sha256,
                '--watcher-sha256',args.source_sha256,'--protocol-sha256',args.protocol_sha256]
        write(folder/'started.json',dict(argv=argv,source_bindings=record['source_bindings'],
            input_bindings=record['input_bindings'],exclusive_slot_attested=True,roster=ROSTER,environment={k:env[k] for k in (*BLAS,
                'PYTHONDONTWRITEBYTECODE','PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_LOW_WATERMARK_RATIO')}))
        def stop(signum,frame): raise TimeoutError('Owned information phase interrupted or timed out')
        for sig in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM): handlers[sig] = signal.signal(sig,stop)
        signal.setitimer(signal.ITIMER_REAL,max(.001,LIMIT-(time.monotonic()-began)))
        resources = (folder/'resources.jsonl').open('x')
        stdout = (folder/'stdout.txt').open('x'); stderr = (folder/'stderr.txt').open('x')
        child = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            cwd=str(P2),env=env,start_new_session=True)
        record['child_pid'] = child.pid; record['child_process_group'] = child.pid
        while child.poll() is None:
            observe()
            time.sleep(min(PERIOD,max(0.,last_sample+INTERVAL-monitor.RESERVE-time.monotonic())))
        child.wait(timeout=0); observe()
        require(not monitor.group_alive(child.pid),'Whole worker group retired, not merely its leader')
        stdout.close(); stdout = None; stderr.close(); stderr = None
        require(child.returncode==0 and record['completed_rule_count']==64,'Entire fixed worker and all64 retired rules')
        rp,cp = child_folder/'result.json',child_folder/'completion.json'
        digest = sha(rp,tick); value,end = read(rp),read(cp)
        require(value['schema']==CHILD_SCHEMA and value['stage']=='information'
            and all(value.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
                'numerical_integrity_passed','all64_rules_sealed','all_four_rank_summaries_sealed'))
            and all(value.get(k) is False for k in ('truth_accessed','targets_accessed','training_targets_accessed',
                'validation_targets_accessed','model_fitting','native_score_measured','native_bill_measured','scientific_advancement'))
            and value['gradient_calls']==0 and value['parameter_updates']==0 and end['passed'] is True
            and end['completed'] is True and end['result_sha256']==digest,'Complete target-blind information acquisition')
        require(value['one_rule_paths']==8192 and value['rules_per_rank']==16 and value['ranks']==list(RANKS)
            and len(value['rule_records'])==64 and set(value['rank_summaries'])==set(map(str,RANKS)),
            'No partial or adapted scientific roster')
        for path,expected in ((Path(__file__).resolve(),args.source_sha256),(HERE/'prototype.py',args.worker_sha256),
                             (HERE/'PROTOCOL.md',args.protocol_sha256)):
            require(value['source_bindings'][str(path)]==expected,'Worker bound the exact reviewed current sources')
        for group in ('source_bindings','input_bindings','weight_bindings','artifacts'): bindings(value[group],tick)
        allocation = value['mps_allocator']; recommended = allocation['recommended_max_memory_bytes']
        require(allocation['configured'] is True and type(recommended) is int and recommended>0
            and allocation['configured_fraction']==min(1.,DRIVER_LIMIT/recommended)
            and allocation['api']=='torch.mps.set_per_process_memory_fraction' and allocation['low_watermark_ratio']=='0.1'
            and type(value['maximum_sampled_mps_driver_bytes']) is int
            and 0<value['maximum_sampled_mps_driver_bytes']<DRIVER_LIMIT,'Configured and cooperatively measured8GB MPS cap')
        require(value['environment']['executable']==AP and value['environment']['nice']>=10
            and value['environment']['versions']==dict(numpy='1.26.4',scipy='1.11.1',torch='2.7.0'),
            'Reviewed AP numerical frame, no threadpoolctl dependency')
        for entry,info in zip(ROSTER,value['rule_records']):
            path = child_folder/f"rule_{entry['rule_index']:02d}_complete.json"
            require(all(info[k]==v for k,v in entry.items()) and Path(info['path']).resolve()==path
                and observed[str(path)]==info['sha256']==value['artifacts'][str(path)]==sha(path,tick),
                'Final markers exactly match the observed ordered retired rules')
            marker(read(path),entry,child_folder)
        info = value['rule_roster']
        require(Path(info['path']).resolve()==child_folder/'rule_roster.json'
            and info['sha256']==roster_digest==value['artifacts'][info['path']],'Same predeclared seed roster sealed')
        seal = value['information_seal']; path = child_folder/'information_seal.json'
        require(Path(seal['path']).resolve()==path and sha(path,tick)==seal['sha256']==value['artifacts'][str(path)],
            'Entire fixed information output seal')
        sealed = read(path)
        require(sealed['rule_records']==value['rule_records'] and sealed['rank_summaries']==value['rank_summaries']
            and sealed['truth_accessed'] is False and sealed['scientific_advancement'] is False,
            'All four networks and64 rules sealed without a score claim')
        record['input_bindings'].update({str(rp):digest,str(cp):sha(cp,tick)})
        record.update(child_result_sha256=digest,child_completion_sha256=sha(cp,tick),
            child_execution_completed=True,whole_owned_tree_retired=True,information_seal_sha256=seal['sha256'],
            maximum_sampled_mps_driver_bytes=value['maximum_sampled_mps_driver_bytes'])
        resources.close(); resources = None
        for name in ('started.json','resources.jsonl','stdout.txt','stderr.txt'):
            path = folder/name; record['artifacts'][str(path)] = sha(path,tick)
        bindings(record['source_bindings'],tick); bindings(record['input_bindings'],tick)
        record['final_resource_sample_after_hashing'] = observe()
        require(not monitor.PENDING and not monitor.group_alive(child.pid),'Final complete owned-group retirement')
        record.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True)
    except BaseException:
        record.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
            child_execution_completed=False,error=traceback.format_exc())
    finally:
        if monitor is not None:
            try:
                monitor.retire_pending(time.monotonic()+.30)
                if child is not None: monitor.cleanup([child],time.monotonic()+.50)
                record['whole_owned_tree_retired'] = not monitor.PENDING and (child is None or not monitor.group_alive(child.pid))
            except BaseException:
                record.update(passed=False,completed=False,whole_owned_tree_retired=False,cleanup_error=traceback.format_exc())
        for stream in (resources,stdout,stderr):
            if stream is not None: stream.close()
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in handlers.items(): signal.signal(sig,handler)
    if record['passed']:
        try:
            record['final_resource_sample_after_cleanup'] = observe(); time_guard()
        except BaseException:
            record.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
    record.update(elapsed_s=time.monotonic()-began,finished_utc=datetime.now(timezone.utc).isoformat())
    write(folder/'result.json',record)
    write(folder/'completion.json',dict(passed=record['passed'],completed=record['completed'],
        result_sha256=sha(folder/'result.json')))
    return 0 if record['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
