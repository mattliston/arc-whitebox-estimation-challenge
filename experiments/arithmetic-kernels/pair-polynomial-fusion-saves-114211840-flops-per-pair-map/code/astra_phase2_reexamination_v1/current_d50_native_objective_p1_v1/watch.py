"""One-shot, stdlib-only supervision of the fixed MPS proposal worker.

Qualification and training are separate explicit launches. Only fully checked,
synchronized and retired model-call markers advance the 120s progress clock.
PASS means execution/provenance/resource integrity, never native derivatives,
native accuracy, scientific advancement, or permission to launch another stage.
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
import shlex
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
AP = Path('/Users/matt/anaconda3/bin/python')
CO = NEW/'current_bd_p_coadaptation_v1'
HELPER = NEW/'current_d50_terminal_linear_v2/watch_fit.py'
HELPER_SHA = 'b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd'
PROTOCOL_SHA = '2141e2d4f025522a7cbea55fbd5ff10a66c313feacb1b4bd49a13f75a4b9af35'
MODEL_SHA = '556f3b97613110e638f48127e40112ab203623fa9f6b076b0eab7f07d24658a0'
FAILED_SHA = '91d1f3b56d4ff6f365b6216e3832226ccf71913d4b5618fb8697e4906f1cae7d'
FAILED_END_SHA = '509563919d146b5d66ed652faca6775ecbde7f6d22a03804d9f4c23dd6e8a4a6'
SPLIT_SHA = '22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b'
SCHEMA = 'current_d50_native_objective_p1_v1'
WATCH_SCHEMA = 'current_d50_native_objective_p1_watch_v1'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
TRAIN = tuple(r for r in range(35,235) if r not in VALID)
QUALIFY = ((100,'parent_forward'),(35,'synthetic_backward'),
           (100,'synthetic_backward'),(35,'synthetic_backward'))
HOST_LIMIT, DRIVER_LIMIT, SWAP_LIMIT = 12_000_000_000,16_000_000_000,512*1024**2
PROGRESS_LIMIT, PERIOD, MAX_INTERVAL = 120.0,.2,1.0
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path,tick=None):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),
            'The resource supervisor never opens target arrays')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
            if tick is not None:
                tick()
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def bind(items,tick=None):
    require(isinstance(items,dict),'Flat provenance map required')
    for path,digest in items.items():
        require(isinstance(path,str) and Path(path).is_absolute() and
                re.fullmatch('[0-9a-f]{64}',digest or '') is not None and sha(path,tick) == digest,
                'Immutable binding: '+str(path))


def inventory():
    raw = subprocess.run(['/bin/ps','-axo','pid=,ppid=,pgid=,rss=,command='],
                         capture_output=True,text=True,check=True,timeout=.4).stdout
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        pieces = line.split(None,4)
        require(len(pieces) == 5,'Complete process-command inventory')
        pid,parent,group,rss = map(int,pieces[:4])
        require(pid > 0 and rss >= 0,'Valid process identity/RSS')
        rows.append(dict(pid=pid,ppid=parent,pgid=group,rss_bytes=rss*1024,command=pieces[4]))
    require(any(x['pid'] == os.getpid() for x in rows),'Watcher appears in inventory')
    return rows


def other_workers(rows,allowed):
    """Detect experiment workers; exclusivity beyond argv inference is attested."""
    found = []
    for row in rows:
        if row['pid'] in allowed:
            continue
        try:
            argv = shlex.split(row['command'])
        except ValueError:
            argv = row['command'].split()
        if not argv:
            continue
        name = Path(argv[0]).name.lower()
        python = re.fullmatch(r'(?:python|pypy)[0-9.]*',name) is not None
        module = any(x in ('flopscope','flopscope.server','whestbench') or
                     x.startswith(('flopscope.','whestbench.')) for x in argv[1:])
        script = any(x.endswith('.py') and (str(P2) in x or str(NEW) in x) for x in argv[1:])
        if (python and (module or script)) or name.startswith(('flopscope','whestbench')):
            found.append(row)
    return found


def roster(path,stage):
    value = read(path)
    require(value['schema'] == 'current_d50_native_objective_p1_roster_v1' and
            value['stage'] == stage and value['seed'] == 9309096,'Fixed model-call roster')
    if stage == 'qualify':
        expected = [dict(graph_index=i,rank=r,kind=k,update=None) for i,(r,k) in enumerate(QUALIFY)]
    else:
        order = value['train_order']
        require(value['generator'] == 'PCG64DXSM' and len(order) == 160 and
                all(type(r) is int for r in order) and set(order) == set(TRAIN) and
                value['validation_order'] == list(VALID),'One complete fixed TRAIN shuffle and VALID order')
        expected = [dict(graph_index=i,rank=r,kind='train_backward',update=i//4) for i,r in enumerate(order)]
        expected += [dict(graph_index=160+i,rank=r,kind='validation_forward',update=None)
                     for i,r in enumerate(VALID)]
    require(value['entries'] == expected,'Exact ordered model-call identities and update ownership')
    return value


def marker(value,entry):
    require(all(value.get(k) is True for k in ('passed','completed','graph_retired')) and
            all(value[k] == entry[k] for k in ('graph_index','kind','rank')),
            'Only the next checked and retired model call is progress')
    resources = value['resources']
    require(type(resources['peak_rss_bytes']) is int and 0 < resources['peak_rss_bytes'] < HOST_LIMIT and
            type(resources['mps_driver_bytes']) is int and 0 <= resources['mps_driver_bytes'] < DRIVER_LIMIT and
            all(math.isfinite(x) and x >= 0 for x in (value['elapsed_s'],resources['elapsed_s'])),
            'Finite cooperative graph-completion resource sample')


def completed_child(folder,digest,stage,args,tick=None):
    rp,cp = folder/'result.json',folder/'completion.json'
    require(sha(rp,tick) == digest,'Exact child result seal')
    value,end = read(rp),read(cp)
    require(value['schema'] == SCHEMA and value['stage'] == stage and
            all(value.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
                                             'no_native_gradient_claim')) and
            value['validation_targets_accessed'] is False and end['passed'] is True and
            end['completed'] is True and end['result_sha256'] == digest,'Complete execution-integrity child')
    if stage == 'qualify':
        require(value['training_targets_accessed'] is False and value['execution_integrity_passed'] is True,
                'New qualification is target-free execution integrity')
    else:
        require(value['prerequisites'] == dict(qualification_result_sha256=args.qualification_result_sha256,
                                               qualification_watch_sha256=args.qualification_watch_sha256),
                'Training used exactly the two admitted execution-integrity seals')
    for path,expected in ((HERE/'worker.py',args.worker_sha256),(HERE/'common.py',args.common_sha256),
                          (HERE/'PROTOCOL.md',PROTOCOL_SHA),(Path(__file__).resolve(),args.source_sha256),
                          (CO/'model.py',MODEL_SHA)):
        require(value['source_bindings'][str(path)] == expected,'Child uses the exact reviewed sources')
    for key in ('source_bindings','input_bindings','weight_bindings','artifacts'):
        bind(value[key],tick)
    allocation = value['mps_allocator']
    recommended = allocation['recommended_max_memory_bytes']
    require(allocation['configured'] is True and type(recommended) is int and recommended > 0 and
            allocation['configured_fraction'] == min(1.,DRIVER_LIMIT/recommended) and
            allocation['api'] == 'torch.mps.set_per_process_memory_fraction' and
            allocation['low_watermark_ratio'] == '0.1','Actual prospectively fixed allocator cap')
    require(type(value['maximum_sampled_mps_driver_bytes']) is int and
            0 < value['maximum_sampled_mps_driver_bytes'] < DRIVER_LIMIT,'Cooperative driver limit')
    info = value['model_call_roster']
    path = folder/'model_call_roster.json'
    require(Path(info['path']).resolve() == path and sha(path,tick) == info['sha256'] and
            value['artifacts'][str(path)] == info['sha256'],'Sealed model-call roster')
    entries = roster(path,stage)['entries']
    require(len(value['graph_records']) == len(entries),'All exact model calls completed')
    for entry,info in zip(entries,value['graph_records']):
        path = folder/f"graph_{entry['graph_index']:04d}_complete.json"
        require(Path(info['path']).resolve() == path and sha(path,tick) == info['sha256'] and
                value['artifacts'][str(path)] == info['sha256'],'Exact immutable graph marker')
        marker(read(path),entry)
    return value,end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--stage',choices=('qualify','train'))
    parser.add_argument('--mps-slot-clear',action='store_true')
    for key in ('source','worker','common','protocol','qualification-result','qualification-watch'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = last_progress = time.monotonic()
    require(args.stage is not None,'One explicit stage')
    limit = 300. if args.stage == 'qualify' else 1800.

    def time_guard():
        now = time.monotonic()
        require(now-began < limit,'Whole phase includes admission and final sealing')
        require(now-last_progress < PROGRESS_LIMIT,'120s without the next fully retired model call')

    required = ('source','worker','common','protocol') + (
        ('qualification_result','qualification_watch') if args.stage == 'train' else ())
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '') for k in required),
            'Explicit reviewed source and prerequisite digests')
    require(args.stage == 'train' or (args.qualification_result_sha256 is None and args.qualification_watch_sha256 is None),
            'No training prerequisite selection during qualification')
    require(args.protocol_sha256 == PROTOCOL_SHA and args.mps_slot_clear,'Fixed protocol and sole-MPS attestation')
    expected_env = dict.fromkeys(BLAS,'1')
    expected_env.update(PYTHONDONTWRITEBYTECODE='1',PYTORCH_ENABLE_MPS_FALLBACK='0',
                        PYTORCH_MPS_LOW_WATERMARK_RATIO='0.1')
    require(AP.is_file() and os.getpriority(os.PRIO_PROCESS,0) >= 10 and sys.flags.optimize == 0 and
            sys.dont_write_bytecode and all(os.environ.get(k) == v for k,v in expected_env.items()),
            'Fresh AP/nice10/BLAS1/bytecode-disabled/allocator/no-fallback environment')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope','threadpoolctl')),'Stdlib-only supervisor')
    sources = {str(Path(__file__).resolve()):args.source_sha256,str(HERE/'worker.py'):args.worker_sha256,
        str(HERE/'common.py'):args.common_sha256,str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,
        str(CO/'model.py'):MODEL_SHA,str(HELPER):HELPER_SHA,str(NEW/'joint_prefix_final_data_v1.py'):SPLIT_SHA}
    inputs = {str(CO/'stage_mps_forward_v1/result.json'):FAILED_SHA,
              str(CO/'stage_mps_forward_v1/completion.json'):FAILED_END_SHA}
    bind(sources,time_guard); bind(inputs,time_guard)
    failed,end = read(CO/'stage_mps_forward_v1/result.json'),read(CO/'stage_mps_forward_v1/completion.json')
    require(failed['stage'] == 'mps_forward' and failed['passed'] is False and failed['completed'] is False and
            failed['alltruth_accessed'] is False and end['passed'] is False and end['result_sha256'] == FAILED_SHA,
            'The preserved CO numerical failure remains FAILED evidence')
    if args.stage == 'train':
        folder,previous_folder = HERE/'qualification_v1',HERE/'qualification_watch_v1'
        for path,digest in ((folder/'result.json',args.qualification_result_sha256),
                            (previous_folder/'result.json',args.qualification_watch_sha256)):
            inputs[str(path)] = digest
        for path in (folder/'completion.json',previous_folder/'completion.json'):
            inputs[str(path)] = sha(path,time_guard)
        bind(inputs,time_guard)
        completed_child(folder,args.qualification_result_sha256,'qualify',args,time_guard)
        old,old_end = read(previous_folder/'result.json'),read(previous_folder/'completion.json')
        require(old['schema'] == WATCH_SCHEMA and old['stage'] == 'qualify' and
                all(old.get(k) is True for k in ('passed','completed','hashes_unchanged','sources_unchanged',
                                               'child_execution_completed','whole_owned_tree_retired')) and
                old['child_result_sha256'] == args.qualification_result_sha256 and
                old['child_completion_sha256'] == inputs[str(folder/'completion.json')] and
                old['completed_graph_count'] == 4 and old_end['passed'] is True and old_end['completed'] is True and
                old_end['result_sha256'] == args.qualification_watch_sha256,'New qualified execution and independent watch')
        require(all(old['source_bindings'].get(k) == v for k,v in sources.items()),'Same fixed supervisor/worker frame')
        for key in ('source_bindings','input_bindings','artifacts'):
            bind(old[key],time_guard)
    child_out = HERE/('qualification_v1' if args.stage == 'qualify' else 'training_v1')
    out = HERE/('qualification_watch_v1' if args.stage == 'qualify' else 'training_watch_v1')
    require(not HERE.is_symlink() and not child_out.exists() and not out.exists(),'Exclusive one-shot paths')
    out.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('p1_owned_resource_helpers',HELPER)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    command = [str(AP),'-B','-u',str(HERE/'worker.py'),'--run','--stage',args.stage,
        '--source-sha256',args.worker_sha256,'--common-sha256',args.common_sha256,
        '--protocol-sha256',PROTOCOL_SHA,'--watcher-sha256',args.source_sha256]
    if args.stage == 'train':
        command += ['--qualification-result-sha256',args.qualification_result_sha256,
                    '--qualification-watch-sha256',args.qualification_watch_sha256]
    result = dict(schema=WATCH_SCHEMA,stage=args.stage,passed=False,completed=False,hashes_unchanged=False,
        sources_unchanged=False,source_bindings=sources,input_bindings=inputs,artifacts={},command=command,
        prerequisites=dict(qualification_result_sha256=args.qualification_result_sha256,
                           qualification_watch_sha256=args.qualification_watch_sha256),
        failed_CO_result_sha256=FAILED_SHA,failed_CO_remains_failed=True,no_native_gradient_claim=True,
        native_objective_measured=False,scientific_advancement=False,auto_launch_next_stage=False,
        mps_slot_clear_attested=True,numerical_imports=False,truth_arrays_opened=False,
        inherited_environment=expected_env,wall_limit_s=limit,combined_rss_limit_bytes=HOST_LIMIT,
        driver_limit_bytes=DRIVER_LIMIT,swap_growth_limit_bytes=SWAP_LIMIT,no_completed_graph_limit_s=PROGRESS_LIMIT,
        requested_sample_period_s=PERIOD,sample_interval_limit_s=MAX_INTERVAL,
        maximum_sampled_rss_bytes=0,maximum_swap_growth_bytes=0.,maximum_sample_interval_s=0.,samples=0,
        first_progress_elapsed_s=None,last_progress_elapsed_s=None,completed_graph_count=0,graph_progress=[],
        child_execution_completed=False,whole_owned_tree_retired=False,
        limitations=['RSS/swap are sampled, not absolute peaks; driver allocation is cooperatively measured by the child.',
            'Driver and host allocations overlap on unified memory.',
            'Process argv identifies experiment scripts/services; broader exclusivity remains an explicit launch attestation.',
            'The fixed worker computes PCG64DXSM; supervisor independently checks unique TRAIN/exact VALID and marker order.',
            'An integrity PASS does not certify native gradients, scores, release, or the next stage.'],
        started_utc=datetime.now(timezone.utc).isoformat())
    child,before,call_roster,roster_sha = None,None,None,None
    handlers = {}; last_sample = time.monotonic()
    seen_records = []

    def interrupted(signum,frame):
        raise RuntimeError('Proposal supervisor interrupted: '+str(signum))

    def progress():
        nonlocal last_progress,call_roster,roster_sha
        rp = child_out/'model_call_roster.json'
        if call_roster is None and rp.exists():
            require(not rp.is_symlink(),'Owned regular roster path')
            call_roster,roster_sha = roster(rp,args.stage),sha(rp,time_guard)
        found = []
        for path in child_out.glob('graph_*_complete.json'):
            match = re.fullmatch(r'graph_([0-9]{4})_complete\.json',path.name)
            require(match is not None and not path.is_symlink(),'Exact published model-call marker name')
            found.append(int(match[1]))
        expected_count = 4 if args.stage == 'qualify' else 200
        require(sorted(found) == list(range(len(found))) and len(found) <= expected_count,
                'Unique contiguous graph-completion sequence')
        require(not found or call_roster is not None,'Model-call roster published before any completion')
        for index in range(result['completed_graph_count'],len(found)):
            path = child_out/f'graph_{index:04d}_complete.json'
            value = read(path)  # Worker publishes atomically after closing the complete JSON.
            marker(value,call_roster['entries'][index])
            digest = sha(path,time_guard)
            observed = time.monotonic()
            require(observed-last_progress < PROGRESS_LIMIT and observed-began < limit,
                    'Late completion cannot reset an already expired deadline')
            last_progress = observed
            elapsed = observed-began
            if result['first_progress_elapsed_s'] is None:
                result['first_progress_elapsed_s'] = elapsed
            result['last_progress_elapsed_s'] = elapsed
            result['completed_graph_count'] = index+1
            result['graph_progress'].append(dict(graph_index=index,rank=value['rank'],kind=value['kind'],
                path=str(path),sha256=digest,observed_elapsed_s=elapsed))
            seen_records.append(value)

    def observe(stream=None,active=True):
        nonlocal last_sample
        rows = inventory()
        numeric = [(x['pid'],x['ppid'],x['pgid'],x['rss_bytes']//1024) for x in rows]
        owned = set()
        if child is not None:
            owned = helper.descendants({child.pid},numeric) | {x['pid'] for x in rows if x['pgid'] == child.pid}
            owned = helper.descendants(owned,numeric)
        combined = helper.descendants({os.getpid()},numeric) | owned
        memory = sum(x['rss_bytes'] for x in rows if x['pid'] in combined)
        allowed = combined | {os.getpid()}
        parent_map = {x['pid']:x['ppid'] for x in rows}
        ancestor = os.getpid()
        while ancestor in parent_map and parent_map[ancestor] not in allowed:
            ancestor = parent_map[ancestor]; allowed.add(ancestor)
        foreign = other_workers(rows,allowed)
        swap = helper.swap_state(); growth = swap['used_bytes']-before['used_bytes']
        progress()
        now = time.monotonic(); interval,last_sample = now-last_sample,now
        item = dict(elapsed_s=now-began,interval_s=interval,combined_rss_bytes=memory,
            watcher_rss_bytes=sum(x['rss_bytes'] for x in rows if x['pid'] == os.getpid()),
            child_tree_rss_bytes=sum(x['rss_bytes'] for x in rows if x['pid'] in owned),
            child_pids=sorted(x['pid'] for x in rows if x['pid'] in owned),swap=swap,swap_growth_bytes=growth,
            completed_graph_count=result['completed_graph_count'],no_completed_graph_s=now-last_progress,
            other_detected_workers=foreign,live_child_interval_gate=active)
        result['samples'] += 1
        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],memory)
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        if active:
            result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
        if stream is not None:
            stream.write(json.dumps(item,allow_nan=False)+'\n'); stream.flush()
        require(not foreign,'Another identifiable numerical experiment worker is active; stop only our owned child')
        require(memory < HOST_LIMIT and growth <= SWAP_LIMIT,'Independent combined host RSS/system-swap limit')
        require(not active or interval <= MAX_INTERVAL,'Observed live-child sampling interval<=1s')
        time_guard()
        return item

    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig] = signal.signal(sig,interrupted)
        before = helper.swap_state(); result['before_swap'] = before
        result['prelaunch_process_inventory'] = inventory()
        write(out/'started.json',result)
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr,\
                (out/'resources.jsonl').open('x') as resources:
            observe(resources,active=False)
            last_sample = time.monotonic()
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                     env=os.environ.copy(),start_new_session=True)
            result['child_pid'] = result['child_process_group'] = child.pid
            while True:
                code = child.poll()
                item = observe(resources)
                require(item['child_pids'] or code is not None or child.poll() is not None,'Live child tree observable')
                if code is not None:
                    result['child_exit_code'] = code
                    require(not item['child_pids'],'No surviving owned descendants')
                    result['whole_owned_tree_retired'] = True
                    break
                time.sleep(PERIOD)
        require(result['child_exit_code'] == 0,'Proposal worker did not complete successfully')
        rp,cp = child_out/'result.json',child_out/'completion.json'
        result.update(child_result_sha256=sha(rp,time_guard),child_completion_sha256=sha(cp,time_guard))
        record,end = completed_child(child_out,result['child_result_sha256'],args.stage,args,time_guard)
        require(result['completed_graph_count'] == (4 if args.stage == 'qualify' else 200),
                'All exact completed model calls were observed')
        require(record['model_call_roster']['sha256'] == roster_sha,'Observed roster is the final roster')
        for item,value,info in zip(result['graph_progress'],seen_records,record['graph_records']):
            require(read(item['path']) == value and sha(item['path'],time_guard) == item['sha256'] == info['sha256'],
                    'Every observed completed model call is immutable')
        result.update(child_execution_completed=True,passed=True,completed=True,
            model_call_roster_sha256=roster_sha,maximum_sampled_mps_driver_bytes=record['maximum_sampled_mps_driver_bytes'],
            mps_allocator=record['mps_allocator'])
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
        if child is not None:
            try:
                helper.stop_owned_group(child); result['whole_owned_tree_retired'] = True
                result['child_exit_code'] = child.poll()
            except BaseException:
                result['cleanup_error'] = traceback.format_exc()
    finally:
        try:
            tick = time_guard if result['passed'] else None
            for path in (out/'started.json',out/'stdout.log',out/'stderr.log',out/'resources.jsonl',
                         child_out/'result.json',child_out/'completion.json',child_out/'model_call_roster.json'):
                if path.is_file():
                    result['artifacts'][str(path)] = sha(path,tick)
            for item in result['graph_progress']:
                result['artifacts'][item['path']] = sha(item['path'],tick)
            bind(sources,tick); bind(inputs,tick)
            result.update(hashes_unchanged=True,sources_unchanged=True)
            if result['passed']:
                result['final_resource_observation_after_hashing'] = observe(active=False)
                require(not result['final_resource_observation_after_hashing']['child_pids'],'No descendants after sealing')
                require(sha(child_out/'result.json',time_guard) == result['child_result_sha256'] and
                        sha(child_out/'completion.json',time_guard) == result['child_completion_sha256'],
                        'Final child result/completion unchanged')
        except BaseException:
            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
                          final_sealing_error=traceback.format_exc())
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
        result.update(elapsed_s=time.monotonic()-began,finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/'result.json',result)
        write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            hashes_unchanged=result['hashes_unchanged'],sources_unchanged=result['sources_unchanged'],
            result_sha256=sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('stage','passed','completed','elapsed_s','completed_graph_count',
        'maximum_sampled_rss_bytes','maximum_sampled_mps_driver_bytes','error','final_sealing_error')},
        allow_nan=False),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
