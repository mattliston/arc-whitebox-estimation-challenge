"""One explicit P1 null, native40, or actual-objective score supervision.

Stdlib only; no automatic next stage. A completed scientific non-advance
is an operational score PASS, never a scientific or release qualification.
Native progress means the next saved complete bill AND its closed/reset
context receipt, not a resource heartbeat or arbitrary file modification.
"""
import argparse
import ast
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
OFFPY = P2/'data_gate/.venv/bin/python'
TL = NEW/'current_d50_terminal_linear_v2'
HELPER = TL/'watch_fit.py'
HELPER_SHA = 'b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd'
TEMPLATE_SHA = 'b8b8d7563487c66e1424fbbc6dd34811bc8836cde1abbb5bca475f29227f7e44'
BUILD_SHA = '57f87081cdcc742266bc8487ef5ba11358bdf25e8341b9b1f4dd4458f06da54c'
PROTOCOL_SHA = '2141e2d4f025522a7cbea55fbd5ff10a66c313feacb1b4bd49a13f75a4b9af35'
BANK = NEW/'birth_direction_fit_v1/build_terminal_v1/estimator.py'
BANK_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
WATCH_SCHEMA = 'current_d50_native_objective_p1_native_watch_v1'
NATIVE_SCHEMA = 'current_d50_native_objective_p1_native_v1'
SCORE_SCHEMA = 'current_d50_native_objective_p1_score_v1'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
INITIALIZATION, BUDGET = 1383913786, 2**41
PERIOD, MAX_INTERVAL, PROGRESS_LIMIT = .2, .5, 120.
SWAP_LIMIT = 512*1024**2
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value,message):
    if not value:
        raise RuntimeError(message)


def sha(path,tick=None):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),
            'Supervisor never opens target arrays')
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


def bind(mapping,tick=None):
    require(isinstance(mapping,dict),'Flat immutable inventory')
    for path,digest in mapping.items():
        require(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}',digest or '') and
                sha(path,tick) == digest,'Immutable binding: '+path)


def completed(folder,digest,tick=None):
    require(sha(folder/'result.json',tick) == digest,'Exact result seal')
    record,end = read(folder/'result.json'),read(folder/'completion.json')
    require(record['completed'] is True and record['hashes_unchanged'] is True and
            end['completed'] is True and end['passed'] is record['passed'] and
            end['result_sha256'] == digest,'Complete integrity receipt')
    return record,end


def literal_span(source):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node,ast.Assign) and len(node.targets) == 1 and
             isinstance(node.targets[0],ast.Name) and node.targets[0].id == '_GNN_LITERAL']
    require(len(nodes) == 1,'Unique existing P assignment')
    node = nodes[0]
    require(isinstance(node.value,ast.Dict),'P remains a dictionary literal')
    lines = source.splitlines(keepends=True)
    return (sum(map(len,lines[:node.lineno-1]))+node.col_offset,
            sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset)


def build_admission(digest,sources,inputs,tick):
    folder = HERE/'build_v1'
    value,end = completed(folder,digest,tick)
    require(value['schema'] == 'current_d50_native_objective_p1_v1' and value['stage'] == 'build' and
            value['passed'] is True and value['sources_unchanged'] is True and
            value['terminal_epoch'] == 1 and value['updates'] == 40 and
            value['parent_sha256'] == value['null_sha256'] == BANK_SHA and
            value['modified_assignment_only'] == '_GNN_LITERAL' and
            value['validation_targets_accessed'] is False and
            all(value.get(k) is True for k in ('inverse_source_byte_exact','null_source_byte_exact',
                'non_PKEYS_byte_exact','unused_layer0_byte_exact','cstar_byte_exact',
                'terminal_literal_roundtrip_byte_exact')),'Sole fixed P-only terminal build')
    require(value['source_bindings'][str(HERE/'build.py')] == BUILD_SHA and
            value['source_bindings'][str(HERE/'PROTOCOL.md')] == PROTOCOL_SHA,
            'Reviewed literal builder and prospective protocol')
    for key in ('source_bindings','input_bindings','artifacts'):
        bind(value[key],tick)
        destination = sources if key == 'source_bindings' else inputs
        for path,item in value[key].items():
            require(path not in destination or destination[path] == item,'No conflicting provenance')
            destination[path] = item
    inputs[str(folder/'result.json')] = digest
    inputs[str(folder/'completion.json')] = sha(folder/'completion.json',tick)
    candidate,null = folder/'estimator.py',folder/'null_estimator.py'
    require(value['artifacts'][str(candidate)] == value['candidate_sha256'] and
            value['artifacts'][str(null)] == BANK_SHA,'Exact sealed native sources')
    parent,terminal = BANK.read_bytes(),candidate.read_bytes()
    require(sha(BANK,tick) == BANK_SHA and null.read_bytes() == parent,'Null source byte-exact to d50')
    a,b = literal_span(parent); x,y = literal_span(terminal)
    require(terminal[:x]+parent[a:b]+terminal[y:] == parent,'Independent exact P-only source inverse')
    tick()
    return value


def native_receipt(folder,digest,role,build,args,tick):
    value,end = completed(folder,digest,tick)
    ranks = [VALID[0],VALID[0]] if role == 'null' else list(VALID)+[VALID[0]]
    candidate_sha = BANK_SHA if role == 'null' else build['candidate_sha256']
    require(value['schema'] == NATIVE_SCHEMA and value['stage'] == 'native' and value['role'] == role and
            value['passed'] is True and value['truth_accessed'] is False and
            value['candidate_sha256'] == candidate_sha and value['initialization_flops'] == INITIALIZATION and
            value['prerequisites']['build_result_sha256'] == args.build_sha256 and
            [t['rank'] for t in value['trials']] == ranks and
            len(value['domain_diagnostics']) == len(ranks) and len(value['artifacts']) == 5*len(ranks) and
            all(value.get(k) is True for k in ('all16_domains_qualified','all47_selectors_qualified',
                'own_complete_bills','all_outputs_and_bills_sealed_before_scoring')),'Qualified actual native roster')
    if role == 'null':
        require(value['null_full_output_bill_parity'] is True,'Measured full null byte/bill parity')
    native_sha = args.native_source_sha256 if args.stage == 'score' else args.worker_sha256
    require(value['source_bindings'][str(HERE/'native.py')] == native_sha and
            value['source_bindings'][str(HERE/'PROTOCOL.md')] == PROTOCOL_SHA,'Same native source/protocol')
    for key in ('source_bindings','artifacts'):
        bind(value[key],tick)
    return value,end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--stage',choices=('null','native','score'))
    for name in ('source','worker','build','protocol','native-source','native-result','native-watch'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    started = last_progress = time.monotonic()
    require(args.stage is not None,'One explicit stage; no automatic pipeline')
    scoring = args.stage == 'score'
    wall_limit,rss_limit = (120.,2_000_000_000) if scoring else (1800.,8_000_000_000)

    def time_guard():
        now = time.monotonic()
        require(now-started < wall_limit,'Whole phase includes admission and final sealing')
        require(scoring or now-last_progress < PROGRESS_LIMIT,'120s without next complete native bill lifecycle')

    required = ('source','worker','build','protocol') + (('native_source','native_result','native_watch') if scoring else ())
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '') for k in required),'Reviewed source/seal digests')
    require(scoring or all(getattr(args,k) is None for k in
            ('native_source_sha256','native_result_sha256','native_watch_sha256')),'No score-only inputs to acquisition')
    require(args.protocol_sha256 == PROTOCOL_SHA and os.getpriority(os.PRIO_PROCESS,0) >= 10 and
            sys.flags.optimize == 0 and sys.dont_write_bytecode and OFFPY.is_file() and
            all(os.environ.get(k) == '1' for k in BLAS) and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1' and
            os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0','nice10/BLAS1/assertions/bytecode-disabled/no-fallback')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope','threadpoolctl')),'Stdlib-only resource supervisor')
    worker = HERE/('judge.py' if scoring else 'native.py')
    child_out = HERE/('score_v1' if scoring else ('native_null_v1' if args.stage == 'null' else 'native_terminal_v1'))
    out = HERE/('score_watch_v1' if scoring else ('native_null_watch_v1' if args.stage == 'null' else 'native_terminal_watch_v1'))
    sources = {str(Path(__file__).resolve()):args.source_sha256,str(worker):args.worker_sha256,
        str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,str(HERE/'build.py'):BUILD_SHA,str(BANK):BANK_SHA,
        str(HELPER):HELPER_SHA,str(TL/'watch_validation.py'):TEMPLATE_SHA}
    inputs = {}; bind(sources,time_guard)
    build = build_admission(args.build_sha256,sources,inputs,time_guard)
    prerequisites = dict(build_sha256=args.build_sha256)
    if args.stage in ('native','score'):
        role = 'null' if args.stage == 'native' else 'terminal'
        prior = HERE/f'native_{role}_v1'; prior_watch = HERE/f'native_{role}_watch_v1'
        digest = sha(prior/'result.json',time_guard) if role == 'null' else args.native_result_sha256
        watch_digest = sha(prior_watch/'result.json',time_guard) if role == 'null' else args.native_watch_sha256
        native_receipt(prior,digest,role,build,args,time_guard)
        watch,end = completed(prior_watch,watch_digest,time_guard)
        require(watch['schema'] == WATCH_SCHEMA and watch['stage'] == ('null' if role == 'null' else 'native') and
                all(watch.get(k) is True for k in ('passed','sources_unchanged','child_execution_completed',
                    'whole_owned_tree_retired')) and watch['child_result_sha256'] == digest and
                watch['child_completion_sha256'] == sha(prior/'completion.json',time_guard) and
                watch['prerequisites']['build_sha256'] == args.build_sha256 and
                watch['source_bindings'][str(Path(__file__).resolve())] == args.source_sha256 and
                watch['completed_native_bill_count'] == (2 if role == 'null' else 41),'Same-build completed independent native watch')
        for key in ('source_bindings','input_bindings','artifacts'):
            bind(watch[key],time_guard)
        for folder in (prior,prior_watch):
            for name in ('result.json','completion.json'):
                inputs[str(folder/name)] = sha(folder/name,time_guard)
        prerequisites.update(native_result_sha256=digest,native_watch_sha256=watch_digest,native_role=role)
        if scoring:
            sources[str(HERE/'native.py')] = args.native_source_sha256
            bind({str(HERE/'native.py'):args.native_source_sha256},time_guard)
    require(not out.exists() and not child_out.exists(),'Exclusive source-bound stage paths; no resume')
    out.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('p1_native_owned_resource_helpers',HELPER)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    command = [str(OFFPY),'-B','-u',str(worker),'--run','--source-sha256',args.worker_sha256,
               '--build-sha256',args.build_sha256,'--protocol-sha256',PROTOCOL_SHA]
    if scoring:
        for name in ('native-source','native-result','native-watch'):
            command += ['--'+name+'-sha256',getattr(args,name.replace('-','_')+'_sha256')]
    else:
        command += ['--role','null' if args.stage == 'null' else 'terminal']
    ranks = [] if scoring else ([37,37] if args.stage == 'null' else list(VALID)+[37])
    result = dict(schema=WATCH_SCHEMA,stage=args.stage,passed=False,completed=False,hashes_unchanged=False,
        sources_unchanged=False,source_bindings=sources,input_bindings=inputs,artifacts={},prerequisites=prerequisites,
        command=command,truth_arrays_opened=False,numerical_imports=False,auto_launch_next_stage=False,
        child_execution_completed=False,whole_owned_tree_retired=False,scientific_advancement=False,
        wall_limit_s=wall_limit,combined_rss_limit_bytes=rss_limit,swap_growth_limit_bytes=SWAP_LIMIT,
        requested_sample_period_s=PERIOD,sample_interval_limit_s=MAX_INTERVAL,
        no_completed_native_bill_limit_s=None if scoring else PROGRESS_LIMIT,
        completed_native_bill_count=0,native_bill_progress=[],first_progress_elapsed_s=None,last_progress_elapsed_s=None,
        maximum_sample_interval_s=0.,maximum_sampled_rss_bytes=0,maximum_swap_growth_bytes=0.,samples=0,
        started_utc=datetime.now(timezone.utc).isoformat(),
        limitations=['RSS/swap are sampled, not continuous peaks; no MPS is used by the child.',
            'Native bill/reset completion is not a claim that every harness temporary has been reclaimed.',
            'This operational PASS is not production, independent-panel, or native-gradient qualification.'])
    child,before = None,None
    handlers = {}; last_sample = time.monotonic()

    def interrupted(signum,frame):
        raise RuntimeError('Native supervisor interrupted '+str(signum))

    def progress():
        nonlocal last_progress
        if scoring:
            return
        found = []
        for path in child_out.glob('call_*_bill.json'):
            if path.name.endswith('_input_bill.json'):
                continue
            match = re.fullmatch(r'call_([0-9]{2})_bill\.json',path.name)
            require(match is not None and not path.is_symlink(),'Exact owned native bill name')
            found.append(int(match[1]))
        require(sorted(found) == list(range(len(found))) and len(found) <= len(ranks),'Unique ordered native bills')
        for index in range(result['completed_native_bill_count'],len(found)):
            bp,lp = child_out/f'call_{index:02d}_bill.json',child_out/f'call_{index:02d}_lifecycle.json'
            if not lp.exists():
                break
            require(not lp.is_symlink(),'Owned lifecycle path')
            try:
                bill,life = read(bp),read(lp)
            except json.JSONDecodeError:
                break  # Original native writer is not atomic; only closed parseable receipts count.
            require(bill['call'] == life['call'] == index and bill['rank'] == life['rank'] == ranks[index] and
                    Path(life['saved_bill']).resolve() == bp and life['saved_bill_sha256'] == sha(bp,time_guard) and
                    bill['lifecycle_receipt'] == str(lp) and life['candidate_state_reset'] is False and
                    life['reset_api'] == 'flopscope.budget_reset' and
                    all(life.get(k) is True for k in ('after_context_exit','closed_context_bill_unchanged_by_reset',
                        'context_released','complete')),'Next complete saved bill and released context')
            require(type(bill['flops']) is int and 0 < bill['flops'] <= BUDGET and
                    all(math.isfinite(bill[k]) and 0 <= bill[k] < 120 for k in ('wall_s','context_wall_s')) and
                    0 < bill['peak_rss_bytes'] < 8_000_000_000 and
                    life['closed_metrics']['flops'] == bill['flops'],'Native full bill resource ownership')
            bsha,lsha = sha(bp,time_guard),sha(lp,time_guard)
            now = time.monotonic()
            require(now-last_progress < PROGRESS_LIMIT and now-started < wall_limit,
                    'Late native completion cannot reset an expired interval')
            last_progress = now
            elapsed = now-started
            if result['first_progress_elapsed_s'] is None:
                result['first_progress_elapsed_s'] = elapsed
            result['last_progress_elapsed_s'] = elapsed
            result['completed_native_bill_count'] = index+1
            result['native_bill_progress'].append(dict(call=index,rank=ranks[index],bill=str(bp),bill_sha256=bsha,
                lifecycle=str(lp),lifecycle_sha256=lsha,observed_elapsed_s=elapsed))

    def observe(stream=None,active=True):
        nonlocal last_sample
        memory = helper.memory_snapshot(child.pid)
        swap = helper.swap_state(); growth = swap['used_bytes']-before['used_bytes']
        progress()
        now = time.monotonic(); interval,last_sample = now-last_sample,now
        memory.update(elapsed_s=now-started,interval_s=interval,swap=swap,swap_growth_bytes=growth,
            completed_native_bill_count=result['completed_native_bill_count'],
            no_completed_native_bill_s=None if scoring else now-last_progress,live_child_interval_gate=active)
        result['samples'] += 1
        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],memory['combined_rss_bytes'])
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        if active:
            result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
        if stream is not None:
            stream.write(json.dumps(memory,allow_nan=False)+'\n'); stream.flush()
        require(memory['combined_rss_bytes'] < rss_limit and growth <= SWAP_LIMIT,'Combined watcher/tree RSS and swap')
        require(not active or interval <= MAX_INTERVAL,'Observed live-child sampling interval<=.5s')
        time_guard()
        return memory

    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig] = signal.signal(sig,interrupted)
        before = helper.swap_state(); result['before_swap'] = before
        result['prelaunch_process_inventory'] = helper.process_table()
        write(out/'started.json',result)
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr,\
                (out/'resources.jsonl').open('x') as resources:
            time_guard(); last_sample = time.monotonic()
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                env=os.environ.copy(),start_new_session=True)
            result['child_pid'] = result['child_process_group'] = child.pid
            while True:
                code = child.poll(); observation = observe(resources)
                require(observation['child_pids'] or code is not None or child.poll() is not None,'Live child observable')
                if code is not None:
                    result['child_exit_code'] = code
                    require(not observation['child_pids'],'No surviving owned descendants')
                    result['whole_owned_tree_retired'] = True
                    break
                time.sleep(PERIOD)
        require(result['child_exit_code'] == 0,'Native or score execution-integrity failure')
        rp,cp = child_out/'result.json',child_out/'completion.json'
        result.update(child_result_sha256=sha(rp,time_guard),child_completion_sha256=sha(cp,time_guard))
        if scoring:
            record,end = completed(child_out,result['child_result_sha256'],time_guard)
            require(record['schema'] == SCORE_SCHEMA and record['stage'] == 'score' and
                    record['integrity_passed'] is True,'Scientific failure may be operationally complete')
            require(record['source_bindings'][str(worker)] == args.worker_sha256 and
                    record['source_bindings'][str(HERE/'PROTOCOL.md')] == PROTOCOL_SHA,'Reviewed score source frame')
            for key in ('source_bindings','input_bindings','artifacts'):
                bind(record.get(key,{}),time_guard)
            result['scientific_advancement'] = bool(record['advances_to_independent14'])
        else:
            record,end = native_receipt(child_out,result['child_result_sha256'],
                'null' if args.stage == 'null' else 'terminal',build,args,time_guard)
            require(result['completed_native_bill_count'] == len(ranks),'All exact bill lifecycles observed')
            for item in result['native_bill_progress']:
                require(record['artifacts'][item['bill']] == item['bill_sha256'] == sha(item['bill'],time_guard) and
                        record['artifacts'][item['lifecycle']] == item['lifecycle_sha256'] == sha(item['lifecycle'],time_guard),
                        'All observed progress is sealed immutable native evidence')
        result.update(child_execution_completed=True,passed=True,completed=True)
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
        if child is not None:
            try:
                helper.stop_owned_group(child); result['whole_owned_tree_retired'] = True
            except BaseException:
                result['cleanup_error'] = traceback.format_exc()
    finally:
        try:
            tick = time_guard if result['passed'] else None
            for path in (out/'started.json',out/'stdout.log',out/'stderr.log',out/'resources.jsonl',
                         child_out/'result.json',child_out/'completion.json'):
                if path.is_file():
                    result['artifacts'][str(path)] = sha(path,tick)
            for item in result['native_bill_progress']:
                for key in ('bill','lifecycle'):
                    result['artifacts'][item[key]] = sha(item[key],tick)
            bind(sources,tick); bind(inputs,tick)
            result.update(hashes_unchanged=True,sources_unchanged=True)
            if result['passed']:
                result['final_resource_observation_after_hashing'] = observe(active=False)
                require(not result['final_resource_observation_after_hashing']['child_pids'],'No descendants after sealing')
                require(sha(child_out/'result.json',time_guard) == result['child_result_sha256'] and
                        sha(child_out/'completion.json',time_guard) == result['child_completion_sha256'],'Final child seal unchanged')
        except BaseException:
            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
                          final_sealing_error=traceback.format_exc())
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
        result.update(elapsed_s=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/'result.json',result)
        write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            hashes_unchanged=result['hashes_unchanged'],sources_unchanged=result['sources_unchanged'],
            result_sha256=sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('stage','passed','completed','elapsed_s','completed_native_bill_count',
        'scientific_advancement','error','final_sealing_error')},allow_nan=False),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
