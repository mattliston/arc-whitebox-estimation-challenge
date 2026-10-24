"""One-shot target-free response census supervisor; inert without --run.

PASS means the two unmodified calls, provenance and sampled resource gates
completed. It is not a price verdict, candidate qualification or score.
Only the fresh child process group may be signalled; no automatic next step.
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
OFFPY = P2/'data_gate/.venv/bin/python'
HELPER = NEW/'current_d50_terminal_linear_v2/watch_fit.py'
HELPER_SHA = 'b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd'
PROTOCOL_SHA = '57dd4ccedd93c9626a52f0864693a298756bc5e2a06ee2537de99a0ebc4c2dc8'
BANK = NEW/'birth_direction_fit_v1/build_terminal_v1/estimator.py'
BANK_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
CONTROL = NEW/'birth_direction_fit_v1/panel_v2/terminal'
CONTROL_SHA = '6ad5687a0baf3265e2c663b2d0ea453262f57fa6a84474fd2fbac9ad1211aade'
CONTROL_END_SHA = '58b51092efefb534a3c8ac81d4cac741abd8cbb9344db647e94d0b6d7f7bf35b'
SCHEMA = 'current_d50_response10_census_watch_v1'
CHILD_SCHEMA = 'current_d50_response10_census_native_v1'
INITIALIZATION, BUDGET = 1_383_913_786, 2**41
WALL_LIMIT, RSS_LIMIT, SWAP_LIMIT = 300.0, 8_000_000_000, 512*1024**2
PERIOD, MAX_INTERVAL, PROGRESS_LIMIT = .2, 1.0, 120.0
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    path = Path(path).resolve()
    require('truthpost' not in path.parts and not
            (path.suffix in ('.npy','.npz') and 'truth' in path.name.lower()),
            'Supervisor never opens target arrays')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def bind(mapping):
    require(isinstance(mapping,dict),'Flat source/input/artifact bindings')
    for path,digest in mapping.items():
        require(isinstance(path,str) and Path(path).is_absolute() and
                isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) and
                sha(path) == digest,'Immutable bound file: '+str(path))


def lifecycle(bill, bill_path, life_path):
    require(all(p.is_file() and not p.is_symlink() for p in (bill_path,life_path)),
            'Regular completed bill and lifecycle files')
    life = read(life_path)
    require(bill['lifecycle_receipt'] == str(life_path) and
            life['saved_bill'] == str(bill_path) and life['saved_bill_sha256'] == sha(bill_path) and
            life['call'] == bill['call'] and life['rank'] == bill['rank'] and
            life['reset_api'] == 'flopscope.budget_reset' and
            all(life.get(k) is True for k in ('complete','after_context_exit','context_released',
                'closed_context_bill_unchanged_by_reset')) and life['candidate_state_reset'] is False,
            'Exact saved bill followed by public reset and released context')
    for left,right in (('flops','flops'),('operation_count','operation_count'),
        ('context_wall_s','context_wall_s'),('backend_s','flopscope_backend_s'),
        ('overhead_s','flopscope_overhead_s'),('residual_s','native_residual_s')):
        value = life['closed_metrics'][left]
        require(value == bill[right] and math.isfinite(value) and value >= 0,
                'Closed metrics preserved across reset: '+left)
    return life


def validate_child(record, end, digest, args, folder, control):
    require(record['schema'] == CHILD_SCHEMA and all(record.get(k) is True for k in
        ('passed','completed','hashes_unchanged','sources_unchanged','own_output_and_bill_parity',
         'all16_domains_qualified','all47_selectors_qualified')) and record['truth_accessed'] is False
        and end['passed'] is True and end['completed'] is True and end['result_sha256'] == digest,
        'Complete target-free census and immutable completion')
    require(record['candidate_sha256'] == BANK_SHA and record['initialization_flops'] == INITIALIZATION
            and len(record['trials']) == len(record['input_preparation']) == 2,
            'Only cold35 and warm35; actual initialization once')
    require(record['source_bindings'][str(HERE/'capture.py')] == args.worker_sha256 and
            record['source_bindings'][str(HERE/'PROTOCOL.md')] == PROTOCOL_SHA and
            record['source_bindings'][str(BANK)] == BANK_SHA,'Exact reviewed sources')
    for group in ('source_bindings','input_bindings','artifacts'):
        if group in record:
            bind(record[group])
    for call,trial in enumerate(record['trials']):
        prior = control['trials'][0 if call == 0 else 14]
        require(trial['call'] == call and trial['rank'] == 35 and
                trial['includes_initialization'] is (call == 0) and
                type(trial['flops']) is int and 0 < trial['flops'] <= BUDGET and
                trial['flops'] == prior['flops'] and
                trial['output_bytes_sha256'] == prior['output_bytes_sha256'],
                'Own full FLOPs/output identity; no unrelated-history operation-count comparison')
        require(all(math.isfinite(trial[k]) and trial[k] >= 0 for k in
                    ('wall_s','context_wall_s','flopscope_backend_s','flopscope_overhead_s','native_residual_s'))
                and trial['wall_s'] < 120 and trial['context_wall_s'] < 120 and
                0 < trial['peak_rss_bytes'] < RSS_LIMIT,'Original per-call resource gates')
        bp,lp = folder/f'call_{call:02d}_bill.json',folder/f'call_{call:02d}_lifecycle.json'
        require(read(bp) == trial and record['artifacts'][str(bp)] == sha(bp) and
                record['artifacts'][str(lp)] == sha(lp),'Actual own bill/lifecycle artifacts')
        lifecycle(trial,bp,lp)
    first,last = record['trials']
    require(first['flops']-last['flops'] == INITIALIZATION and
            first['output_bytes_sha256'] == last['output_bytes_sha256'],'Exact repeat and measured init')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    for name in ('source','worker','protocol'):
        parser.add_argument('--'+name+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,k+'_sha256') or '')
                for k in ('source','worker','protocol')),'Explicit reviewed source digests')
    require(args.protocol_sha256 == PROTOCOL_SHA,'Sole fixed response.1 census protocol')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')),'Stdlib-only watcher')
    require(OFFPY.is_file() and sys.flags.optimize == 0 and os.getpriority(os.PRIO_PROCESS,0) >= 10 and
            all(os.environ.get(k) == '1' for k in BLAS) and
            os.environ.get('PYTHONDONTWRITEBYTECODE') == '1' and
            os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0','OFFPY/nice10/BLAS1/no fallback')
    worker,child_out,out = HERE/'capture.py',HERE/'native',HERE/'native_watch'
    sources = {str(Path(__file__).resolve()):args.source_sha256,str(worker):args.worker_sha256,
        str(HERE/'PROTOCOL.md'):PROTOCOL_SHA,str(HELPER):HELPER_SHA,str(BANK):BANK_SHA}
    inputs = {str(CONTROL/'result.json'):CONTROL_SHA,str(CONTROL/'completion.json'):CONTROL_END_SHA}
    bind(sources); bind(inputs)
    control,control_end = read(CONTROL/'result.json'),read(CONTROL/'completion.json')
    require(control['passed'] and control['completed'] and control['hashes_unchanged'] and
            control['candidate_sha256'] == BANK_SHA and control['initialization_flops'] == INITIALIZATION and
            control_end['passed'] and control_end['completed'] and control_end['result_sha256'] == CONTROL_SHA,
            'Preserved completed d50 independent14 reference metadata')
    require(not HERE.is_symlink() and not child_out.exists() and not child_out.is_symlink() and
            not out.exists() and not out.is_symlink(),'Exclusive fixed output directories; no retry/resume')
    spec = importlib.util.spec_from_file_location('response_census_resource_helpers',HELPER)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    out.mkdir(exist_ok=False)
    command = [str(OFFPY),'-B','-u',str(worker),'--run',
        '--source-sha256',args.worker_sha256,'--protocol-sha256',PROTOCOL_SHA]
    result = dict(schema=SCHEMA,passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
        source_bindings=sources,input_bindings=inputs,artifacts={},command=command,
        candidate_sha256=BANK_SHA,control_result_sha256=CONTROL_SHA,truth_arrays_opened=False,
        numerical_imports=False,operational_verdict_only=True,price_measured=False,scientific_advancement=False,
        child_execution_completed=False,whole_owned_tree_retired=False,completed_bill_count=0,progress=[],
        first_progress_elapsed_s=None,last_progress_elapsed_s=None,progress_requires_closed_lifecycle=True,
        wall_limit_s=WALL_LIMIT,combined_rss_limit_bytes=RSS_LIMIT,swap_growth_limit_bytes=SWAP_LIMIT,
        no_completed_bill_limit_s=PROGRESS_LIMIT,requested_sample_period_s=PERIOD,sample_interval_limit_s=MAX_INTERVAL,
        maximum_sample_interval_s=0.0,maximum_sampled_rss_bytes=0,maximum_swap_growth_bytes=0.0,samples=0,
        limitations=['RSS/swap are sampled, not continuous peak guarantees.',
            'Swap growth is system-wide, not attributed to this child.',
            'PASS is acquisition integrity, not hypothetical savings or candidate raw/adjusted qualification.'],
        started_utc=datetime.now(timezone.utc).isoformat())
    child,before,handlers = None,None,{}
    last = last_progress = began

    def interrupted(signum, frame):
        raise RuntimeError('Census watcher interrupted: '+str(signum))

    def progress():
        nonlocal last_progress
        found = []
        for path in child_out.glob('call_*_bill.json'):
            if path.name.endswith('_input_bill.json'):
                continue
            match = re.fullmatch(r'call_([0-9]{2})_bill\.json',path.name)
            require(match is not None and not path.is_symlink(),'Only fixed regular prediction bill names')
            found.append(int(match[1]))
        require(sorted(found) == list(range(len(found))) and len(found) <= 2,'Ordered call00/01 only')
        for call in range(result['completed_bill_count'],len(found)):
            bp,lp = child_out/f'call_{call:02d}_bill.json',child_out/f'call_{call:02d}_lifecycle.json'
            if not lp.exists():
                break
            try:
                bill = read(bp); lifecycle(bill,bp,lp)
            except json.JSONDecodeError:
                break  # The one writer has not closed both receipts yet.
            require(bill['call'] == call and bill['rank'] == 35 and
                    bill['includes_initialization'] is (call == 0) and type(bill['flops']) is int and
                    0 < bill['flops'] <= BUDGET,'Next actual cold/warm prediction bill')
            now = time.monotonic()
            require(now-last_progress < PROGRESS_LIMIT,'Completed lifecycle first observed beyond120s deadline')
            last_progress = now
            result['completed_bill_count'] = call+1
            result['first_progress_elapsed_s'] = result['first_progress_elapsed_s'] if call else now-began
            result['last_progress_elapsed_s'] = now-began
            result['progress'].append(dict(call=call,rank=35,path=str(bp),sha256=sha(bp),
                lifecycle_path=str(lp),lifecycle_sha256=sha(lp),observed_elapsed_s=now-began))

    def observe(stream=None, active=True):
        nonlocal last
        memory = helper.memory_snapshot(child.pid if child is not None else -1)
        sampled = time.monotonic(); interval,last = sampled-last,sampled
        swap = helper.swap_state(); growth = swap['used_bytes']-before['used_bytes']
        progress(); now = time.monotonic()
        item = dict(elapsed_s=now-began,interval_s=interval,**memory,swap=swap,swap_growth_bytes=growth,
            completed_bill_count=result['completed_bill_count'],no_completed_bill_s=now-last_progress,
            live_child_interval_gate=active)
        result['samples'] += 1
        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],memory['combined_rss_bytes'])
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
        if active:
            result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
        if stream is not None:
            stream.write(json.dumps(item,allow_nan=False)+'\n'); stream.flush()
        require(now-began < WALL_LIMIT and memory['combined_rss_bytes'] < RSS_LIMIT and growth <= SWAP_LIMIT,
                'Independent whole-phase wall/combined-RSS/swap limits')
        require(not active or interval <= MAX_INTERVAL,'Live-child sampling interval<=1s')
        require(not active or now-last_progress < PROGRESS_LIMIT,'120s without next bill AND released lifecycle')
        return item

    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig] = signal.signal(sig,interrupted)
        before = helper.swap_state(); result['before_swap'] = before
        write(out/'started.json',result)
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr,\
                (out/'resources.jsonl').open('x') as resources:
            observe(resources,active=False); last = time.monotonic()
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                     env=os.environ.copy(),start_new_session=True)
            result['child_pid'] = result['child_process_group'] = child.pid
            while True:
                code = child.poll(); item = observe(resources)
                require(item['child_pids'] or code is not None or child.poll() is not None,'Live child tree visible')
                if code is not None:
                    result['child_exit_code'] = code
                    require(not item['child_pids'],'No surviving owned descendants')
                    break
                time.sleep(PERIOD)
        require(result['child_exit_code'] == 0,'Census acquisition failed')
        rp,cp = child_out/'result.json',child_out/'completion.json'
        record,end = read(rp),read(cp)
        result.update(child_result_sha256=sha(rp),child_completion_sha256=sha(cp))
        validate_child(record,end,result['child_result_sha256'],args,child_out,control)
        require(result['completed_bill_count'] == 2,'Both completed lifecycles observed in order')
        for item,trial in zip(result['progress'],record['trials']):
            require(read(item['path']) == trial and sha(item['path']) == item['sha256'] and
                    sha(item['lifecycle_path']) == item['lifecycle_sha256'],'Progress remains exactly sealed')
        result.update(passed=True,completed=True,child_execution_completed=True,whole_owned_tree_retired=True)
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
                      error=traceback.format_exc())
        if child is not None:
            try:
                helper.stop_owned_group(child)
                child.wait(timeout=2)
                require(not helper.memory_snapshot(child.pid)['child_pids'],'No descendants after cleanup')
                result.update(whole_owned_tree_retired=True,child_exit_code=child.returncode)
            except BaseException:
                result['cleanup_error'] = traceback.format_exc()
    finally:
        try:
            for path in (out/'started.json',out/'stdout.log',out/'stderr.log',out/'resources.jsonl',
                         child_out/'result.json',child_out/'completion.json'):
                if path.is_file():
                    result['artifacts'][str(path)] = sha(path)
            bind(sources); bind(inputs)
            result.update(hashes_unchanged=True,sources_unchanged=True)
            if result['passed']:
                item = observe(active=False)
                require(not item['child_pids'],'No owned descendants after final sealing')
                require(sha(child_out/'result.json') == result['child_result_sha256'] and
                        sha(child_out/'completion.json') == result['child_completion_sha256'],
                        'Child result/completion immutable through final verification')
                result['final_resource_observation'] = item
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
    print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s','completed_bill_count',
        'maximum_sampled_rss_bytes','maximum_sample_interval_s','error','cleanup_error','final_sealing_error')},
        allow_nan=False),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
