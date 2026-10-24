"""Bounded target-free native capture: three serial checks, then fixed TRAIN160.

Stdlib-only supervisor. Each child is a fresh cold owned process group. The
two phases require separate explicit calls; TRAIN cannot bypass qualification.
"""
import argparse
from datetime import datetime, timezone
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
NEW, P2 = HERE.parent, HERE.parent.parent
OFFPY = P2/'data_gate/.venv/bin/python'
HELPER = NEW/'current_d50_return_quantization_v1/watch.py'
HELPER_SHA = 'f79749448c5a6015540474b18b58b6981944fc8b8c9182200effc0095728e32b'
CAP = NEW/'strategic_reset_20260909_v1/cap_native_objective_v1'
CAP_RESULT_SHA = '92853467117de93914097ac316670b771367dc3e29ea9e947a9d87e9180506bb'
CAP_WATCH_SHA = '9f4c7bb707ccecaf3946f3667f9e28fe68759f6e01e1543b396033b4bc5fa329'
PROTOCOL_SHA = 'febed4bdccc7c031cb846e6d97f695536ca98d3bef10d07de48f1418a91b6a58'
BANK_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
VALID = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
         139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
         204,219,224,227,233,234)
TRAIN = tuple(r for r in range(35,235) if r not in VALID)
BLAS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
PERIOD = .2
INTERVAL_LIMIT = .5
CHILD_LIMIT = 8_000_000_000
TREE_LIMIT = 17_000_000_000
SWAP_LIMIT = 512*1024**2


def require(test, message):
    if not test:
        raise RuntimeError(message)


def sha(path):
    p = Path(path).resolve()
    require('truthpost' not in p.parts and not
            (p.suffix in ('.npz','.npy') and 'truth' in p.name.lower()), 'No target access')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def verify_capture(item, artifacts, deep=True):
    path = Path(item['path']).resolve()
    require(HERE in path.parents and path.name == 'result.json', 'Owned child result')
    require(sha(path) == item['sha256'], 'Child result immutable')
    record, end = read(path), read(path.parent/'completion.json')
    artifacts[str(path)] = item['sha256']
    artifacts[str(path.parent/'completion.json')] = sha(path.parent/'completion.json')
    require(all(record[k] is True for k in ('passed','completed','hashes_unchanged','sources_unchanged'))
            and end['passed'] and end['completed'] and end['result_sha256'] == item['sha256'],
            'Completed qualified child')
    require(record['rank'] == item['rank'] and record['rank'] in TRAIN and
            record['candidate_sha256'] == BANK_SHA and record['truth_accessed'] is False and
            record['parameter_updates'] == record['gradient_calls'] == 0 and
            record['network_calls'] == record['observed_predict_calls'] == 1 and
            record['observed_gnn_calls'] == 16 and record['MPS_used'] is False,
            'One exact target-free native network')
    require(record['all16_domains_present'] and record['all47_selectors_present'] and
            record['served_reconstruction_byte_exact'] and record['own_complete_bill'] and
            not record['candidate_source_edited'] and not record['input_preparation_in_participant_bill'],
            'Native output/domain/accounting ownership')
    replay = record['head_replay']
    require(all(replay[k] for k in ('original_payload_arrays_byte_exact','original_P_unchanged',
        'captured_parameter_bytes_exact','delta_byte_exact','widened_hook_delta_byte_exact',
        'final_raw_byte_exact','final_served_byte_exact')), 'Exact original terminal replay')
    require(set(record['captures']) == {'host','hidden','sigma','delta','raw','served'}, 'Fixed capture fields')
    diag = record['diagnostics']
    require(diag['variance_count'] == 16 and diag['selector_count'] == 47 and
            diag['added_participant_array_operations'] == 0 and
            all(v['finite'] and v['minimum'] >= 1e-10 and v['below_floor_count'] == 0
                for v in diag['variance_domain'].values()), 'All native variance domains')
    bill = record['complete_bill']
    require(bill['cold'] and bill['includes_initialization'] and
            type(bill['flops']) is int and 0 < bill['flops'] <= 2**41 and
            bill['wall_s'] < 120 and 0 < bill['peak_rss_bytes'] < CHILD_LIMIT,
            'Actual complete cold bill/resources')
    for key in ('input_lifecycle','lifecycle'):
        life = record[key]
        require(life['complete'] and life['context_released'] and life['after_context_exit'] and
                life['closed_context_bill_unchanged_by_reset'] and life['candidate_state_reset'] is False,
                'Original post-bill-only context release')
    if item['rank'] in (35,100):
        require(all(record['archived_cold_reference'].values()), 'Original FQ reference identity')
    if deep:
        for name in ('bindings','source_bindings','artifacts'):
            for p, value in record[name].items():
                if isinstance(value, dict):
                    p, digest = value['path'], value['sha256']
                else:
                    digest = value
                require(sha(p) == digest, 'Child authenticated input/artifact changed '+str(p))
        for value in record['captures'].values():
            require(sha(value['path']) == value['sha256'], 'Capture file immutable')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--phase', choices=('qualify','train'))
    for key in ('source','native','common','protocol','qualification'):
        parser.add_argument('--'+key+'-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(args.phase is not None, 'Explicit phase required')
    for key in ('source','native','common','protocol'):
        require(re.fullmatch('[0-9a-f]{64}', getattr(args,key+'_sha256') or '') is not None, 'Reviewed hash '+key)
    require(args.protocol_sha256 == PROTOCOL_SHA and os.getpriority(os.PRIO_PROCESS,0) >= 10 and
            all(os.environ.get(k) == '1' for k in BLAS) and
            os.environ.get('PYTHONDONTWRITEBYTECODE') == '1' and
            os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0', 'Protocol/nice10/BLAS1')
    pins = {str(Path(__file__).resolve()):args.source_sha256,
        str(HERE/'native.py'):args.native_sha256, str(HERE/'common.py'):args.common_sha256,
        str(HERE/'PROTOCOL.md'):PROTOCOL_SHA, str(HELPER):HELPER_SHA,
        str(CAP/'analysis_v2/result.json'):CAP_RESULT_SHA, str(CAP/'watch_v2/result.json'):CAP_WATCH_SHA}
    for p, digest in pins.items():
        require(sha(p) == digest, 'Reviewed fixed source changed '+p)
    cap, capwatch = read(CAP/'analysis_v2/result.json'), read(CAP/'watch_v2/result.json')
    require(cap['passed'] and cap['completed'] and cap['hashes_unchanged'] and
            not cap['advances_to_independent14'] and capwatch['passed'] and capwatch['completed'],
            'Completed non-advancing native CAP objective')
    for folder, digest in ((CAP/'analysis_v2',CAP_RESULT_SHA),(CAP/'watch_v2',CAP_WATCH_SHA)):
        end = read(folder/'completion.json')
        require(end['passed'] and end['completed'] and end['result_sha256'] == digest, 'CAP seals')
        pins[str(folder/'completion.json')] = sha(folder/'completion.json')
    require(not any(k in sys.modules for k in ('numpy','torch','flopscope')), 'Stdlib-only watcher')
    spec = importlib.util.spec_from_file_location('terminal_capture_resource_helpers',HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    records, artifacts = [], {}
    qualification = None
    if args.phase == 'qualify':
        require(args.qualification_sha256 is None, 'No qualification input to initial phase')
        out = HERE/'qualify_v1'
        jobs = [(35,'native_0035'),(100,'native_0100'),(35,'native_0035_repeat')]
        capacity, wall_limit = 1, 450.0
    else:
        require(re.fullmatch('[0-9a-f]{64}',args.qualification_sha256 or '') is not None, 'Reviewed qualification digest')
        qp = HERE/'qualify_v1/result.json'
        require(sha(qp) == args.qualification_sha256, 'Exact completed qualification')
        q, qe = read(qp), read(qp.parent/'completion.json')
        require(q['passed'] and q['completed'] and q['hashes_unchanged'] and
                qe['passed'] and qe['completed'] and qe['result_sha256'] == args.qualification_sha256 and
                q['phase'] == 'qualify' and q['repeat_byte_exact'] and
                [r['rank'] for r in q['captures']] == [35,100,35], 'Initial three-call qualification')
        for p,digest in q['source_bindings'].items():
            require(pins.get(p) == digest and sha(p) == digest, 'Unchanged qualified worker/observer source')
        for item in q['captures']:
            verify_capture(item,artifacts)
        records = q['captures'][:2].copy()
        qualification = dict(path=str(qp),sha256=args.qualification_sha256)
        pins[str(qp)] = args.qualification_sha256
        pins[str(qp.parent/'completion.json')] = sha(qp.parent/'completion.json')
        out = HERE/'capture_train_v1'
        jobs = [(rank,f'native_{rank:04d}') for rank in TRAIN if rank not in (35,100)]
        capacity, wall_limit = 2, 7200.0
    require(not out.exists(), 'Exclusive phase; no implicit restart')
    out.mkdir()
    result = dict(schema='current_d50_terminal_linear_capture_watch_v1', phase=args.phase,
        passed=False,completed=False,hashes_unchanged=False,target_accessed=False,
        source_bindings=pins,artifacts=artifacts,captures=records,qualification=qualification,
        candidate_sha256=BANK_SHA,capacity=capacity,requested_seals=vars(args),
        maximum_sampled_rss_bytes=0,maximum_child_rss_bytes=0,maximum_sample_interval_s=0.0,
        maximum_swap_growth_bytes=0.0,samples=0,wall_limit_s=wall_limit,
        started_utc=datetime.now(timezone.utc).isoformat(),
        limitations=['RSS/swap are sampled, not continuous peaks.','No production or score qualification.'])
    started = time.monotonic()
    active, all_children, logs, log_paths, handlers = {}, [], [], [], {}
    before = None
    try:
        def interrupted(signum, frame):
            raise RuntimeError('Capture supervisor interrupted '+str(signum))
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig] = signal.signal(sig,interrupted)
        before = helper.swap_state()
        result['before_swap'] = before
        write(out/'started.json',result)
        pending = list(jobs)
        last = time.monotonic()
        with (out/'resources.jsonl').open('x') as resources, (out/'progress.jsonl').open('x') as progress:
            while pending or active:
                require(time.monotonic()-started < wall_limit, 'Whole phase wall limit')
                while pending and len(active) < capacity:
                    rank, label = pending.pop(0)
                    folder = out/label
                    require(not folder.exists(), 'No reused child output')
                    stdout = (out/(label+'.stdout.log')).open('xb')
                    stderr = (out/(label+'.stderr.log')).open('xb')
                    logs.extend((stdout,stderr))
                    log_paths.extend((out/(label+'.stdout.log'),out/(label+'.stderr.log')))
                    command = [str(OFFPY),'-B','-u',str(HERE/'native.py'),'--run','--rank',str(rank),
                        '--output',str(folder),'--source-sha256',args.native_sha256,
                        '--common-sha256',args.common_sha256,'--protocol-sha256',PROTOCOL_SHA]
                    child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                        env=os.environ.copy(),start_new_session=True)
                    active[child.pid] = dict(child=child,rank=rank,folder=folder,start=time.monotonic(),
                                            stdout=stdout,stderr=stderr)
                    all_children.append(child)
                statuses = {pid:job['child'].poll() for pid,job in active.items()}
                table = helper.process_table()
                combined = helper.descendants({os.getpid()},table)
                child_sets = {}
                for pid in active:
                    owned = helper.descendants({pid},table) | {p for p,pp,g,rss in table if g == pid}
                    child_sets[pid] = helper.descendants(owned,table)
                    combined |= child_sets[pid]
                tree_rss = sum(rss*1024 for pid,pp,g,rss in table if pid in combined)
                by_child = {pid:sum(rss*1024 for p,pp,g,rss in table if p in owned)
                            for pid,owned in child_sets.items()}
                swap = helper.swap_state()
                now = time.monotonic()
                interval, last = now-last, now
                growth = swap['used_bytes']-before['used_bytes']
                result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],tree_rss)
                result['maximum_child_rss_bytes'] = max(result['maximum_child_rss_bytes'],*by_child.values())
                result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
                result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
                result['samples'] += 1
                resources.write(json.dumps(dict(elapsed_s=now-started,interval_s=interval,
                    combined_rss_bytes=tree_rss,child_rss_bytes=by_child,swap=swap))+'\n')
                resources.flush()
                require(interval <= INTERVAL_LIMIT and tree_rss < TREE_LIMIT and
                        all(rss < CHILD_LIMIT for rss in by_child.values()) and growth <= SWAP_LIMIT,
                        'Sampled interval/RSS/swap safety limit')
                for pid,job in list(active.items()):
                    code = statuses[pid]
                    require(now-job['start'] < 120, 'Independent whole-child wall limit')
                    if code is not None:
                        require(code == 0, 'Native worker failed; see '+str(job['folder']))
                        require(not any(p in child_sets[pid] for p,pp,g,rss in table), 'No surviving child descendants')
                        rp = job['folder']/'result.json'
                        item = dict(rank=job['rank'],path=str(rp),sha256=sha(rp))
                        verify_capture(item,artifacts,deep=False)
                        records.append(item)
                        job['stdout'].close(); job['stderr'].close()
                        del active[pid]
                        update = dict(rank=job['rank'],completed_new=len(records)-(2 if args.phase=='train' else 0),
                            remaining=len(pending)+len(active),elapsed_s=time.monotonic()-started)
                        progress.write(json.dumps(update)+'\n');progress.flush()
                        print(json.dumps(update),flush=True)
                if pending or active:
                    time.sleep(PERIOD)
        require(not active, 'All owned child groups retired')
        checked = [verify_capture(item,artifacts) for item in records]
        if args.phase == 'qualify':
            require([r['rank'] for r in checked] == [35,100,35], 'Serial fixed qualification order')
            for name in ('host','hidden','sigma','delta','raw','served'):
                require(checked[0]['captures'][name]['raw_sha256'] == checked[2]['captures'][name]['raw_sha256'],
                        'Fresh-cold35 repeat exact '+name)
            require(checked[0]['complete_bill']['flops'] == checked[2]['complete_bill']['flops'] and
                    checked[0]['diagnostics'] == checked[2]['diagnostics'], 'Exact repeat bill and domains/selectors')
            result['repeat_byte_exact'] = True
        else:
            require(len(checked) == 160 and {r['rank'] for r in checked} == set(TRAIN), 'Every TRAIN primary once')
            manifest = dict(schema='current_d50_terminal_linear_train_capture_v1',completed=True,
                target_accessed=False,train_ranks=list(TRAIN),candidate_sha256=BANK_SHA,
                captures={str(item['rank']):dict(path=item['path'],sha256=item['sha256']) for item in records},
                source_bindings=pins,qualification=qualification,
                watch_result_path=str(out/'result.json'),watch_completion_path=str(out/'completion.json'))
            write(out/'manifest.json',manifest)
            artifacts[str(out/'manifest.json')] = sha(out/'manifest.json')
        for p,digest in pins.items():
            require(sha(p) == digest, 'Reviewed source changed during capture '+p)
        for path in (out/'started.json',out/'resources.jsonl',out/'progress.jsonl',*log_paths):
            artifacts[str(path)] = sha(path)
        # All numerical children have exited. Observe the watcher and its own
        # short-lived ps helper after final metadata verification and hashing.
        final_memory = helper.memory_snapshot(-1)
        require(not final_memory['child_pids'] and final_memory['combined_rss_bytes'] < TREE_LIMIT,
                'Final watcher RSS and retired owned numerical children')
        result['final_resource_observation'] = final_memory
        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],
                                                 final_memory['combined_rss_bytes'])
        after = helper.swap_state()
        result['after_swap'] = after
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],after['used_bytes']-before['used_bytes'])
        require(result['maximum_swap_growth_bytes'] <= SWAP_LIMIT and time.monotonic()-started < wall_limit,
                'Final phase wall/swap guard')
        result.update(passed=True,completed=True,hashes_unchanged=True)
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,error=traceback.format_exc())
        for job in active.values():
            try:
                helper.stop_owned_group(job['child'])
            except BaseException:
                result.setdefault('cleanup_errors',[]).append(traceback.format_exc())
    finally:
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
        for f in logs:
            f.close()
        result.update(elapsed_s=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write(out/'result.json',result)
        write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            result_sha256=sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('phase','passed','completed','elapsed_s',
        'maximum_sampled_rss_bytes','maximum_swap_growth_bytes','error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
