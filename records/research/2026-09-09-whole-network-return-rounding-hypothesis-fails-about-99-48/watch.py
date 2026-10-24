"""One-shot stdlib supervisor, fixed twenty forwards then an artifact judge."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback

import common as c

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent/'current_d50_return_quantization_v1/watch.py'
HELPER_SHA = 'f79749448c5a6015540474b18b58b6981944fc8b8c9182200effc0095728e32b'
AP = Path('/Users/matt/anaconda3/bin/python')
OFFPY = HERE.parent.parent/'data_gate/.venv/bin/python'
WALL, RSS, SWAP, PERIOD = 900., 8_000_000_000, 512*1024**2, .2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--manifest')
    parser.add_argument('--manifest-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    c.require(Path(args.manifest or '').resolve() == HERE/'launch_manifest_v1.json'
              and re.fullmatch('[0-9a-f]{64}',args.manifest_sha256 or '') is not None,
              'Exact externally reviewed manifest')
    c.require(c.sha(args.manifest) == args.manifest_sha256, 'Manifest changed')
    manifest = json.loads(Path(args.manifest).read_text())
    pins = manifest['source_bindings']
    expected = {str(HERE/name) for name in ('common.py','cpu.py','native.py','judge.py','watch.py','PROTOCOL.md')} | {str(HELPER)}
    c.require(set(pins) == expected and pins[str(HELPER)] == HELPER_SHA, 'Complete fixed source inventory')
    for path,digest in pins.items():
        c.require(c.sha(path) == digest, 'Reviewed source changed: ' + path)
    roster = [dict(role=role,case=case,rank=rank,folder=str(HERE/f'{role}_{case}_r{rank:04d}'))
              for role in ('native','cpu') for case in c.CASES for rank in c.RANKS]
    c.require(manifest['roster'] == roster and manifest['schema'] == 'full_return_launch_manifest_v1', 'Fixed exact twenty-cell roster')
    c.require(os.getpriority(os.PRIO_PROCESS,0) >= 10
              and all(os.environ.get(k) == '1' for k in c.BLAS)
              and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
              and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0') == '0', 'nice10/BLAS1/no bytecode/no MPS')
    c.require(all(not Path(x['folder']).exists() for x in roster)
              and not (HERE/'judgment_v1').exists(), 'One-shot outputs already exist')
    helper = c.module(HELPER,'full_return_saved_watch_helpers')
    out = HERE/'watch_v1'
    out.mkdir()
    result = dict(schema='full_return_watch_v1',passed=False,completed=False,sources_unchanged=False,
        hashes_unchanged=False,scientific_advancement=False,source_bindings=pins,
        manifest=dict(path=args.manifest,sha256=args.manifest_sha256),roster=roster,children=[],artifacts={},
        wall_limit_s=WALL,combined_rss_limit_bytes=RSS,swap_growth_limit_bytes=SWAP,
        sample_interval_limit_s=1.,requested_sample_period_s=PERIOD,
        maximum_sampled_rss_bytes=0,maximum_sample_interval_s=0.,maximum_swap_growth_bytes=0.,samples=0,
        network_calls_requested=20,numerical_children_concurrent_limit=1,
        started_utc=datetime.now(timezone.utc).isoformat(),
        limitations=['RSS/swap are sampled; no continuous peak guarantee.',
                     'Swap growth is system-wide. PASS is acquisition, never score advancement.'])
    started = time.monotonic()
    child = None
    prior = {}

    def interrupted(signum,frame):
        raise RuntimeError('Supervisor interrupted by signal ' + str(signum))

    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            prior[sig] = signal.signal(sig,interrupted)
        before = helper.swap_state()
        result['before_swap'] = before
        c.write(out/'started.json',result)
        jobs = roster + [dict(role='judge',case='null',rank=35,folder=str(HERE/'judgment_v1'))]
        with (out/'resources.jsonl').open('x') as resources:
            for index,job in enumerate(jobs):
                c.require(time.monotonic()-started < WALL, 'Whole900s cap before child')
                source = HERE/(job['role']+'.py')
                for path,digest in pins.items():
                    c.require(c.sha(path) == digest, 'Source changed before child: ' + path)
                command = [str(OFFPY if job['role'] == 'native' else AP),'-B','-u',str(source),'--run',
                    '--case',job['case'],'--rank',str(job['rank']),
                    '--source-sha256',pins[str(source)],'--common-sha256',pins[str(HERE/'common.py')],
                    '--protocol-sha256',pins[str(HERE/'PROTOCOL.md')],
                    '--manifest',args.manifest,'--manifest-sha256',args.manifest_sha256]
                entry = dict(index=index,**job,command=command,passed=False)
                result['children'].append(entry)
                print(json.dumps(dict(start=index,role=job['role'],case=job['case'],rank=job['rank'])),flush=True)
                last = child_started = time.monotonic()
                log_out,log_err = out/f'child_{index:02d}.stdout.log',out/f'child_{index:02d}.stderr.log'
                with log_out.open('xb') as stdout,log_err.open('xb') as stderr:
                    child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                        env=os.environ.copy(),start_new_session=True)
                    entry.update(pid=child.pid,process_group=child.pid)
                    while True:
                        c.require(time.monotonic()-started < WALL, 'Whole900s cap')
                        code = child.poll()
                        item = helper.memory_snapshot(child.pid)
                        sampled = time.monotonic()
                        interval,last = sampled-last,sampled
                        swap = helper.swap_state()
                        now = time.monotonic()
                        growth = swap['used_bytes']-before['used_bytes']
                        item.update(index=index,elapsed_s=now-started,child_elapsed_s=now-child_started,
                            interval_s=interval,swap=swap,swap_growth_bytes=growth)
                        resources.write(json.dumps(item,allow_nan=False)+'\n')
                        resources.flush()
                        result['samples'] += 1
                        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],item['combined_rss_bytes'])
                        result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],interval)
                        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
                        c.require(now-started < WALL and now-child_started < 120, 'Whole/per-child wall cap')
                        c.require(interval <= 1., 'Resource sample interval exceeds1s')
                        c.require(item['combined_rss_bytes'] < RSS, 'Combined8GB cap')
                        c.require(growth <= SWAP, 'System swap-growth cap')
                        c.require(item['child_pids'] or code is not None or child.poll() is not None,
                                  'Live child not observable')
                        if code is not None:
                            entry.update(exit_code=code,elapsed_s=now-child_started)
                            c.require(not item['child_pids'], 'Reaped child left descendants')
                            break
                        time.sleep(PERIOD)
                c.require(entry['exit_code'] == 0, 'Child failed; preserve outputs, no retry')
                rp,cp = Path(job['folder'])/'result.json',Path(job['folder'])/'completion.json'
                rec,seal = json.loads(rp.read_text()),json.loads(cp.read_text())
                rsha,csha = c.sha(rp),c.sha(cp)
                c.require(rec['passed'] and rec['completed'] and rec['hashes_unchanged'] and rec['sources_unchanged']
                          and seal['passed'] and seal['completed'] and seal['result_sha256'] == rsha,
                          'Complete sealed child')
                c.require(rec['role'] == job['role'] and rec['case'] == job['case'] and rec['rank'] == job['rank']
                          and rec['source_bindings'][str(source)] == pins[str(source)]
                          and rec['source_bindings'][str(HERE/'common.py')] == pins[str(HERE/'common.py')]
                          and rec['source_bindings'][str(HERE/'PROTOCOL.md')] == pins[str(HERE/'PROTOCOL.md')],
                          'Actual exact reviewed child')
                for path in (rp,cp,log_out,log_err):
                    result['artifacts'][str(path)] = c.sha(path)
                entry.update(passed=True,result_sha256=rsha,completion_sha256=csha)
                print(json.dumps(dict(done=index,role=job['role'],case=job['case'],rank=job['rank'],
                                      elapsed_s=entry['elapsed_s'])),flush=True)
                child = None
        c.require(len(result['children']) == 21 and all(x['passed'] for x in result['children']), 'Complete fixed roster and judge')
        result['after_swap'] = helper.swap_state()
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],result['after_swap']['used_bytes']-before['used_bytes'])
        c.require(result['maximum_swap_growth_bytes'] <= SWAP and time.monotonic()-started < WALL, 'Final resource limits')
        result.update(passed=True,completed=True)
    except BaseException:
        result.update(passed=False,completed=False,error=traceback.format_exc())
        if child is not None:
            try:
                helper.stop_owned_group(child)
                result['owned_group_cleanup_complete'] = True
                result['children'][-1]['exit_code'] = child.poll()
            except BaseException:
                result['cleanup_error'] = traceback.format_exc()
    finally:
        for sig,handler in prior.items():
            signal.signal(sig,handler)
        try:
            for path,digest in pins.items():
                c.require(c.sha(path) == digest, 'Final source changed: ' + path)
            c.require(c.sha(args.manifest) == args.manifest_sha256, 'Final manifest changed')
            for path,digest in list(result['artifacts'].items()):
                c.require(c.sha(path) == digest, 'Final child artifact changed: ' + path)
            for path in sorted(out.iterdir()):
                if path.is_file():
                    result['artifacts'][str(path)] = c.sha(path)
            result.update(hashes_unchanged=True,sources_unchanged=True)
            c.require(time.monotonic()-started < WALL or not result['passed'], 'Final sealing wall cap')
        except BaseException:
            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
                          sealing_error=traceback.format_exc())
        result.update(elapsed_s=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        c.write(out/'result.json',result)
        c.write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=c.sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s','maximum_sampled_rss_bytes',
          'maximum_sample_interval_s','maximum_swap_growth_bytes','error','cleanup_error')},sort_keys=True),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
