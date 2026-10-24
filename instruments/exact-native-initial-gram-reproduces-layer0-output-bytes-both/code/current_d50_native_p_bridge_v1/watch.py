"""Independent one-native-case supervisor; no numerical imports or retries."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OLD_WATCH = HERE.parent/'current_bd_p_coadaptation_v1/watch.py'
OLD_SHA = 'b51590c06abf4394ed78274756e99046de9bccc2240dd55e8543f64e83e7304b'
OFFPY = HERE.parent.parent/'data_gate/.venv/bin/python'
CASES = ('null','plus025','minus025','plus050','minus050')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,choices=CASES)
    parser.add_argument('--watch-source-sha256',required=True)
    parser.add_argument('--child-source-sha256',required=True)
    parser.add_argument('--build-sha256',required=True)
    parser.add_argument('--hybrid-sha256',required=True)
    parser.add_argument('--protocol-sha256',required=True)
    args = parser.parse_args()
    if os.getpriority(os.PRIO_PROCESS,0)<10 or os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':
        raise RuntimeError('nice10/no-bytecode environment required')
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')):
        raise RuntimeError('Explicit BLAS1 required')
    pins = {str(Path(__file__).resolve()):args.watch_source_sha256,
            str(HERE/'native.py'):args.child_source_sha256,
            str(HERE/'PROTOCOL.md'):args.protocol_sha256,
            str(HERE/'build_probes_v1/build.json'):args.build_sha256,
            str(HERE.parent/'current_d50_hybrid_cp_qualification_v1/stage_mps_gradient_v1/result.json'):args.hybrid_sha256,
            str(OLD_WATCH):OLD_SHA}
    for path,digest in pins.items():
        if sha(path)!=digest:
            raise RuntimeError('Reviewed supervisor input changed: '+path)
    spec = importlib.util.spec_from_file_location('native_bridge_owned_process_utils',OLD_WATCH)
    helpers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helpers
    spec.loader.exec_module(helpers)
    out = HERE/('watch_'+args.case+'_v1')
    child_out = HERE/('run_'+args.case+'_v1')
    if child_out.exists() or (HERE/('admission_'+args.case+'_v1.json')).exists():
        raise RuntimeError('One-shot native case already exists')
    out.mkdir(exist_ok=False)
    command = [str(OFFPY),'-u',str(HERE/'native.py'),'--run','--case',args.case,
               '--source-sha256',args.child_source_sha256,'--build-sha256',args.build_sha256,
               '--hybrid-sha256',args.hybrid_sha256,'--protocol-sha256',args.protocol_sha256]
    before = helpers.swap_state()
    result = dict(schema='current_d50_native_p_bridge_watch_v1',case=args.case,passed=False,
                  completed=False,terminated=False,source_bindings=pins,command=command,
                  wall_limit_s=900,rss_limit_bytes=8_000_000_000,swap_growth_limit_bytes=2*1024**3,
                  maximum_sampled_rss_bytes=0,maximum_swap_growth_bytes=0.,samples=0,
                  requested_sample_period_s=.25,maximum_sample_interval_s=0.,before_swap=before,
                  started_utc=datetime.now(timezone.utc).isoformat())
    helpers.write(out/'started.json',result)
    samples = (out/'resources.jsonl').open('x')
    child = None
    start = last = last_swap = time.monotonic()
    try:
        child = subprocess.Popen(command,stdin=subprocess.DEVNULL,start_new_session=True)
        result['child_pid'] = result['child_process_group'] = child.pid
        while child.poll() is None:
            now = time.monotonic()
            rss,pids = helpers.tree_memory(child.pid,helpers.process_table())
            if not pids and child.poll() is None:
                raise RuntimeError('Running native child tree unobservable')
            result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'],rss)
            result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'],now-last)
            result['samples'] += 1
            item = dict(elapsed_s=now-start,rss_bytes=rss,pids=pids,interval_s=now-last)
            last = now
            if now-last_swap >= 5:
                swap = helpers.swap_state()
                growth = swap['used_bytes']-before['used_bytes']
                item.update(swap=swap,swap_growth_bytes=growth)
                result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'],growth)
                last_swap = now
            samples.write(json.dumps(item,allow_nan=False)+'\n'); samples.flush()
            if now-start>=900 or rss>=8_000_000_000 or result['maximum_swap_growth_bytes']>2*1024**3:
                result.update(terminated=True,reason='Independent wall/RSS/swap limit')
                helpers.stop_owned_group(child)
                break
            time.sleep(min(.25,max(0.,900-(time.monotonic()-start))))
        result['child_exit_code'] = child.wait(timeout=5)
        result['child_elapsed_s'] = time.monotonic()-start
        if result['child_elapsed_s']>=900:
            result.update(terminated=True,reason='Deadline exceeded at completion')
        rss,pids = helpers.tree_memory(child.pid,helpers.process_table())
        if pids:
            raise RuntimeError('Owned native descendants left alive')
        rp,cp = child_out/'result.json',child_out/'completion.json'
        if rp.exists() and cp.exists():
            record,seal = json.loads(rp.read_text()),json.loads(cp.read_text())
            result['child_result_sha256'] = sha(rp)
            result['passed'] = bool(not result['terminated'] and result['child_exit_code']==0
                and record['passed'] and record['hashes_unchanged'] and seal['passed']
                and seal['result_sha256']==sha(rp) and len(record['trials'])==3)
        for path,digest in pins.items():
            if sha(path)!=digest:
                raise RuntimeError('Supervisor input changed during acquisition: '+path)
        result.update(completed=True,sources_unchanged=True)
    except BaseException:
        result.update(passed=False,error=traceback.format_exc())
        if child is not None:
            helpers.stop_owned_group(child)
    finally:
        samples.close()
        result.update(elapsed_s=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat())
        try:
            result['after_swap'] = helpers.swap_state()
            if result['after_swap']['used_bytes']-before['used_bytes']>2*1024**3:
                result.update(passed=False,reason='Final swap-growth limit')
        except BaseException:
            result.update(passed=False,after_swap_error=traceback.format_exc())
        helpers.write(out/'result.json',result)
        helpers.write(out/'completion.json',dict(passed=result['passed'],completed=result['completed'],
                                               result_sha256=sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('case','passed','completed','terminated','elapsed_s',
          'maximum_sampled_rss_bytes','maximum_swap_growth_bytes','reason','error')},sort_keys=True),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
