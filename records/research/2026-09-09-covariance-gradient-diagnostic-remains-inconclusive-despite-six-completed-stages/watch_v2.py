"""Independent v2 stdlib watchdog; preserves the unexecuted v1 source.

No arbitrary command path, descendant tree kill, numerical import or retry.
Only the exact child spawned here may be terminated.  All output is new.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
RUNNER = HERE / 'run.py'
STAGES = ('tiny','cpu_forward','mps_forward','cpu_gradient','mps_gradient','judge')
PINS = {
    HERE/'watch.py': 'd52898bb08fdc9c02ad251f2f539b5cdbce30bc50e7a054323bcc1d9dd8b7928',
    RUNNER: '81aae49d2985059bc3e41e02db8d3bc6cfddffb22331f6f8be16ba378ca281f5',
    HERE/'model.py': 'faf98e7f35820cf80074386153aea7f9d9c5b10c0c567d30ab97b0da559b281a',
    HERE/'PROTOCOL.md': '8c82e58285a90cd859bd9a86f27130e8ab68550b8b818725388351994bfb9957',
    HERE/'IMPLEMENTATION.md': 'fb5ce0a8830ff5cbb18e558677c84cf013b86e9a8a63748aa7b407e3ed1ad209',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')

def system_snapshot():
    swap = subprocess.run(['/usr/sbin/sysctl','vm.swapusage'],capture_output=True,text=True,check=True,timeout=5).stdout.strip()
    data = subprocess.run(['/bin/ps','-axo','pid,ppid,nice,rss,command'],capture_output=True,text=True,check=True,timeout=5).stdout
    processes = [line for line in data.splitlines() if any(t in line for t in ('python','flopscope-server','winner_lock'))]
    return dict(utc=datetime.now(timezone.utc).isoformat(),swapusage=swap,process_inventory=processes)

def stop_child(child):
    if child.poll() is not None: return
    child.terminate()
    try: child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        child.kill(); child.wait(timeout=5)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch-source-sha256',required=True)
    parser.add_argument('--supplement-sha256',required=True)
    args, child_args=parser.parse_known_args()
    if child_args and child_args[0]=='--': child_args=child_args[1:]
    if '--stage' not in child_args: parser.error('Explicit child stage required')
    stage=child_args[child_args.index('--stage')+1]
    if stage not in STAGES: parser.error('Invalid child stage')
    if os.getpriority(os.PRIO_PROCESS,0)<10: raise RuntimeError('Watchdog and child require nice10')
    checks=dict(PINS)
    checks[Path(__file__).resolve()]=args.watch_source_sha256
    checks[HERE/'RESOURCE_QUALIFICATION_V2.md']=args.supplement_sha256
    for path,digest in checks.items():
        if sha(path)!=digest: raise RuntimeError('Reviewed watchdog input changed: '+str(path))
    out=HERE/('watch_'+stage+'_v1')
    out.mkdir(exist_ok=False)
    interpreter=(HERE.parent.parent/'data_gate/.venv/bin/python' if stage=='judge'
                 else Path('/Users/matt/anaconda3/bin/python'))
    command=[str(interpreter),str(RUNNER),*child_args]
    limit=120 if stage in ('tiny','judge') else 900
    result=dict(schema='current_covariance_gap_external_watch_v1',stage=stage,completed=False,
                passed=False,source_sha256={str(p):d for p,d in checks.items()},command=command,
                wall_limit_seconds=limit,rss_limit_bytes=12000000000,requested_period_seconds=.25,
                child_pid=None,terminated=False,before=system_snapshot(),maximum_sampled_rss_bytes=0,
                maximum_sample_interval_seconds=0,samples=0)
    write(out/'started.json',result)
    child=None; start=time.monotonic(); last=start
    samples=(out/'resources.jsonl').open('x')
    try:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL)
        result['child_pid']=child.pid
        while child.poll() is None:
            now=time.monotonic()
            if now-start>=limit:
                result['terminated']=True
                result['reason']='independent wall deadline'
                stop_child(child); break
            ps=subprocess.run(['/bin/ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True,timeout=2)
            if ps.returncode==0 and ps.stdout.strip():
                rss=int(ps.stdout.strip())*1024
                interval=now-last; last=now
                result['maximum_sampled_rss_bytes']=max(result['maximum_sampled_rss_bytes'],rss)
                result['maximum_sample_interval_seconds']=max(result['maximum_sample_interval_seconds'],interval)
                result['samples']+=1
                samples.write(json.dumps(dict(elapsed_s=now-start,rss_bytes=rss,interval_s=interval))+'\n')
                samples.flush()
                if rss>=12000000000:
                    result['terminated']=True; result['reason']='independent sampled RSS limit'
                    stop_child(child); break
            elif child.poll() is None:
                raise RuntimeError('Cannot observe still-running child RSS')
            remaining=max(0.,min(.25,limit-(time.monotonic()-start)))
            if remaining: time.sleep(remaining)
        result['child_exit_code']=child.wait(timeout=5)
        result['child_elapsed_s']=time.monotonic()-start
        if result['child_elapsed_s'] >= limit:
            result['terminated']=True
            result['reason']='deadline exceeded at completion check'
        child_folder=HERE/('stage_'+stage+'_v1')
        rp,cp=child_folder/'result.json',child_folder/'completion.json'
        if rp.exists() and cp.exists():
            rec=json.loads(rp.read_text()); seal=json.loads(cp.read_text())
            result['child_result_sha256']=sha(rp)
            result['child_completed']=bool(rec['completed'] and seal['completed']
                                            and seal['result_sha256']==sha(rp))
            result['passed']=bool(not result['terminated'] and result['child_exit_code']==0
                                  and rec['passed'] and result['child_completed'])
        for path,digest in checks.items():
            if sha(path)!=digest: raise RuntimeError('Watchdog source changed during execution')
        result['sources_unchanged']=True
        result['completed']=True
    except BaseException as exc:
        result['passed']=False
        result['error']=repr(exc);result['traceback']=traceback.format_exc()
        if child is not None: stop_child(child)
    finally:
        samples.close()
        result['elapsed_s']=time.monotonic()-start
        try: result['after']=system_snapshot()
        except BaseException as exc: result['after_snapshot_error']=repr(exc);result['passed']=False
        write(out/'result.json',result)
        write(out/'completion.json',dict(completed=result['completed'],passed=result['passed'],result_sha256=sha(out/'result.json')))
    print(json.dumps({k:result.get(k) for k in ('stage','passed','completed','child_result_sha256','elapsed_s',
          'maximum_sampled_rss_bytes','maximum_sample_interval_seconds','samples','terminated','reason','error')},sort_keys=True),flush=True)
    return 0 if result['passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())

