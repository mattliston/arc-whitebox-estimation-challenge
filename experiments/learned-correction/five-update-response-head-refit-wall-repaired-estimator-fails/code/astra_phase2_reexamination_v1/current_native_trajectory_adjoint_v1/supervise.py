"""Independent bounded supervisor for a single fresh native/adjoint worker."""
import argparse
import hashlib
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
CAP,RSS = 120.,8_000_000_000


def require(ok,message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''): h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')


def command(args):
    return subprocess.run(args,text=True,capture_output=True,check=True,timeout=.6).stdout


def swap_used():
    m = re.search(r'used\s*=\s*([\d.]+)([MG])',command(['/usr/sbin/sysctl','vm.swapusage']))
    require(m is not None,'Actual swap observation')
    return round(float(m.group(1))*(1024**2 if m.group(2) == 'M' else 1024**3))


def snapshot():
    return [tuple(map(int,line.split())) for line in
        command(['/bin/ps','-axo','pid=,ppid=,pgid=,rss=']).splitlines()]


def owned_rss(rows,worker):
    ids = {os.getpid()}
    if worker is not None: ids.add(worker)
    while True:
        before = len(ids)
        ids.update(pid for pid,ppid,pgid,rss in rows if ppid in ids or
            (worker is not None and pgid == worker))
        if len(ids) == before: break
    return sum(rss*1024 for pid,ppid,pgid,rss in rows if pid in ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=('native','adjoint'),required=True)
    ap.add_argument('--pins',required=True)
    args = ap.parse_args(); began=time.monotonic(); last=began
    output = HERE/(args.phase+'_v1'); directory=HERE/(args.phase+'_watch_v1')
    require(not directory.exists() and not output.exists(),'Exclusive fresh stage, no resume/overwrite')
    directory.mkdir(); child=None; swap0=None; log=None
    result = dict(passed=False,completed=False,phase=args.phase,observations=0,
        maximum_combined_rss_bytes=0,maximum_swap_growth_bytes=0,maximum_interval_s=0.)
    def observe():
        nonlocal last
        rows=snapshot(); swap=swap_used(); now=time.monotonic()
        gap=now-last; last=now
        rss=owned_rss(rows,None if child is None else child.pid)
        growth=0 if swap0 is None else max(0,swap-swap0)
        result['observations']+=1
        result['maximum_interval_s']=max(result['maximum_interval_s'],gap)
        result['maximum_combined_rss_bytes']=max(result['maximum_combined_rss_bytes'],rss)
        result['maximum_swap_growth_bytes']=max(result['maximum_swap_growth_bytes'],growth)
        require(now-began<CAP and gap<=1. and rss<RSS and growth==0,
            'Whole120s, completed observations<=1s, tree<8GB, zero added swap')
        return rows
    try:
        pins=json.loads(Path(args.pins).read_text())
        require(all(Path(k).name == k and sha(HERE/k) == v for k,v in pins.items())
            and 'supervise.py' in pins,'Exact reviewed direct local source owners')
        worker='run_'+args.phase+'.py'
        require(worker in pins,'Reviewed fixed worker')
        expected=(HERE.parent.parent/'data_gate/.venv/bin/python' if args.phase=='native'
            else Path('/Users/matt/anaconda3/bin/python'))
        require(Path(sys.executable).resolve()==expected.resolve()
            and os.getpriority(os.PRIO_PROCESS,0)>=10,'Fixed stage interpreter and inherited nice10')
        swap0=swap_used(); log=(directory/'worker.log').open('x')
        child=subprocess.Popen([sys.executable,'-B','-u',str(HERE/worker),'--run',
            '--pins',str(Path(args.pins).resolve())],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        result['worker_pid']=child.pid
        while True:
            observe()
            if child.poll() is not None: break
            time.sleep(.15)
        code=child.wait(timeout=1); log.close(); log=None
        names=('result.json','completion.json','finalization.json')
        records=[json.loads((output/k).read_text()) for k in names]
        rh,ch,fh=(sha(output/k) for k in names)
        require(all(x['completed'] is True for x in records)
            and all(x['passed']==records[0]['passed'] for x in records)
            and code==(0 if records[0]['passed'] else 1)
            and records[1]['result_sha256']==records[2]['result_sha256']==rh
            and records[2]['completion_sha256']==ch,'Actual completed child three-seal chain')
        require(records[0]['truth_accessed'] is False and records[0]['fits']==0
            and records[0]['full_model_calls']==0 and records[0]['all_owners_retired'],
            'Actual target-free retired local qualifier, not a fitted estimator')
        if args.phase=='native':
            require(records[0]['partial_model_calls']==records[0]['retired_calls']==3,
                'Actual N/plus/minus native calls')
        require(not any(pgid==child.pid for pid,ppid,pgid,rss in snapshot()),'Worker process group retired')
        require(all(sha(HERE/k)==v for k,v in pins.items()),'Exact sources unchanged')
        result.update(passed=records[0]['passed'],completed=True,whole_owned_tree_retired=True,
            source_bindings=pins,child_result_sha256=rh,child_completion_sha256=ch,
            child_finalization_sha256=fh)
    except BaseException:
        result['error']=traceback.format_exc()
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid,signal.SIGTERM); child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait(timeout=2)
            except ProcessLookupError: child.wait(timeout=2)
        if log is not None: log.close()
        if child is not None:
            try:
                remaining=[pid for pid,ppid,pgid,rss in snapshot() if pgid==child.pid]
                if remaining: os.killpg(child.pid,signal.SIGKILL)
                result['whole_owned_tree_retired']=not remaining
            except BaseException: result['whole_owned_tree_retired']=False
        try: observe()
        except BaseException:
            result['passed']=False; result['tail_error']=traceback.format_exc()
        result['elapsed_s']=time.monotonic()-began
        result['passed']=bool(result['passed'] and result.get('whole_owned_tree_retired')
            and result['elapsed_s']<CAP)
        write(directory/'result.json',result); rh=sha(directory/'result.json')
        require(time.monotonic()<min(began+CAP,last+1.),'Bounded result tail')
        write(directory/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=rh))
        require(time.monotonic()<min(began+CAP,last+1.),'Bounded completion tail')
        write(directory/'finalization.json',dict(passed=result['passed'],completed=result['completed'],
            result_sha256=rh,completion_sha256=sha(directory/'completion.json'),
            elapsed_s=time.monotonic()-began,bounded_publication_tail=True))
        require(time.monotonic()<min(began+CAP,last+1.),'Final seal inside observed deadline')
    print(json.dumps({k:result.get(k) for k in ('passed','completed','phase','elapsed_s',
        'maximum_combined_rss_bytes','maximum_swap_growth_bytes','maximum_interval_s','error','tail_error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
