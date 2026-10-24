"""Independent whole120s/8GB/zero-swap supervisor for two serial ENTRY8 children."""
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

HERE=Path(__file__).resolve().parent
P2=HERE.parent.parent
AP=Path('/Users/matt/anaconda3/bin/python')
OFFPY=P2/'data_gate/.venv/bin/python'
NAMES={'entry8_worker.py','watch_entry8.py','entry8_diagnostic.py',
       'ENTRY8_SELECTION_20260913_UTC.md'}
CAP=120.
RSS=8_000_000_000


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def command(args):
    return subprocess.run(args,text=True,capture_output=True,check=True,timeout=.6).stdout


def snapshot():
    return [tuple(map(int,row.split())) for row in
            command(['/bin/ps','-axo','pid=,ppid=,pgid=,rss=']).splitlines()]


def swap():
    m=re.search(r'used\s*=\s*([\d.]+)([MG])',command(['/usr/sbin/sysctl','vm.swapusage']))
    require(m is not None,'Actual swap observation')
    return round(float(m[1])*(1024**2 if m[2]=='M' else 1024**3))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pins',required=True)
    args=ap.parse_args();began=time.monotonic();last=began
    directory=HERE/'entry8_watch_v1'
    require(not directory.exists() and all(not (HERE/('entry8_'+p+'_v1')).exists()
            for p in ('native','torch')),'Exclusive fixed two-stage run')
    directory.mkdir()
    result=dict(passed=False,completed=False,maximum_combined_rss_bytes=0,
                maximum_swap_growth_bytes=0,maximum_interval_s=0.,observations=0,children=[])
    child=None;log=None;swap0=None;groups=[]
    def observe():
        nonlocal last
        rows=snapshot();current=swap();now=time.monotonic()
        ids={os.getpid()}
        if child is not None:ids.add(child.pid)
        while True:
            previous=len(ids)
            ids.update(pid for pid,ppid,pgid,rss in rows if ppid in ids or pgid in groups)
            if previous==len(ids):break
        combined=sum(rss*1024 for pid,ppid,pgid,rss in rows if pid in ids)
        growth=0 if swap0 is None else max(0,current-swap0)
        gap=now-last;last=now
        result['observations']+=1
        result['maximum_combined_rss_bytes']=max(result['maximum_combined_rss_bytes'],combined)
        result['maximum_swap_growth_bytes']=max(result['maximum_swap_growth_bytes'],growth)
        result['maximum_interval_s']=max(result['maximum_interval_s'],gap)
        require(now-began<CAP and gap<=1.,'Whole120s, completed observation intervals<=1s')
        require(combined<RSS and growth==0,'Observed owned tree<8GB and zero added swap')
        return rows
    try:
        pins=read(args.pins)
        require(set(pins)==NAMES and all(sha(HERE/k)==v for k,v in pins.items()),
                'Exact reviewed four-source roster')
        require(Path(sys.executable).resolve()==AP.resolve() and
                os.getpriority(os.PRIO_PROCESS,0)>=10,'Fixed nice10 stdlib supervisor')
        swap0=swap();native_seals=None
        for phase,python in (('native',OFFPY),('torch',AP)):
            observe()
            output=HERE/('entry8_'+phase+'_v1')
            cmd=[str(python),'-B','-u',str(HERE/'entry8_worker.py'),
                 '--phase',phase,'--pins',str(Path(args.pins).resolve())]
            if phase=='torch':
                require(native_seals is not None and
                        not any(pgid in groups for pid,ppid,pgid,rss in snapshot()),
                        'Native group actually retired before Torch launch')
                cmd+=['--native-seals',json.dumps(native_seals,sort_keys=True)]
            log=(directory/(phase+'.log')).open('x')
            child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True,close_fds=True)
            groups.append(child.pid)
            while True:
                observe()
                if child.poll() is not None:break
                time.sleep(.15)
            code=child.wait(timeout=1);log.close();log=None
            paths=[output/k for k in ('result.json','completion.json','finalization.json')]
            seals={p.name:sha(p) for p in paths};r,c,f=map(read,paths)
            require(code==0 and all(v['passed'] and v['completed'] for v in (r,c,f)) and
                    c['result_sha256']==f['result_sha256']==seals['result.json'] and
                    f['completion_sha256']==seals['completion.json'] and
                    r['phase']==phase and r['all_owners_retired'] and
                    r['truth_accessed'] is False and r['score_measured'] is False and
                    r['full_model_calls']==r['parameter_updates']==0,
                    'Actual operational child three-seal completion')
            require(not any(pgid==child.pid for pid,ppid,pgid,rss in snapshot()),
                    'Exited child group empty')
            result['children'].append(dict(phase=phase,pid=child.pid,seals=seals,group_retired=True))
            if phase=='native':
                require(r['own_complete_bill'] and 0<r['complete_research_flops']<=
                        r['cost_certificate']['native_upper'],'Complete native price')
                native_seals=seals
            else:
                require(r['native_seals']==native_seals,'Torch joined exact prior native child')
                result.update(diagnostic_outcome=r['diagnostic_outcome'],
                    rangefinder_dominant_support=r['rangefinder_dominant_support'],
                    native_Y_over_own_Y_squared_error=r['native_Y_over_own_Y_squared_error'],
                    complete_source_planning_upper=r['complete_source_planning_upper'])
            child=None
        require(all(sha(HERE/k)==v for k,v in pins.items()),'Source roster unchanged')
        result.update(passed=True,completed=True,source_bindings=pins)
    except BaseException:
        result['error']=traceback.format_exc()
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=2)
            except ProcessLookupError:child.wait(timeout=2)
        if log is not None:log.close()
        try:
            rows=observe()
            remaining={pgid for pid,ppid,pgid,rss in rows if pgid in groups}
            for pgid in remaining:os.killpg(pgid,signal.SIGKILL)
            result['whole_owned_tree_retired']=not remaining
            require(not remaining,'No lingering owned process group')
        except BaseException:
            result.update(passed=False,tail_error=traceback.format_exc())
        result['elapsed_s']=time.monotonic()-began
        result['passed']=bool(result['passed'] and result.get('whole_owned_tree_retired') and
                              result['elapsed_s']<CAP)
        write(directory/'result.json',result);rh=sha(directory/'result.json')
        require(time.monotonic()<min(began+CAP,last+1.),'Bounded result publication')
        write(directory/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=rh))
        require(time.monotonic()<min(began+CAP,last+1.),'Bounded completion publication')
        write(directory/'finalization.json',dict(passed=result['passed'],completed=result['completed'],
            result_sha256=rh,completion_sha256=sha(directory/'completion.json'),
            elapsed_s=time.monotonic()-began,bounded_publication_tail=True))
        require(time.monotonic()<min(began+CAP,last+1.),'Bounded final seal')
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
