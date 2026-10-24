"""Independent source-only native builder watch; no training or native run."""
import argparse
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

HERE=Path(__file__).resolve().parent
CAP,RSS=60.,2_000_000_000
KEYS=('source','watch','protocol','watch_spec','training','training_watch','payload')


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path,tick=None):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
            if tick is not None:tick()
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def command(argv):
    return subprocess.run(argv,text=True,capture_output=True,check=True,timeout=.6).stdout


def snapshot():
    return [tuple(map(int,line.split())) for line in
        command(['/bin/ps','-axo','pid=,ppid=,pgid=,rss=']).splitlines()]


def swap_used():
    found=re.search(r'used\s*=\s*([\d.]+)([MG])',command(['/usr/sbin/sysctl','vm.swapusage']))
    require(found is not None,'Actual system swap observation')
    return round(float(found.group(1))*(1024**2 if found.group(2)=='M' else 1024**3))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in KEYS:ap.add_argument('--'+key.replace('_','-')+'-sha256',required=True)
    args=ap.parse_args();began=last=time.monotonic()
    directory=HERE/'native_build_watch_v1';output=HERE/'native_build_v1'
    require(not directory.exists() and not output.exists(),'One fresh source build, no overwrite/resume')
    require(Path(sys.executable).resolve()==Path('/Users/matt/anaconda3/bin/python').resolve()
        and os.getpriority(os.PRIO_PROCESS,0)>=10 and sys.dont_write_bytecode and sys.flags.optimize==0,
        'Fixed nice10 stdlib interpreter and assertions')
    require(not {'numpy','torch','flopscope','scipy'}&set(sys.modules),'No numerical imports in watcher')
    directory.mkdir();child=None;log=None;swap0=None
    result=dict(passed=False,completed=False,numerical_imports=False,truth_accessed=False,
        estimator_executed=False,observations=0,maximum_combined_rss_bytes=0,
        maximum_swap_growth_bytes=0,maximum_interval_s=0.)
    def observe():
        nonlocal last
        rows=snapshot();swap=swap_used();now=time.monotonic();gap=now-last;last=now
        ids={os.getpid()}
        if child is not None:ids.add(child.pid)
        while True:
            before=len(ids)
            ids.update(pid for pid,ppid,pgid,rss in rows if ppid in ids or
                (child is not None and pgid==child.pid))
            if len(ids)==before:break
        rss=sum(value*1024 for pid,ppid,pgid,value in rows if pid in ids)
        growth=0 if swap0 is None else max(0,swap-swap0)
        result['observations']+=1
        result['maximum_combined_rss_bytes']=max(result['maximum_combined_rss_bytes'],rss)
        result['maximum_swap_growth_bytes']=max(result['maximum_swap_growth_bytes'],growth)
        result['maximum_interval_s']=max(result['maximum_interval_s'],gap)
        require(now-began<CAP and gap<=1. and rss<RSS and growth==0,
            'Whole60s, completed observations<=1s, combined2GB, zero added swap')
        return rows
    def tick():
        if time.monotonic()-last>=.15:observe()
        require(time.monotonic()-began<CAP,'Whole source build60s')
    try:
        sources={HERE/'build_native.py':args.source_sha256,HERE/'watch_build_native.py':args.watch_sha256,
            HERE/'BUILD_NATIVE_PROTOCOL.md':args.protocol_sha256,HERE/'BUILD_NATIVE_WATCH_SPEC.md':args.watch_spec_sha256}
        require(all(re.fullmatch('[0-9a-f]{64}',getattr(args,key+'_sha256')) for key in KEYS),
            'All exact prospective direct and actual prerequisite hashes')
        swap0=swap_used();observe()
        for path,digest in sources.items():
            require(path.is_file() and not path.is_symlink() and sha(path,tick)==digest,'Reviewed direct source owner')
        result['source_bindings']={str(p):h for p,h in sources.items()}
        argv=[sys.executable,'-B','-u',str(HERE/'build_native.py'),'--run']
        for key in KEYS:argv.extend(('--'+key.replace('_','-')+'-sha256',getattr(args,key+'_sha256')))
        result['worker_argv']=argv
        log=(directory/'worker.log').open('x')
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
        result['worker_pid']=child.pid
        while True:
            observe()
            if child.poll() is not None:break
            time.sleep(.15)
        code=child.wait(timeout=1);log.close();log=None
        require(not any(pgid==child.pid for pid,ppid,pgid,rss in snapshot()),'Actual child group retired')
        names=('result.json','completion.json','finalization.json')
        records=[json.loads((output/name).read_text()) for name in names]
        rh,ch,fh=(sha(output/name,tick) for name in names)
        require(all(r['passed'] is True and r['completed'] is True for r in records) and code==0
            and records[1]['result_sha256']==records[2]['result_sha256']==rh
            and records[2]['completion_sha256']==ch,'Actual positive child three-seal chain')
        r=records[0]
        require(r['estimator_executed'] is False and r['truth_accessed'] is False
            and r['parameter_updates']==0 and r['numerical_imports'] is False
            and r['payload_decoded'] and r['source_build_qualified'] and r['all_owners_retired']
            and r['payload_sha256']==args.payload_sha256,'Only sole-payload representation build, no estimator run')
        require(sha(HERE/'build_native.py',tick)==args.source_sha256,'Reviewed inert verifier source')
        spec=importlib.util.spec_from_file_location('owned_anchored_native_build_verifier',HERE/'build_native.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        verified=builder.verify_build(rh,r['candidate_sha256'],args.source_sha256,args.protocol_sha256,tick)
        require(verified['candidate_sha256']==r['candidate_sha256'],'Independently verified complete source build')
        for path,digest in sources.items():require(sha(path,tick)==digest,'Every direct source unchanged')
        require(not {'numpy','torch','flopscope','scipy'}&set(sys.modules),'No hidden numerical verifier import')
        result.update(passed=True,completed=True,whole_owned_tree_retired=True,
            child_result_sha256=rh,child_completion_sha256=ch,child_finalization_sha256=fh,
            candidate_sha256=r['candidate_sha256'],payload_sha256=args.payload_sha256,
            source_build_qualified=True,native_score_measured=False,release_ready=False)
    except BaseException:result['error']=traceback.format_exc()
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=2)
            except ProcessLookupError:child.wait(timeout=2)
        if log is not None:log.close()
        if child is not None:
            try:
                remaining=[pid for pid,ppid,pgid,rss in snapshot() if pgid==child.pid]
                if remaining:os.killpg(child.pid,signal.SIGKILL)
                result['whole_owned_tree_retired']=not remaining
            except BaseException:result['whole_owned_tree_retired']=False
        try:observe()
        except BaseException:result['passed']=False;result['tail_error']=traceback.format_exc()
        result['elapsed_s']=time.monotonic()-began
        result['passed']=bool(result['passed'] and result.get('whole_owned_tree_retired') and result['elapsed_s']<CAP)
        write(directory/'result.json',result);rh=sha(directory/'result.json')
        require(time.monotonic()<min(began+CAP,last+1.),'Bounded result publication tail')
        write(directory/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=rh))
        require(time.monotonic()<min(began+CAP,last+1.),'Bounded completion publication tail')
        write(directory/'finalization.json',dict(passed=result['passed'],completed=result['completed'],
            result_sha256=rh,completion_sha256=sha(directory/'completion.json'),
            elapsed_s=time.monotonic()-began,bounded_publication_tail=True))
        require(time.monotonic()<min(began+CAP,last+1.),'Final seal inside observed deadline')
    print(json.dumps({k:result.get(k) for k in ('passed','completed','elapsed_s',
        'maximum_combined_rss_bytes','maximum_swap_growth_bytes','maximum_interval_s','error','tail_error')}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
