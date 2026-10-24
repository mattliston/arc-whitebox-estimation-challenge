"""Root-only worker20/setup3/contract. CONFIG CONFIG_SHA VALIDATION_ADMISSION_SHA."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time


def main():
    assert len(sys.argv)==4
    path=Path(sys.argv[1]).resolve(); raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
    c=json.loads(raw); source=Path(__file__).resolve().parent/'common.py'
    assert c['common']['path']==str(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==c['common']['sha256']
    spec=importlib.util.spec_from_file_location('serving_common',source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ctx=m.load_context(path,sys.argv[2],sys.argv[3]);q=ctx.q;c=ctx.c;study=ctx.study
    jobs=m.schedule(ctx);prefix=ctx.PREFIX
    outputs=[study/(prefix+x) for x in ('qualification_attempt.json','qualification_run.json','qualification_assessment.json')]
    outputs += [ctx.H2/(c['release_tag']+'_worker'),ctx.H2/(c['release_tag']+'_setup')]
    for name,_,_ in jobs:
        outputs += [study/(prefix+name+x) for x in ('.log','_guard.json','_job.json')]
    assert not any(p.exists() for p in outputs),'Existing serving outputs; no retry'
    attempt=study/(prefix+'qualification_attempt.json')
    q.once(attempt,dict(candidate=c['candidate'],config=ctx.config_pin,validation_admission=ctx.admission_pin,
        runner=c['runner'],inputs=ctx.inputs,jobs=[dict(name=n,seconds=s,command=cmd) for n,s,cmd in jobs],
        reused_boundary=ctx.broad['boundary'],runtime_recheck=ctx.runtime,serial=True,no_automatic_retry=True))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    completed=[]
    for name,seconds,command in jobs:
        m.load_context(path,sys.argv[2],sys.argv[3])
        log_path,guard_path=(study/(prefix+name+x) for x in ('.log','_guard.json'))
        guarded=[str(ctx.LIGHT),c['guard_source']['path'],'--record',str(guard_path),
                 '--seconds',str(seconds),'--rss-limit','7500000000',*command]
        started,error,returncode=time.monotonic(),None,None
        with log_path.open('x') as log:
            process=subprocess.Popen(guarded,cwd=ctx.H2,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:returncode=process.wait()
            except BaseException as exc:
                error=repr(exc)
                if process.poll() is None:process.send_signal(signal.SIGINT)
                returncode=process.wait()
        failures=[];gp=None
        if guard_path.exists():
            gp=q.pin(guard_path)
            try:m.guard(ctx,guard_path,seconds,command)
            except Exception as exc:failures.append('invalid_guard_receipt:'+repr(exc))
        else:failures.append('missing_guard_receipt')
        if returncode!=0:failures.append('guard_process_exit')
        if error is not None:failures.append('orchestrator_interrupted')
        try:
            for record in ctx.inputs:q.checked(record)
        except Exception as exc:failures.append('input_pin_changed:'+repr(exc))
        job=study/(prefix+name+'_job.json')
        q.once(job,dict(name=name,passed=not failures,failures=failures,command=command,seconds=seconds,
            guarded_command=guarded,guard_exit_code=returncode,error=error,
            orchestrator_elapsed_s=time.monotonic()-started,candidate=c['candidate'],config=ctx.config_pin,
            validation_admission=ctx.admission_pin,guard=gp,log=q.pin(log_path)))
        print(json.dumps(dict(job=name,passed=not failures,failures=failures)),flush=True)
        if failures:raise RuntimeError(f'{name} serving job failed; partial receipts retained: {failures}')
        completed.append(q.pin(job))
    m.load_context(path,sys.argv[2],sys.argv[3])
    q.once(study/(prefix+'qualification_run.json'),dict(completed=True,all_guards_passed=True,
        candidate=c['candidate'],config=ctx.config_pin,validation_admission=ctx.admission_pin,
        runner=c['runner'],reused_boundary=ctx.broad['boundary'],attempt=q.pin(attempt),jobs=completed,
        assessment_not_run=True,release_admitted=False,submitted=False))


if __name__=='__main__':main()
