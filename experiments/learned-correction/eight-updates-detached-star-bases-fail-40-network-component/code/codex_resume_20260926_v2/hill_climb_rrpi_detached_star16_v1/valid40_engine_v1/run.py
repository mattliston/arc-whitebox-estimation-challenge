"""Root-only serial VALID40; CONFIG CONFIG_SHA BROAD_ADMISSION_SHA. No retry."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys


def main():
    assert len(sys.argv) == 4
    path = Path(sys.argv[1]).resolve(); raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sys.argv[2]
    c = json.loads(raw); source = Path(__file__).resolve().parent/'assess.py'
    assert c['engine']['path'] == str(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == c['engine']['sha256']
    spec = importlib.util.spec_from_file_location('fixed_valid40_assessor',source)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    q = m.initialize(); cp,c = m.configure(q,path,sys.argv[2])
    assert c['runner'] == q.pin(__file__)
    ap,a,runtime = m.prerequisites(q,cp,c,sys.argv[3]); study=path.parent
    guard_source = c['guard_source']; light = q.H2.parent.parents[3]/'.venv/bin/python'
    assert guard_source == q.pin(q.H2.parent/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py')
    jobs=[]
    for key in m.ROSTERS:
        b=c['candidate']['blocks'][key]
        assert b['no_bytecode'] is True
        gp=study/f'broad_{key}_guard.json'; log=study/f'broad_{key}.log'; receipt=study/f'broad_{key}_job.json'
        assert b['guard']==str(gp)
        for p in (gp,log,receipt,q.H2/b['tag']): assert not p.exists(),str(p)
        command=[str(q.NATIVE),'-B',b['driver']['path'],c['candidate']['source']['path'],
                 ','.join(map(str,b['ranks'])),b['tag']]
        jobs.append(dict(block=key,command=command,guard=str(gp),log=str(log),receipt=str(receipt)))
    for name in ('broad_attempt.json','broad_run.json','broad_run_failure.json','broad_assessment.json'):
        assert not (study/name).exists()
    q.once(study/'broad_attempt.json',dict(config=cp,admission=ap,candidate=c['candidate']['source'],
        runner=c['runner'],assessor=c['engine'],jobs=jobs,runtime_recheck=runtime,serial=True,no_automatic_retry=True))
    receipts=[]
    try:
        for job in jobs:
            m.configure(q,path,sys.argv[2]); m.prerequisites(q,cp,c,sys.argv[3])
            guarded=[str(light),'-B',guard_source['path'],'--record',job['guard'],
                     '--seconds','420','--rss-limit','7500000000',*job['command']]
            print(json.dumps(dict(event='launch',block=job['block'],command=job['command'])),flush=True)
            with Path(job['log']).open('x') as stream:
                process=subprocess.run(guarded,stdout=stream,stderr=subprocess.STDOUT)
            receipt=dict(block=job['block'],candidate=c['candidate']['source'],config=cp,admission=ap,
                command=job['command'],guarded_command=guarded,guard_exit_code=process.returncode,log=q.pin(job['log']))
            try:
                assert process.returncode==0
                g=q.guard(job['guard'],m.LIMITS,job['command'])
                receipt.update(passed=True,guard=q.pin(job['guard']))
            except Exception as exc:
                receipt.update(passed=False,error=dict(type=type(exc).__name__,message=str(exc)))
                if Path(job['guard']).exists(): receipt['guard']=q.pin(job['guard'])
                q.once(Path(job['receipt']),receipt); raise
            q.once(Path(job['receipt']),receipt); receipts.append(q.pin(job['receipt']))
            print(json.dumps(dict(event='retired',block=job['block'],elapsed_s=g['elapsed_s'],max_rss_bytes=g['max_rss_bytes'])),flush=True)
        m.prerequisites(q,cp,c,sys.argv[3])
        q.once(study/'broad_run.json',dict(completed=True,all_guards_passed=True,config=cp,admission=ap,
            candidate=c['candidate']['source'],runner=c['runner'],attempt=q.pin(study/'broad_attempt.json'),
            jobs=receipts,assessment_not_run=True,release_qualified=False,submitted=False))
    except BaseException as exc:
        q.once(study/'broad_run_failure.json',dict(completed=False,config=cp,admission=ap,completed_jobs=receipts,
            error=dict(type=type(exc).__name__,message=str(exc)),no_automatic_retry=True))
        raise


if __name__ == '__main__': main()
