"""One guarded numerical group at a time; all log files exclusive and stable."""
import subprocess
import sys
from common import HERE,D,RUNTIME,read,write,pin,seal_check,guarded

seal_check();stages=[]
jobs=[('unit_v3','unit_v3.py',[]),
      ('disabled','native_v3.py',['--out','disabled','--ranks','35,100,35','--disabled']),
      ('enabled','native_v3.py',['--out','enabled','--ranks','35,100,35']),
      ('cold100','native_v3.py',['--out','cold100','--ranks','100']),
      ('qualify','qualify.py',[]),('score','score.py',[])]
for name,script,args in jobs:
    command=[sys.executable,'-B',str(D/'run_guarded.py'),'--record',str(HERE/f'{name}_watch.json'),
             '--seconds','240',str(RUNTIME),'-B',str(HERE/script),*args]
    print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as f:
        completed=subprocess.run(command,cwd=D,stdout=f,stderr=subprocess.STDOUT)
    if completed.returncode:
        print(dict(failed=name,exit_code=completed.returncode,log=str(HERE/f'{name}.log')),flush=True)
        raise SystemExit(completed.returncode)
    g=guarded(name);stages.append(dict(name=name,guard=pin(HERE/f'{name}_watch.json'),log=pin(HERE/f'{name}.log')))
    print(dict(completed=name,elapsed_s=g['elapsed_s'],rss=g['max_rss_bytes'],added_swap=g['max_swap_growth_bytes']),flush=True)
write(HERE/'development_completed.json',dict(stages=stages,local_study_completed=True,
    development_passed=read(HERE/'score.json')['passed']))
print(read(HERE/'score.json')['passed'],flush=True)
