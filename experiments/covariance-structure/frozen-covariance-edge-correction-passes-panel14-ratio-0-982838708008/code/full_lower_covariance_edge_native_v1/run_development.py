import subprocess,sys
from common import HERE,D,RUNTIME,TORCH_RUNTIME,read,write,pin,seal_check,guarded
seal_check();stages=[]
jobs=[('build','build.py',[],TORCH_RUNTIME),('unit','unit.py',[],RUNTIME),
      ('disabled','native.py',['--out','disabled','--ranks','35,100,35','--disabled'],RUNTIME),
      ('enabled','native.py',['--out','enabled','--ranks','36,38,36'],RUNTIME),
      ('cold38','native.py',['--out','cold38','--ranks','38'],RUNTIME),
      ('qualify','qualify.py',[],RUNTIME),('score','score.py',[],RUNTIME)]
for name,script,args,runtime in jobs:
    command=[sys.executable,'-B',str(D/'run_guarded.py'),'--record',str(HERE/f'{name}_watch.json'),
             '--seconds','240',str(runtime),'-B',str(HERE/script),*args]
    print(dict(starting=name),flush=True)
    with (HERE/f'{name}.log').open('x') as f:r=subprocess.run(command,cwd=D,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:
        print(dict(failed=name,exit_code=r.returncode,log=str(HERE/f'{name}.log')),flush=True);raise SystemExit(r.returncode)
    g=guarded(name);stages.append(dict(name=name,guard=pin(HERE/f'{name}_watch.json'),log=pin(HERE/f'{name}.log')))
    print(dict(completed=name,elapsed_s=g['elapsed_s'],rss=g['max_rss_bytes'],added_swap=g['max_swap_growth_bytes']),flush=True)
write(HERE/'development_completed.json',dict(completed=True,stages=stages,passed=read(HERE/'score.json')['passed']))
print(read(HERE/'score.json')['passed'],flush=True)
