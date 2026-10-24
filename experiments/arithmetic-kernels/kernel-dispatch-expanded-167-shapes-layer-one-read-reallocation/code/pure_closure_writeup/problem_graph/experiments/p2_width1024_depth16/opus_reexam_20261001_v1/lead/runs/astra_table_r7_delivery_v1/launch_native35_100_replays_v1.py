"""opus-release: launch both b709 captured-setup replay controllers concurrently, wait, record actual exits."""
import json,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;PY='/Users/matt/arc-white-box-estimation/.venv/bin/python'
procs={}
for mode in ('normal','slow'):
    Z=D/f'native35_100_replay_{mode}_v1'
    log=open(Z/'controller.log','x')
    procs[mode]=(subprocess.Popen([PY,'-B',str(Z/'run_native35_100_v1.py')],stdout=log,stderr=subprocess.STDOUT,cwd=Z),Z)
    with open(Z/'LAUNCH_v1.json','x') as f: json.dump(dict(controller_pid=procs[mode][0].pid,mode=mode,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),f,indent=2)
    time.sleep(3)
for mode,(p,Z) in procs.items():
    rc=p.wait()
    with open(Z/'ACTUAL_EXIT_v1.json','x') as f: json.dump(dict(controller_pid=p.pid,actual_exit_observed=True,exit_code=rc,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),f,indent=2)
    print(mode,'exit',rc,flush=True)
