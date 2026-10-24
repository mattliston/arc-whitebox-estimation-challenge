"""One natural-capacity opportunity, never alters or waits on the Nc worker."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

started=time.monotonic()
path=Path('cert/edband_used651_opportunity_v1.jsonl')
command=[sys.executable,'-B','../release/guard_footprint.py','--record','cert/edband_used651_guard_v1.json',
         '--seconds','180','--footprint-limit','5000000000','--rss-backstop','5000000000','/usr/bin/env']
envvars=dict(TRAINER_CARD='M',TRAINER_PL='1',TRAINER_F4C='1',TRAINER_THREADS='3',TRAINER_RELIEF='1',
             TRAINER_STAR_ATTACH='0',TRAINER_BIRTH_ATTACH='0',TRAINER_RSS_LIMIT='4500000000',
             OMP_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',MKL_NUM_THREADS='3',VECLIB_MAXIMUM_THREADS='3')
command += [key+'='+value for key,value in envvars.items()]+[sys.executable,'-B','cert/edband_used651_pair_v1.py']
outer={k:v for k,v in os.environ.items() if not k.startswith(('TRAINER_','NCAR6_'))}
with path.open('x') as stream:
    while time.monotonic()-started<180:
        vm=subprocess.check_output(['vm_stat'],text=True)
        size=int(re.search(r'page size of (\d+)',vm)[1])
        pages=lambda name:int(re.search(r'^'+re.escape(name)+r':\s+(\d+)',vm,re.M)[1])
        free=(pages('Pages free')+pages('Pages inactive'))*size/(1024**3)
        pressure=int(re.search(r'(\d+)%',subprocess.check_output(['memory_pressure'],text=True).strip().splitlines()[-1])[1])
        row=dict(utc=subprocess.check_output(['date','-u','+%FT%TZ'],text=True).strip(),headroom_gib=free,
                 threshold_gib=22,pressure_percent=pressure,admitted=free>=22 and pressure>=30)
        stream.write(json.dumps(row)+'\n');stream.flush()
        if row['admitted']:
            row['command']=command
            with Path('cert/edband_used651_launch_v1.json').open('x') as out:json.dump(row,out,indent=2)
            print(json.dumps(row),flush=True)
            raise SystemExit(subprocess.call(command,env=outer))
        time.sleep(10)
print(json.dumps(dict(status='PARKED_NO_NATURAL_CAPACITY',resource_samples=str(path),Nc_unchanged=True)))
