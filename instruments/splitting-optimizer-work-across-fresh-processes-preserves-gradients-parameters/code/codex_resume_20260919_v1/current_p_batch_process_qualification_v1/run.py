from datetime import datetime,timezone
import os
import time
from iso_common import *

verify()
assert not (HERE/'run_claim.json').exists() and not (HERE/'result.json').exists()
assert read(FIT/'resource_failure.json')['closed'] and not numerical_workers()
write(HERE/'run_claim.json',dict(pid=os.getpid(),pgid=os.getpgrp(),utc=datetime.now(timezone.utc).isoformat(),
    sources=pin(HERE/'sources_sealed.json'),maximum_backward_graphs=16,maximum_updates=4,
    no_full_fit_launch=True,no_external_actions=True))
start=time.monotonic()
result=dict(completed=False,passed=False,error=None,no_retry=True,no_full_fit_launch=True,
    no_native_calls=True,no_candidate=True,no_external_actions=True)
try:
    guarded('reference',900,'worker.py','reference','0','2')
    guarded('split0',480,'worker.py','split0','0','1')
    guarded('split1',480,'worker.py','split1','1','2','--resume',HERE/'split0/checkpoint_01.json')
    guarded('qualify',180,'qualify.py')
    qualification=read(HERE/'qualification.json')
    assert qualification['passed']
    result.update(completed=True,passed=True,qualification=pin(HERE/'qualification.json'),backward_graphs=16,updates=4)
except BaseException as error:
    result['error']=repr(error)
finally:
    result.update(wall_s=time.monotonic()-start,sources=pin(HERE/'sources_sealed.json'))
    write(HERE/'result.json',result)
    print(result,flush=True)
raise SystemExit(0 if result['passed'] else 1)
