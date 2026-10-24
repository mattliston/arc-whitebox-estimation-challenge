import datetime
import os
import time
from fit_common import *
verify(data=True);assert read(HERE/'admission.json')['passed'];assert not numerical_workers();assert os.getpgrp()==os.getpid()
write(HERE/'run_claim.json',dict(started=True,pid=os.getpid(),pgid=os.getpgrp(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_seal=pin(HERE/'sources_sealed.json'),admission=pin(HERE/'admission.json')))
started=time.monotonic();result=dict(completed=False,passed=False,error=None,no_automatic_native_launch=True,no_upload=True)
try:
    for update in range(40):
        assert time.monotonic()-started<14400
        args=[update]
        if update:args+=['--resume',HERE/f'batch_{update-1:02d}'/f'checkpoint_{update:02d}.json']
        guarded(f'batch_{update:02d}',480,'worker.py',*args)
    guarded('verify_fit',180,'verify_fit.py')
    f=read(HERE/'fit_result.json');assert f['completed'] and f['passed']
    result.update(completed=True,passed=True,fit=pin(HERE/'fit_result.json'),training_graphs=160,updates=40)
except BaseException as e:
    result['error']=repr(e);raise
finally:
    result['wall_s']=time.monotonic()-started;result['sources']=pin(HERE/'sources_sealed.json')
    write(HERE/'controller_result.json',result);print(result,flush=True)
