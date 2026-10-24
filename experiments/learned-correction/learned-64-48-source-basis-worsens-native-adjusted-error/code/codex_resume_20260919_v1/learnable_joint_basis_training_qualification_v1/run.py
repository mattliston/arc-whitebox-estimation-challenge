import datetime,time
from common import *
verify();assert not numerical_workers();assert not (HERE/'run_claim.json').exists()
write(HERE/'run_claim.json',dict(pid=os.getpid(),pgid=os.getpgrp(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=pin(HERE/'sources_sealed.json'),no_fit_or_upload=True))
started=time.monotonic();result=dict(completed=False,passed=False,error=None,no_retry=True,no_fit_or_upload=True)
try:
    guarded('unit',180,'unit.py')
    assert read(HERE/'unit_result.json')['passed']
    guarded('full',1800,'full.py')
    f=read(HERE/'full_result.json');assert f['passed'] and f['full_calls']==7 and f['synthetic_backwards']==5
    result.update(completed=True,passed=True,unit=pin(HERE/'unit_result.json'),full=pin(HERE/'full_result.json'))
except BaseException as error:
    result['error']=repr(error);result['completed']=not numerical_workers()
finally:
    result['wall_s']=time.monotonic()-started
    write(HERE/'controller_result.json',result);print(result,flush=True)
raise SystemExit(0 if result['completed'] else 1)
