import datetime,os,time
from common import *
admission();verify(data=True);assert not numerical_workers()
write(HERE/'run_claim.json',dict(pid=os.getpid(),pgid=os.getpgrp(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=pin(HERE/'sources_sealed.json'),maximum_full_network_calls=16,no_fit_or_upload=True))
started=time.monotonic();result=dict(completed=False,passed=False,error=None,no_retry=True,no_fit_or_upload=True)
try:
    guarded('native_disabled',480,'native.py','control','disabled')
    guarded('native_zero',480,'native.py','control','zero')
    guarded('native_zero_cold100',180,'native.py','control','zero','--rank',100)
    guarded('native_identity_sentinel',480,'native.py','control','identity_sentinel')
    guarded('native_basis_zero_cold35',180,'native.py','control','basis_zero','--rank',35)
    guarded('native_sentinel',480,'native.py','control','sentinel')
    guarded('native_physical_only_cold35',180,'native.py','control','physical_only','--rank',35)
    guarded('native_head_only_cold35',180,'native.py','control','head_only','--rank',35)
    guarded('qualify',180,'qualify.py')
    q=read(HERE/'qualification.json');assert q['passed'] and q['full_native_calls']==16
    result.update(completed=True,passed=True,qualification=pin(HERE/'qualification.json'))
except BaseException as error:
    result['error']=repr(error)
    result['completed']=not numerical_workers()
finally:
    result['wall_s']=time.monotonic()-started
    write(HERE/'controller_result.json',result);print(result,flush=True)
raise SystemExit(0 if result['completed'] else 1)
