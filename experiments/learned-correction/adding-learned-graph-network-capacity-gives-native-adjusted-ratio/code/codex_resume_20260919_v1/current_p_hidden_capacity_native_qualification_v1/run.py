import datetime
import time
from common import *

verify(data=True);activation();assert not numerical_workers()
write(HERE/'run_claim.json',dict(pid=os.getpid(),pgid=os.getpgrp(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    sources=pin(HERE/'sources_sealed.json'),maximum_full_native_calls=10,targets_permitted=False,no_fit_or_upload=True))
start=time.monotonic();result=dict(completed=False,passed=False,error=None,no_retry=True,no_fit_or_upload=True)
try:
    guarded('unit',180,'unit.py')
    guarded('native_disabled',480,'native.py','control','disabled')
    guarded('native_zero',480,'native.py','control','zero')
    guarded('native_zero_cold100',180,'native.py','control','zero','--rank',100)
    guarded('native_sentinel',480,'native.py','control','sentinel')
    guarded('qualify',180,'qualify.py')
    q=read(HERE/'qualification.json');assert q['passed']
    result.update(completed=True,passed=True,qualification=pin(HERE/'qualification.json'),full_native_calls=10,
        targets_decoded=False,model_fitted=False,native_derivative_certified=False,training_translation_qualified=False)
except BaseException as error:
    result['error']=repr(error)
finally:
    result['wall_s']=time.monotonic()-start;result['sources']=pin(HERE/'sources_sealed.json')
    write(HERE/'result.json',result);print(result,flush=True)
raise SystemExit(0 if result['passed'] else 1)
