import datetime
import time
from common import *
verify();assert not numerical_workers()
write(HERE/'run_claim.json',dict(pid=os.getpid(),pgid=os.getpgrp(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_seal=pin(HERE/'sources_sealed.json'),native_qualification=pin(N/'qualification.json'),no_fit_or_upload=True))
start=time.monotonic();result=dict(completed=False,passed=False,error=None,no_retry=True)
try:
    guarded('unit',180,'unit.py')
    guarded('full',1200,'full.py')
    u=read(HERE/'unit_result.json');f=read(HERE/'full_result.json')
    assert u['passed'] and f['passed']
    result.update(completed=True,passed=True,unit=pin(HERE/'unit_result.json'),full=pin(HERE/'full_result.json'),
        targets_decoded=False,model_fitted=False,production_candidate_exported=False,native_derivative_certified=False,
        hidden_trainable_values=12352,initial_outgoing_zero=True,initial_incoming_copied=True)
except BaseException as error:
    result['error']=repr(error)
finally:
    result['wall_s']=time.monotonic()-start;result['sources']=pin(HERE/'sources_sealed.json')
    write(HERE/'result.json',result);print(result,flush=True)
raise SystemExit(0 if result['passed'] else 1)
