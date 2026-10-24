import time
from common import *
verify();write(HERE/'run_claim.json',dict(started=True,source_seal=pin(HERE/'sources_sealed.json')))
start=time.monotonic();result=dict(completed=False,passed=False,error=None)
try:
    guarded('unit',180,'unit.py')
    guarded('full',1200,'full.py')
    u=read(HERE/'unit_result.json');f=read(HERE/'full_result.json')
    assert u['passed'] and f['passed']
    result.update(completed=True,passed=True,unit=pin(HERE/'unit_result.json'),full=pin(HERE/'full_result.json'),
        targets_decoded=False,model_fitted=False,production_candidate_exported=False,native_derivative_certified=False)
except BaseException as e:
    result['error']=repr(e);raise
finally:
    result['wall_s']=time.monotonic()-start
    write(HERE/'result.json',result);print(result,flush=True)
