from datetime import datetime,timezone
import os
import time
from common import *

fit_success()
assert not (HERE/'run_claim.json').exists() and not (HERE/'controller_result.json').exists()
assert not numerical_workers()
write(HERE/'run_claim.json',dict(pid=os.getpid(),pgid=os.getpgrp(),utc=datetime.now(timezone.utc).isoformat(),
    sources=pin(HERE/'sources_sealed.json'),fit=pin(F/'fit_complete.json'),maximum_native_calls=86,
    no_other_numerical_worker=True,no_automatic_downstream=True,no_upload=True))
start = time.monotonic()
result = dict(completed=False,scientific_passed=None,error=None,no_retry=True,no_external_actions=True,
    no_automatic_downstream=True,fit_changed=False)
try:
    guarded('build',180,'build.py')
    guarded('native_null',480,'native.py','control','null')
    guarded('native_null_cold100',240,'native.py','control','null','--rank','100')
    guarded('qualify_null',180,'qualify_null.py')
    assert read(HERE/'native_null_qualification.json')['passed']
    guarded('native_parent',2400,'native.py','validation','parent')
    guarded('native_trained',2400,'native.py','validation','trained')
    guarded('seal_predictions',180,'seal_predictions.py')
    guarded('score_native40',180,'score_native40.py')
    score = read(HERE/'native40_score.json')
    assert score['completed'] and score['integrity_passed']
    result.update(completed=True,scientific_passed=score['passed'],score=pin(HERE/'native40_score.json'),
        native_calls=86,validation_targets_accessed=True)
except BaseException as error:
    result['error']=repr(error)
finally:
    result.update(wall_s=time.monotonic()-start,sources=pin(HERE/'sources_sealed.json'))
    write(HERE/'controller_result.json',result)
    print(result,flush=True)
raise SystemExit(0 if result['completed'] else 1)
