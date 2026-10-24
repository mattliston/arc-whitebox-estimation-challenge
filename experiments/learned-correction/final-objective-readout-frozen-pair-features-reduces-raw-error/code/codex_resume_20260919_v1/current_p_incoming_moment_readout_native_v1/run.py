import datetime
import os
import time
from common import *
fit_success();verify(data=True);assert not numerical_workers()
write(HERE/'run_claim.json',dict(pid=os.getpid(),pgid=os.getpgrp(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    sources=pin(HERE/'sources_sealed.json'),fit=pin(F/'fit_result.json'),maximum_new_native_calls=87,
    matched_cold_comparison=True,no_automatic_downstream=True,no_upload=True))
start=time.monotonic();result=dict(completed=False,scientific_passed=None,error=None,no_retry=True,no_external_actions=True,no_automatic_downstream=True)
try:
    guarded('build',180,'build.py')
    guarded('native_null',480,'native.py','control','null')
    guarded('native_null_cold100',180,'native.py','control','null','--rank',100)
    guarded('qualify_null',180,'qualify_null.py')
    guarded('native_trained_control',480,'native.py','control','trained')
    guarded('qualify_controls',180,'qualify_controls.py')
    for rank in read(HERE/'inputs.json')['valid']:
        for role in ('parent','trained'):guarded(f'{role}_{rank}',180,'native.py','validation',role,'--rank',rank)
    guarded('seal_predictions',180,'seal_predictions.py')
    guarded('score_native40',180,'score_native40.py')
    score=read(HERE/'native40_score.json');assert score['completed'] and score['integrity_passed']
    result.update(completed=True,scientific_passed=score['passed'],score=pin(HERE/'native40_score.json'),
        new_native_calls=87,validation_targets_accessed=True)
except BaseException as error:
    result['error']=repr(error)
finally:
    result['wall_s']=time.monotonic()-start;result['sources']=pin(HERE/'sources_sealed.json')
    write(HERE/'controller_result.json',result);print(result,flush=True)
raise SystemExit(0 if result['completed'] else 1)
