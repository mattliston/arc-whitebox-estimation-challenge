from common import HERE,RUNTIME,read,write,pin,verify,guarded,seal_check
seal_check('fit_sources_sealed_v2.json');assert read(HERE/'captures_sealed.json')['passed']
g=read(HERE/'capture_controller_watch.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
for p in read(HERE/'analysis_sources_sealed.json')['pins']:verify(p)
for name,seconds,script in [('unit',90,'unit.py'),('unit_v2',180,'unit_v2.py'),('design',480,'design.py'),('fit',180,'fit_v2.py')]:
    guarded(name,seconds,[RUNTIME,'-B',HERE/script])
if read(HERE/'cv_result.json')['passed']:guarded('check',60,[RUNTIME,'-B',HERE/'check.py'])
write(HERE/'analysis_completed.json',dict(completed=True,CV_passed=read(HERE/'cv_result.json')['passed'],CHECK_run=(HERE/'check_result.json').exists(),outcome=read(HERE/'check_outcome.json') if (HERE/'check_outcome.json').exists() else read(HERE/'outcome.json')))
print(read(HERE/'analysis_completed.json'),flush=True)
