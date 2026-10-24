from datetime import datetime,timezone
import os
import time
from iso_common import *

verify()
assert not (HERE/'run_claim.json').exists() and not (HERE/'controller_result.json').exists()
assert read(HERE/'admission.json')['passed'] and not numerical_workers()
write(HERE/'run_claim.json',dict(pid=os.getpid(),pgid=os.getpgrp(),utc=datetime.now(timezone.utc).isoformat(),
    sources=pin(HERE/'sources_sealed.json'),maximum_backward_graphs=160,maximum_optimizer_updates=40,
    whole_controller_ceiling_s=14400,no_native_launch=True,no_external_actions=True))
began=time.monotonic()
result=dict(completed=False,passed=False,error=None,no_retry=True,no_native_launch=True,no_external_actions=True)
try:
    for update in range(40):
        remaining=14400-(time.monotonic()-began)
        assert remaining>0
        args=[update]
        if update:args+=['--resume',HERE/f'batch_{update-1:02d}'/f'checkpoint_{update:02d}.json']
        guarded(f'batch_{update:02d}',min(480,remaining),'worker.py',*args)
        worker=read(HERE/f'batch_{update:02d}/result.json')
        assert worker['completed'] and worker['passed'] and worker['training_graphs']==4 and worker['updates']==1
        assert worker['start_update']==update and worker['stop_update']==update+1
        for entry in worker['checkpoints']:check(entry);check(read(entry['path'])['payload'])
    remaining=14400-(time.monotonic()-began)
    assert remaining>0
    guarded('finalize',min(180,remaining),'finalize.py')
    fit=read(HERE/'fit_result.json')
    assert fit['completed'] and fit['passed'] and fit['training_graphs']==160 and fit['updates']==40
    assert time.monotonic()-began<14400
    watches=[clean_watch(f'batch_{i:02d}') for i in range(40)]+[clean_watch('finalize')]
    write(HERE/'fit_resource_summary.json',dict(completed=True,passed=True,
        kind='aggregate_of_40_separate_batch_guards_and_finalizer',guard_count=41,
        guards=[pin(HERE/f'batch_{i:02d}_watch.json') for i in range(40)]+[pin(HERE/'finalize_watch.json')],
        max_rss_bytes=max(w['max_rss_bytes'] for w in watches),max_swap_growth_bytes=max(w['max_swap_growth_bytes'] for w in watches),
        every_group_retired=True,remaining_rows=[],elapsed_s=time.monotonic()-began,
        max_retired_graph_rss_bytes=fit['max_retired_graph_rss_bytes']))
    write(HERE/'fit_complete.json',dict(completed=True,result=pin(HERE/'fit_result.json'),
        resource_summary=pin(HERE/'fit_resource_summary.json'),sources=pin(HERE/'sources_sealed.json')))
    result.update(completed=True,passed=True,fit_complete=pin(HERE/'fit_complete.json'))
except BaseException as error:
    result['error']=repr(error)
finally:
    result.update(wall_s=time.monotonic()-began,sources=pin(HERE/'sources_sealed.json'))
    write(HERE/'controller_result.json',result)
    print(result,flush=True)
raise SystemExit(0 if result['passed'] else 1)
