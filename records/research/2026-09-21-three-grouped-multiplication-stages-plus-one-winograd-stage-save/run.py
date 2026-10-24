from common import *
seal_check()
guarded('audit',240,'audit.py')
guarded('replay',90,'replay.py')
write(HERE/'completed.json',dict(completed=True,audit=pin(HERE/'audit_result.json'),replay=pin(HERE/'replay_result.json'),guards=[pin(HERE/(n+'_watch.json')) for n in ['audit','replay']],opportunity_gate_passed=read(HERE/'audit_result.json')['opportunity_gate_passed'],new_model_calls=0))
print(read(HERE/'completed.json'),flush=True)
