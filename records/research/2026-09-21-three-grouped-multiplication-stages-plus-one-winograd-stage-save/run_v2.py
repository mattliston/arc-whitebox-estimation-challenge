from common import *
seal_check()
for p in read(HERE/'repair_sources_sealed.json')['pins']:verify(p)
guarded('audit_v2',240,'audit_v2.py')
guarded('replay',90,'replay.py')
write(HERE/'completed.json',dict(completed=True,audit=pin(HERE/'audit_result.json'),replay=pin(HERE/'replay_result.json'),guards=[pin(HERE/(n+'_watch.json')) for n in ['audit','audit_v2','replay']],opportunity_gate_passed=read(HERE/'audit_result.json')['opportunity_gate_passed'],new_model_calls=0,original_failure_preserved=True))
print(read(HERE/'completed.json'),flush=True)
