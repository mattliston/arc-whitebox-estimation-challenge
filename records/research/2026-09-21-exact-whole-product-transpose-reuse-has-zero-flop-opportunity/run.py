from common import *
seal_check();guarded('unit',90,'unit.py')
for rank in (35,100):guarded(f'census{rank}',180,'capture.py',rank)
write(HERE/'traces_sealed.json',dict(pins=[pin(HERE/f'rank{r:04d}/{name}') for r in (35,100) for name in ('trace.json','result.json','prediction.npy')],before_grouping=True,targets_read=False))
guarded('analyze',90,'analyze.py')
write(HERE/'completed.json',dict(completed=True,implementation_gate_passed=read(HERE/'result.json')['implementation_gate_passed'],
    guards=['unit','census35','census100','analyze'],full_native_calls=2,targets_read=False))
print(dict(completed=True,implementation_gate_passed=read(HERE/'result.json')['implementation_gate_passed']),flush=True)
