from common import *
seal_check();guarded('unit',90,'unit.py')
for rank in (35,100):guarded(f'capture{rank}',240,'capture.py',rank)
guarded('analyze',60,'analyze.py');guarded('audit',90,'audit.py')
write(HERE/'completed.json',dict(completed=True,guards=['unit','capture35','capture100','analyze','audit'],
    full_native_calls=2,both_pass_planning_gate=read(HERE/'result.json')['both_pass_planning_gate']))
print(dict(completed=True,both_pass_planning_gate=read(HERE/'result.json')['both_pass_planning_gate']),flush=True)
