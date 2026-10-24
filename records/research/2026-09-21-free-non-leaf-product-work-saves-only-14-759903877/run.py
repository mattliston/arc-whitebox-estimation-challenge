from common import *
seal_check();guarded('analyze','analyze.py');guarded('audit','audit.py')
write(HERE/'completed.json',dict(completed=True,guards=['analyze','audit'],new_native_calls=0,
    both_pass_planning_gate=read(HERE/'result.json')['both_pass_planning_gate']))
print(dict(completed=True,both_pass_planning_gate=read(HERE/'result.json')['both_pass_planning_gate']),flush=True)
