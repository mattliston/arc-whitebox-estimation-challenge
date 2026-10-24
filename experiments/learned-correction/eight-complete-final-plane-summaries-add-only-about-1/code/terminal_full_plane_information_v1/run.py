from common import *
seal_check()
for name,seconds,script in [('unit',90,'unit.py'),('features',120,'compute.py'),('oracle',60,'oracle.py'),('audit',60,'audit.py')]:
    guarded(name,seconds,script)
write(HERE/'completed.json',dict(completed=True,passed=read(HERE/'result.json')['passed'],audit_passed=read(HERE/'audit_result.json')['passed'],
    guards=[pin(HERE/f'{n}_watch.json') for n in ['unit','features','oracle','audit']]))
print(dict(completed=True,passed=read(HERE/'result.json')['passed']),flush=True)
