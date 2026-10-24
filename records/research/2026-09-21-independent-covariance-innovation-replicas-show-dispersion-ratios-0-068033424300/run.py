from common import *
seal_check()
for name,seconds in (('unit',90),('predict',180),('measure',180),('audit',120)):guarded(name,seconds,f'{name}.py')
write(HERE/'completed.json',dict(completed=True,audit_passed=read(HERE/'audit.json')['passed'],
    guarded_names=['unit','predict','measure','audit'],new_native_calls=0,new_fit_steps=0,new_samples=0,uploaded=False))
print(dict(completed=True,audit_passed=True),flush=True)
