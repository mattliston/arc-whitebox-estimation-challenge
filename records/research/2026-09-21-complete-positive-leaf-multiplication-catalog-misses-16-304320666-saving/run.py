from common import *
seal_check()
guarded('reprice',900,'reprice.py')
guarded('verify_paths',90,'verify_paths.py')
guarded('audit',180,'audit.py')
write(HERE/'completed.json',dict(completed=True,audit_passed=read(HERE/'audit_result.json')['passed'],
    guards=[pin(HERE/f'{n}_watch.json') for n in ['reprice','verify_paths','audit']]))
print(dict(completed=True,audit_passed=True),flush=True)
