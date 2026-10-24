from common import *
seal_check();guarded('unit',120,'unit.py')
for rank in (35,100):guarded(f'rank{rank:04d}',300,'capture.py',rank)
guarded('analysis',90,'analyze.py')
write(HERE/'completed.json',dict(completed=True,guards=['unit','rank0035','rank0100','analysis'],
    full_native_calls=2,diagnostic_qualified=True,both_networks_flagged=read(HERE/'result.json')['both_networks_flagged']))
print(dict(completed=True,both_networks_flagged=read(HERE/'result.json')['both_networks_flagged']),flush=True)
