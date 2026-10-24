from common import *
seal_check();guarded('unit_v2',120,'unit_v2.py')
for rank in (35,100):guarded(f'rank{rank:04d}',300,'capture_v2.py',rank)
guarded('analysis',90,'analyze_v2.py')
write(HERE/'completed.json',dict(completed=True,guards=['unit_v2','rank0035','rank0100','analysis'],
    preserved_failed_guards=['unit'],full_native_calls=2,diagnostic_qualified=True,both_networks_flagged=read(HERE/'result.json')['both_networks_flagged']))
print(dict(completed=True,both_networks_flagged=read(HERE/'result.json')['both_networks_flagged']),flush=True)
