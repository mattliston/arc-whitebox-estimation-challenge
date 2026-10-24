from common import *
verify();verify('runtime_repair_sealed.json')
guarded('unit_v2',180,'unit_v2.py')
for ordinal in TRAIN+HELDOUT:
    guarded(f'weights_mini{ordinal:04d}',600,'weights_dispatch.py',ordinal)
    guarded(f'capture_mini{ordinal:04d}',900,'capture_v2.py',ordinal)
write(HERE/'inputs_completed.json',dict(completed=True,target_values_decoded=0,
      unit=pin(HERE/'unit_result.json'),captures=[pin(HERE/f'capture_mini{o:04d}.json') for o in TRAIN+HELDOUT],
      full_native_parent_calls=10,original_import_failure_preserved=True))
