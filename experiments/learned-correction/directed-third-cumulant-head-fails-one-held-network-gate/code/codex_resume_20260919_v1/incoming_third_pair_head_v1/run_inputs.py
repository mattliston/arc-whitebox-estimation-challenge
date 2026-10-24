from common import *
verify()
guarded('unit',180,'unit.py')
for ordinal in TRAIN+HELDOUT:
    guarded(f'capture_mini{ordinal:04d}',900,'capture.py',ordinal)
write(HERE/'inputs_completed.json',dict(completed=True,target_values_decoded=0,
      unit=pin(HERE/'unit_result.json'),captures=[pin(HERE/f'capture_mini{o:04d}.json') for o in TRAIN+HELDOUT],
      full_native_parent_calls=10))
