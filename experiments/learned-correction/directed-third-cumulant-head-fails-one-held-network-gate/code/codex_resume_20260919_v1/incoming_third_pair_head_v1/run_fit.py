from common import *
verify();verify('fit_sources_sealed.json')
for ordinal in TRAIN+HELDOUT:guarded(f'freeze_mini{ordinal:04d}',600,'freeze_inputs.py',ordinal)
write(HERE/'frozen_inputs_completed.json',dict(completed=True,target_values_decoded=0,
      inputs=[pin(HERE/f'inputs_mini{o:04d}.json') for o in TRAIN+HELDOUT]))
for ordinal in TRAIN:
    guarded(f'download_mini{ordinal:04d}',300,'download.py',ordinal)
    guarded(f'qualify_mini{ordinal:04d}',600,'qualify_targets.py',ordinal)
write(HERE/'train_targets_completed.json',dict(completed=True,targets=[pin(HERE/f'targets_mini{o:04d}.json') for o in TRAIN]))
guarded('train',5400,'train.py')
for ordinal in HELDOUT:
    guarded(f'download_mini{ordinal:04d}',300,'download.py',ordinal)
    guarded(f'qualify_mini{ordinal:04d}',600,'qualify_targets.py',ordinal)
guarded('judge',600,'judge.py')
write(HERE/'fit_completed.json',dict(completed=True,information=pin(HERE/'information_result.json'),
      information_passed=read(HERE/'information_result.json')['passed'],model_frozen=pin(HERE/'model_frozen.json')))
