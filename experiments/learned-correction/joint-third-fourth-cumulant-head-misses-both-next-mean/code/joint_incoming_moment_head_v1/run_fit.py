from common import *
verify();verify('fit_sources_sealed.json');assert read(HERE/'unit_result.json')['passed']
for ordinal in TRAIN:guarded(f'targets_mini{ordinal:04d}',600,'targets.py',ordinal)
write(HERE/'train_targets_completed.json',dict(completed=True,targets=[pin(HERE/f'targets_mini{o:04d}.json') for o in TRAIN]))
guarded('train',5400,'train.py')
for ordinal in VALID:guarded(f'targets_mini{ordinal:04d}',600,'targets.py',ordinal)
guarded('judge',600,'judge.py')
write(HERE/'fit_completed.json',dict(completed=True,information=pin(HERE/'information_result.json'),information_passed=read(HERE/'information_result.json')['passed'],model_frozen=pin(HERE/'model_frozen.json')))
