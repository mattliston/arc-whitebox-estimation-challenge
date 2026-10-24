from fit_common import *
verify()
guarded('fit', 14400, RUNTIME, 'train.py')
assert read(HERE / 'fit_result.json')['completed']
write(HERE / 'fit_complete.json', dict(completed=True, result=pin(HERE / 'fit_result.json'),
    watch=pin(HERE / 'fit_watch.json'), validation_targets_accessed=False, native_score_measured=False))
