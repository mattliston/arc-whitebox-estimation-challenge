from endpoint_common import *
endpoint_verify()
done('fit')
assert read(HERE / 'fit_complete.json')['completed']
for name, seconds, runtime, script, args in (
    ('build', 180, RUNTIME, 'build.py', []),
    ('native_null', 240, NATIVE_RUNTIME, 'native.py', ['null']),
    ('qualify_null', 180, NATIVE_RUNTIME, 'qualify_null.py', []),
    ('native_parent', 2400, NATIVE_RUNTIME, 'native.py', ['parent']),
    ('native_trained', 2400, NATIVE_RUNTIME, 'native.py', ['trained']),
    ('seal_predictions', 180, NATIVE_RUNTIME, 'seal_predictions.py', []),
    ('score_native40', 180, NATIVE_RUNTIME, 'score_native40.py', [])):
    guarded(name, seconds, runtime, script, *args)
write(HERE / 'endpoint_complete.json', dict(completed=True,
    passed=read(HERE / 'native40_score.json')['passed'], native_calls=85,
    score=pin(HERE / 'native40_score.json'), no_automatic_downstream_launch=True,
    official_changed=False))
