import runpy
from endpoint_common import *
endpoint_verify()
pins = []
for role in ['null','parent','trained']:
    done(f'qualify_{role}_lifecycle')
    path = HERE/f'{role}_lifecycle_qualified.json'
    q = read(path)
    assert q['passed'] and q['role'] == role and q['closed_contexts_released'] and q['saved_bills_unchanged']
    assert not q['truth_accessed'] and q['calls'] == (3 if role == 'null' else 41)
    for p in q['lifecycle_pins']+q['original_comparison_pins']+[q['records'],q['watch']]:
        check(p);pins.append(p)
    pins.extend([pin(path),pin(HERE/f'qualify_{role}_lifecycle_watch.json')])
write(HERE/'lifecycle_receipts_sealed.json',dict(completed=True,before_validation_targets=True,
    pins=list({p['path']:p for p in pins}.values()),all_85_closed_context_calls_qualified=True))
runpy.run_path(str(HERE/'seal_predictions.py'),run_name='__main__')
