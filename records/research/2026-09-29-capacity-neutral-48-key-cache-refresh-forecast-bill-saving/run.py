"""Root-only passive side-key diagnostic; all recorded timings are ineligible.

CLI: NATIVE -B run.py PROTOCOL_SHA
Run once under the protocol's external guard. No estimator source is edited.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys

D = Path(__file__).resolve().parent


def pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def checked(record):
    assert pin(record['path']) == record, record['path']


assert len(sys.argv) == 2
protocol_pin = pin(D / 'protocol.json')
assert protocol_pin['sha256'] == sys.argv[1]
protocol = json.loads((D / 'protocol.json').read_text())
for record in protocol['pins'].values():
    checked(record)
assert Path(sys.executable).absolute() == Path(protocol['runtime'])
assert protocol['ranks'] == [35, 100]
assert protocol['timing_qualification_eligible'] is False
target = D / 'key_census.jsonl'
assert not target.exists() and not (D / 'completed.json').exists()
baseline_rows = {row['rank']: row for row in
                 map(json.loads, Path(protocol['pins']['baseline_rows']['path']).read_text().splitlines())}
baseline_predictions = {int(k): v for k, v in protocol['baseline_predictions'].items()}
assert set(baseline_predictions) == {35, 100}
for record in baseline_predictions.values():
    checked(record)
source = Path(protocol['pins']['native_driver']['path']).read_text()

# All observer state contains strings/integers/booleans only. Cache-entry and
# frame references below remain call-local; dict.get bypasses _CapLRU.get.
_kd_counts = Counter()
_kd_keys, _kd_callers, _kd_events = {}, {}, []
_kd_rank = None
_kd_original = None
_kd_module = None


def _kd_cache_snapshot():
    return [[[int(x) for x in key] + [entry[3] is not None]
             for key, entry in dict.items(cache)]
            for cache in (_kd_module._IPT_A, _kd_module._IPT_B, _kd_module._IPT_C)]


def _kd_side(kind, level, rows, cols):
    key = (int(kind), int(level), int(rows), int(cols))
    cache = (_kd_module._IPT_A, _kd_module._IPT_B, _kd_module._IPT_C)[kind]
    entry = dict.get(cache, (level, rows, cols))
    hit = entry is not None
    before_fused = hit and entry[3] is not None
    del entry
    frame = sys._getframe(1)
    try:
        assert frame.f_code.co_filename == protocol['pins']['candidate']['path']
        caller = (frame.f_code.co_name, frame.f_code.co_firstlineno, frame.f_lineno)
    finally:
        del frame
    key_id = _kd_keys.setdefault(key, len(_kd_keys))
    caller_id = _kd_callers.setdefault(caller, len(_kd_callers))
    assert len(_kd_keys) <= 4096 and len(_kd_callers) <= 128
    result = _kd_original(kind, level, rows, cols)
    assert len(result) == 5
    fused = result[3] is not None
    _kd_counts[(key_id, caller_id, hit, before_fused, fused)] += 1
    if protocol['chronological_events']:
        assert len(_kd_events) < 30000
        _kd_events.append([key_id, caller_id, int(hit), int(before_fused), int(fused)])
    return result


install = '''
_kd_module = mod
_kd_original = mod._ipt_side
assert _kd_original.__code__.co_firstlineno == 6989
assert _kd_original.__code__.co_filename == str(candidate)
mod._ipt_side = _kd_side
'''
begin = '''        _kd_rank = rank
        _kd_counts.clear(); _kd_keys.clear(); _kd_callers.clear(); _kd_events.clear()
        _kd_initial = _kd_cache_snapshot()
'''
finish = '''        assert row['flops'] == baseline_rows[rank]['flops']
        assert pin(out/f'pred_{rank}.npy')['sha256'] == baseline_predictions[rank]['sha256']
        _kd_record = dict(rank=rank, candidate=protocol['pins']['candidate'],
            protocol=protocol_pin, driver=pin(__file__), native_driver=protocol['pins']['native_driver'],
            keys=[list(k) for k in _kd_keys], callers=[list(k) for k in _kd_callers],
            counts=[dict(key_id=k,caller_id=c,calls=n,hit=hit,before_fused=before,returned_fused=after)
                    for (k,c,hit,before,after),n in sorted(_kd_counts.items())],
            events=list(_kd_events), initial_cache=_kd_initial, final_cache=_kd_cache_snapshot(),
            flops=row['flops'], baseline_prediction=baseline_predictions[rank],
            prediction=pin(out/f'pred_{rank}.npy'), all16_saved_file_bytes_exact=True,
            full_bill_exact=True, timing_qualification_eligible=False)
        _kd_fp.write(json.dumps(_kd_record,allow_nan=False)+'\\n'); _kd_fp.flush()
'''
for marker, replacement in (
    ('record=dict(candidate=', install + '\nrecord=dict(candidate='),
    ("(out/'manifest.json').write_text", "record['timing_qualification_eligible'] = False\n(out/'manifest.json').write_text"),
    ('        weights=np.load(', begin + '        weights=np.load('),
    ('row=dict(rank=rank,', 'row=dict(timing_qualification_eligible=False,rank=rank,'),
    ('        del weights,y,truth\n', finish + '        del weights,y,truth\n'),
):
    assert source.count(marker) == 1, marker
    source = source.replace(marker, replacement)
original_argv = sys.argv
sys.argv = [protocol['pins']['native_driver']['path'],
            protocol['pins']['candidate']['path'], '35,100', protocol['tag']]
try:
    with target.open('x') as _kd_fp:
        exec(compile(source, protocol['pins']['native_driver']['path'], 'exec'), globals())
finally:
    if _kd_original is not None:
        _kd_module._ipt_side = _kd_original
    sys.argv = original_argv
for record in protocol['pins'].values():
    checked(record)
for record in baseline_predictions.values():
    checked(record)
checked(protocol_pin)
with (D / 'completed.json').open('x') as stream:
    json.dump(dict(passed=True, protocol=protocol_pin, driver=pin(__file__),
        census=pin(target), ranks=protocol['ranks'], all16_bytes_and_full_bill_exact=True,
        observer_restored=True, timing_qualification_eligible=False,
        external_guard_required=True, candidate_changed=False), stream, indent=2)
    stream.write('\n')
