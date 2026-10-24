"""Read-only input-transform fingerprints; no estimator arithmetic changes."""
from pathlib import Path
import ctypes
import gc
import hashlib
import json
import resource
import sys
import time
import types
import numpy as np
import flopscope as flops

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
P2 = H2.parent
PARENT = H2 / 'candidate_old_both_tail_trace_bounded_v6.py'
EXPECTED = '5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'


def sha(path):
    d = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            d.update(chunk)
    return d.hexdigest()


def fingerprint(value):
    a = np.asarray(value)
    d = hashlib.sha256()
    iterator = np.nditer(a, flags=['external_loop', 'buffered', 'zerosize_ok'],
                        op_flags=['readonly'], order='C', buffersize=262144)
    for block in iterator:
        d.update(memoryview(np.ascontiguousarray(block)).cast('B'))
    return dict(shape=list(a.shape), dtype=a.dtype.str,
                bytes=int(a.nbytes), sha256=d.hexdigest())


def hash_unit():
    base = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
    cases = [base, base.transpose(2, 0, 1), base[:, ::2, ::-2],
             base[::-1], base[:0], np.array([0., -0., 1.], dtype=np.float64),
             np.arange(400001, dtype=np.float32).reshape(1, -1)[:, ::-1]]
    for a in cases:
        old = a.tobytes(order='C')
        result = fingerprint(a)
        assert result['sha256'] == hashlib.sha256(old).hexdigest()
        assert a.tobytes(order='C') == old
        assert result['bytes'] == len(old)
    changed = base.copy()
    before = fingerprint(changed)
    changed[0, 0, 0] = 1
    assert fingerprint(changed) != before
    assert fingerprint(np.array([0.], np.float32)) != fingerprint(np.array([-0.], np.float32))
    return dict(passed=True, fixture_count=len(cases), input_preservation=True,
                changed_value_detected=True, signed_zero_preserved=True)


assert sha(PARENT) == EXPECTED
assert not (HERE / 'result.json').exists()
unit = hash_unit()
with (HERE / 'hash_qualification.json').open('x') as f:
    json.dump(unit, f, indent=2)
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('factor_transform_census_bank')
mod.__file__ = str(PARENT)
sys.modules[mod.__name__] = mod
exec(compile(PARENT.read_text(), str(PARENT), 'exec'), mod.__dict__)
originals = {name: getattr(mod, name) for name in ['_ipt_forward_a_ab', '_ipt_forward_b_ab']}
records = []
current = {}


def observer(side, original):
    def call(a, L, first, second):
        if not current:
            return original(a, L, first, second)
        start = time.monotonic()
        inp = fingerprint(a)
        frame = sys._getframe(1)
        callers = []
        while frame is not None:
            if frame.f_code.co_filename == str(PARENT):
                callers.append(frame.f_code.co_name)
            frame = frame.f_back
        before = int(current['ctx'].flops_used)
        value = original(a, L, first, second)
        cost = int(current['ctx'].flops_used) - before
        out = fingerprint(value)
        record = dict(rank=current['rank'], index=len(records), side=side,
                      depth=int(L), dims=[int(first), int(second)],
                      input=inp, output=out, flops=cost, callers=callers,
                      diagnostic_elapsed_s=time.monotonic() - start)
        records.append(record)
        return value
    return call


for name, original in originals.items():
    setattr(mod, name, observer('A' if name.endswith('a_ab') else 'B', original))
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41,
          api_version='2', scratch_dir=None, submission_dir=None, seed=0))
store = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
weight_rows = json.loads((store / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
controls = {r['rank']: r for r in map(json.loads, (H2 / 'both_tail_train2_v6/rows.jsonl').read_text().splitlines())}
relief = ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
relief.restype = ctypes.c_size_t
rows = []
begin = time.monotonic()
try:
    for rank in [35, 100]:
        wp = store / Path(weight_rows[rank]['path']).parent / 'weights.npy'
        weights = np.load(wp)
        mlp = types.SimpleNamespace(width=1024, depth=16,
              weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
        offset = len(records)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            current.update(rank=rank, ctx=ctx)
            y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
            current.clear()
        bill = int(ctx.summary_dict()['flops_used'])
        base = np.load(H2 / 'both_tail_train2_v6' / f'pred_{rank}.npy')
        exact = bool(y.shape == base.shape and y.dtype == base.dtype
                     and y.tobytes() == base.tobytes())
        assert exact and bill == controls[rank]['flops'], (rank, exact, bill, controls[rank]['flops'])
        with (HERE / f'pred_{rank}.npy').open('xb') as f:
            np.save(f, y)
        with (HERE / f'transforms_{rank}.json').open('x') as f:
            json.dump(records[offset:], f, indent=2)
        row = dict(rank=rank, flops=bill, all16_exact=True, full_bill_exact=True,
                   transforms=len(records) - offset, weights_sha256=sha(wp),
                   targets_accessed=False, elapsed_s=time.monotonic() - begin)
        rows.append(row)
        print(json.dumps(dict(event='capture', **row)), flush=True)
        del weights, mlp, y, base
        gc.collect()
        relief(None, 0)
finally:
    current.clear()
    for name, original in originals.items():
        setattr(mod, name, original)

summaries = []
for row in rows:
    seen = {}
    duplicates = []
    total = 0
    for r in (r for r in records if r['rank'] == row['rank']):
        key = json.dumps([r['side'], r['depth'], r['dims'], r['input']], sort_keys=True)
        total += r['flops']
        if key in seen:
            earlier = seen[key]
            assert r['output'] == earlier['output'], (r['index'], earlier['index'])
            duplicates.append(dict(index=r['index'], first_index=earlier['index'],
                side=r['side'], depth=r['depth'], dims=r['dims'], saved_ceiling=r['flops'],
                output_bytes=r['output']['bytes'], callers=r['callers']))
        else:
            seen[key] = r
    saving = sum(r['saved_ceiling'] for r in duplicates)
    summaries.append(dict(**row, measured_transform_flops=total,
                         distinct_keys=len(seen), duplicate_count=len(duplicates),
                         optimistic_removable_flops=saving,
                         optimistic_saving_pct=100 * saving / row['flops'],
                         duplicate_records=duplicates))
result = dict(parent_sha256=EXPECTED, protocol_sha256=sha(HERE / 'PROTOCOL.md'),
              census_sha256=sha(Path(__file__)), hash_qualification=unit,
              networks=summaries, targets_accessed=False, valid40_accessed=False,
              fits_run=False, candidate_built=False,
              passes_ceiling_gate=all(r['optimistic_saving_pct'] >= 1.5 for r in summaries),
              elapsed_s=time.monotonic() - begin,
              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              scope='Only identical input transforms in the two observed global bindings; free unlimited cache ceiling')
with (HERE / 'result.json').open('x') as f:
    json.dump(result, f, indent=2)
assert sum(p.stat().st_size for p in HERE.glob('*') if p.is_file()) < 5_000_000
print(json.dumps(dict(event='complete', ceiling_pct=[r['optimistic_saving_pct'] for r in summaries],
                     passes=result['passes_ceiling_gate'], elapsed_s=result['elapsed_s'],
                     peak_rss_bytes=result['peak_rss_bytes'])), flush=True)
