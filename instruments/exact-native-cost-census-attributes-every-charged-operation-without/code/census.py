"""One passive S32 rank35 paid-bill census, adapted from bill_census_layers_e2.

Run only under the root's native guard. No target data or intermediate values
are read. Setup is untouched and finishes before the accounting hook exists.
"""
from pathlib import Path
import collections
import hashlib
import json
import os
import sys
import time
import types
import numpy as np
import flopscope as flops
from flopscope import _budget

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
P2 = H2.parent
CANDIDATE = H2/'hill_climb_star_projection32_v1/candidate.py'
CONTROL = H2/'hill_climb_star_projection32_train2_v1'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
ADAPTED = P2/'fable_engineer2_v1/bill27/bill_census_layers_e2.py'
RANK = 35
EXPECTED_BILL = 276549645399
PINS = [
    (CANDIDATE, 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'),
    (STORE/'manifest.json', '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50'),
    (STORE/'rows/observer_fit/0035/row_manifest.json', '5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809'),
    (STORE/'rows/observer_fit/0035/weights.npy', '14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2'),
    (CONTROL/'manifest.json', '4327a61916482e1b699de367ca8c9a79ca9ff25c7467f1e37f4b54ce68285324'),
    (CONTROL/'rows.jsonl', '9dfa3606493dd151cc3c2086dd6b35012912b4a9db2eb043ce7847e8be62e510'),
    (CONTROL/'pred_35.npy', '478936da129a74f5880e5e1d8a3b56128a6f8c42d669fb9938324f9849188971'),
    (ADAPTED, '0882e90b9158c9321701c9192a91f8049d684c5fd50132917dd6cafb474545a7'),
]


def pin(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def main():
    out = HERE/'census.json'
    assert not out.exists(), 'Exclusive census output already exists'
    assert np.__version__ == '2.4.6'
    native = P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1'
    assert Path(sys.executable).parent == native/'bin'
    assert str(Path(flops.__file__).resolve()).startswith(str(native.resolve())+'/')
    threads = {n: os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}
    assert all(value == '1' for value in threads.values())
    sources = []
    for path, digest in PINS:
        record = pin(path); assert record['sha256'] == digest, str(path); sources.append(record)
    manifest = json.loads((STORE/'manifest.json').read_text())
    row = manifest['roles']['observer_fit']['row_manifests'][RANK]
    assert row['role_rank'] == RANK and row['path'] == 'rows/observer_fit/0035/row_manifest.json'
    rm = json.loads((STORE/row['path']).read_text())
    assert rm['files']['weights.npy']['sha256'] == PINS[3][1]
    cm = json.loads((CONTROL/'manifest.json').read_text())
    assert cm['candidate'] == str(CANDIDATE) and cm['sha256'] == PINS[0][1] and cm['ranks'] == [35,100]
    controls = [json.loads(line) for line in (CONTROL/'rows.jsonl').read_text().splitlines()]
    assert [r['rank'] for r in controls] == [35,100] and controls[0]['flops'] == EXPECTED_BILL
    stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    flops.configure(symmetry_warnings=False)
    mod = types.ModuleType('s32_bill_census'); mod.__file__ = str(CANDIDATE); sys.modules[mod.__name__] = mod
    source = CANDIDATE.read_text()
    exec(compile(source, str(CANDIDATE), 'exec', dont_inherit=True), mod.__dict__)
    live = mod._predict_binary
    while hasattr(live, '__wrapped__'): live = live.__wrapped__
    live_code = live.__code__  # Identification only; prediction calls the original public estimator.
    est = mod.Estimator(); started = time.monotonic()
    est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41,
        api_version='2', scratch_dir=None, submission_dir=None, seed=0))
    setup = dict(seconds=time.monotonic()-started, pb_log=[list(x) for x in mod._PB_LOG])
    assert [r[1] for r in setup['pb_log'] if r[0]=='bound'] == [624]
    assert [r[1] for r in setup['pb_log'] if r[0]=='lvfuse'] == [600]
    print(json.dumps(dict(event='untouched_setup', **setup)), flush=True)
    weights = np.load(STORE/Path(row['path']).parent/'weights.npy', allow_pickle=False)
    assert weights.shape == (16,1024,1024) and weights.dtype == np.dtype('float32')
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)],
                                seed=0, name='phase2-35')
    cost = collections.Counter(); counts = collections.Counter(); snapshots = {}
    original = _budget.BudgetContext._charge_op

    def charged(self, op_name, flop_cost, subscripts, shapes, **kwargs):
        before = self._flops_used
        result = original(self, op_name, flop_cost, subscripts, shapes, **kwargs)
        paid = self._flops_used-before
        shape_key = tuple(tuple(int(d) for d in sh) for sh in shapes) if op_name in ('matmul','dot','einsum') else ()
        dtype_key = tuple(str(d) for d in (kwargs.get('dtypes') or ()))
        frame = sys._getframe(1); chain = []; owner = None
        while frame is not None:
            if frame.f_code is live_code: owner = frame; break
            if frame.f_code.co_filename == str(CANDIDATE) and frame.f_code.co_name not in ('w','wrapped'):
                chain.append((frame.f_code.co_name, frame.f_lineno))
            frame = frame.f_back
        layer, line = (-1,0) if owner is None else (owner.f_locals.get('l'),owner.f_lineno)
        outer = tuple(reversed(chain[-4:])); inner = chain[0] if chain else ('',0)
        key = (layer,line,outer,inner,op_name,shape_key,dtype_key,subscripts)
        cost[key] += paid; counts[key] += 1
        if owner is not None and layer not in snapshots:
            loc = owner.f_locals; old = loc.get('shared_old') or {}
            lam, li = loc.get('lam'), loc.get('live_indices')
            snapshots[layer] = dict(layer=layer, cols=None if lam is None else int(lam.shape[0]),
                old_cols=old.get('columns'), live=None if li is None else int(li.shape[0]),
                **{name: None if old.get(name) is None else list(old[name].shape)
                   for name in ('Q','F','core','pre_basis')})
        return result

    try:
        _budget.BudgetContext._charge_op = charged
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            prediction = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    finally:
        _budget.BudgetContext._charge_op = original
    bill = int(ctx.summary_dict()['flops_used'])
    reference = np.load(CONTROL/'pred_35.npy', allow_pickle=False)
    assert prediction.shape == reference.shape == (16,1024)
    assert prediction.dtype == reference.dtype == np.dtype('float64')
    assert np.isfinite(prediction).all() and prediction.tobytes() == reference.tobytes()
    assert bill == EXPECTED_BILL and sum(cost.values()) == bill
    assert _budget.BudgetContext._charge_op is original
    for record in sources: assert pin(Path(record['path'])) == record
    rows = [dict(layer=k[0], line=k[1], outer=k[2], inner=k[3], op=k[4], shapes=k[5],
                 dtypes=k[6], subscripts=k[7], flops=value, ops=counts[k]) for k,value in cost.items()]
    record = dict(passed=True, candidate=sources[0], rank=RANK, bill=bill, attributed=sum(cost.values()),
        all16_bytes_exact=True, full_bill_exact=True, hook_restored=True, setup=setup,
        runtime=sys.executable, numpy=np.__version__, thread_env=threads,
        prediction_bytes_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),
        rows=rows, snapshots=list(snapshots.values()), targets_accessed=False,
        sources=sources, census_source=pin(Path(__file__).resolve()),
        note='Paid-operation attribution only; profiling elapsed time is not a performance measurement.')
    with out.open('x') as stream: json.dump(record, stream, indent=1, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(passed=True, bill=bill, attributed=sum(cost.values()), keys=len(rows),
                         all16_bytes_exact=True, hook_restored=True)), flush=True)


if __name__ == '__main__':
    main()
