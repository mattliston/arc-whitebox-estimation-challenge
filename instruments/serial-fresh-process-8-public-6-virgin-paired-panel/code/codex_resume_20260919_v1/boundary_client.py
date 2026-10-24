"""NumPy-free production-client qualification; never imports truth labels."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _connection

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'


def main():
    candidate, out, ranks = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
    out.mkdir(exist_ok=False)
    assert importlib.util.find_spec('numpy') is None
    stub = types.ModuleType('whestbench')
    stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    spec = importlib.util.spec_from_file_location('boundary_candidate', candidate)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    counts = collections.Counter()
    active = False
    send = _connection.Connection.send_recv

    def counted(self, request):
        if active:
            frame = sys._getframe(1)
            while frame is not None and frame.f_code.co_filename != str(candidate):
                frame = frame.f_back
            counts[frame.f_code.co_name if frame is not None else 'boundary'] += 1
        return send(self, request)

    _connection.Connection.send_recv = counted
    manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    records = []
    for i, rank in enumerate(map(int, ranks.split(','))):
        weights_path = (STORE/Path(manifest[rank]['path']).parent/'weights.npy'
                        if rank < 3016500 else P2/'public_g19_attribution_v1'/f'fresh_{rank}_weights.npy')
        with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f'ingress-{i}'):
            blob = fnp.load(str(weights_path))
            weights = [blob[l] for l in range(16)]
        mlp = types.SimpleNamespace(width=1024, depth=16, weights=weights, seed=0, name=f'rank{rank}')
        counts.clear()
        began = time.monotonic()
        with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f'predict-{i}') as ctx:
            active = True
            if i == 0:
                spec.loader.exec_module(mod)
                estimator = mod.Estimator()
            prediction = estimator.predict(mlp, 2**41)
            active = False
        elapsed = time.monotonic()-began
        summary = ctx.summary_dict()
        with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f'save-{i}'):
            fnp.save(str(out/f'call_{i:02d}_rank{rank:04d}.npy'), prediction)
        rec = dict(index=i, rank=rank, wall_s=elapsed, requests=sum(counts.values()),
                   by_function=dict(counts), summary=summary)
        records.append(rec)
        with (out/f'call_{i:02d}.json').open('x') as f:
            json.dump(rec, f, indent=2)
        print(json.dumps({k: v for k, v in rec.items() if k != 'by_function'}), flush=True)
        del prediction, mlp, weights, blob
    with (out/'completed.json').open('x') as f:
        json.dump(dict(completed=True, candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                       records=records, truth_accessed=False), f, indent=2)


if __name__ == '__main__':
    main()
