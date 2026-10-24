"""Save native outputs and actual bills without opening any truth labels."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import types

HERE = Path(__file__).resolve().parent.parent
P2 = HERE.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--ranks', default='37,37')
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(x['num_threads'] == 1 for x in threadpool_info())
    stub = types.ModuleType('whestbench')
    stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    manifest = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    spec = importlib.util.spec_from_file_location('screen_candidate', args.candidate.resolve())
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    records = []
    for index, rank in enumerate(map(int, args.ranks.split(','))):
        weight_path = (STORE / Path(manifest[rank]['path']).parent / 'weights.npy'
                       if rank < 3016500 else
                       P2 / 'public_g19_attribution_v1' / f'fresh_{rank}_weights.npy')
        weights = np.load(weight_path)
        assert weights.shape == (16, 1024, 1024)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ingress:
            ww = [fnp.asarray(weights[l]) for l in range(16)]
        mlp = types.SimpleNamespace(width=1024, depth=16, weights=ww, seed=0, name=f'rank{rank}')
        began = time.monotonic()
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            if index == 0:
                spec.loader.exec_module(mod)
                estimator = mod.Estimator()
            result = estimator.predict(mlp, 2**41)
        seconds = time.monotonic() - began
        prediction = np.asarray(result, dtype=np.float32).copy()
        assert prediction.shape == (16, 1024) and np.isfinite(prediction).all()
        assert (prediction >= 0).all()
        path = args.out / f'call_{index:02d}_rank{rank:04d}.npy'
        np.save(path, prediction)
        bill = ctx.summary_dict()['flops_used']
        record = dict(index=index, rank=rank, output=str(path.resolve()),
                      output_sha256=sha(path), weights_sha256=sha(weight_path),
                      flops=bill, multiplier=max(.1, bill / 2**41), wall_s=seconds,
                      includes_initialization=index == 0,
                      input_flops_separate=ingress.summary_dict()['flops_used'],
                      max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        record['weight5_cache'] = mod._W5_PORT_LAST
        assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed']
        assert mod._W5_PORT_LAST['numeric_cache_retired']
        records.append(record)
        with (args.out / f'call_{index:02d}.json').open('x') as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps(record), flush=True)
        del prediction, result, weights, ww, mlp
    receipt = dict(candidate_sha256=sha(args.candidate), runner_sha256=sha(__file__),
                   numpy_version=np.__version__, numpy_file=np.__file__,
                   threads=threadpool_info(), truth_accessed=False, records=records)
    if len(records) == 2 and records[0]['rank'] == records[1]['rank']:
        receipt['repeat_byte_exact'] = records[0]['output_sha256'] == records[1]['output_sha256']
        assert receipt['repeat_byte_exact']
    with (args.out / 'result.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)


if __name__ == '__main__':
    main()
