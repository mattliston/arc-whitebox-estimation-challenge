"""Observe already-computed terminal representations; no targets or new math."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import types

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
CANDIDATE = D / 'read5_packet_v1/candidate.py'
SHA = 'd7723068ae269155e0369e4ba322b447222bbb31295d89394c069695ad0e824b'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def observed_source():
    source = CANDIDATE.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == SHA
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'gnn_step']
    original = ast.get_source_segment(source, nodes[-1])
    changed = original
    edits = []
    for name, line in (
        ('input', '    h=fnp.concatenate([x,fnp.clip(msgs,-8,8)],axis=1)'),
        ('silu1', '    h=silu(h@P["enc0_w"].T+P["enc0_b"])'),
        ('silu2', '    h=silu(h@P["enc2_w"].T+P["enc2_b"])'),
        ('hidden', '    state=h@P["enc4_w"].T+P["enc4_b"]'),
    ):
        assert changed.count(line) == 1
        value = 'state' if name == 'hidden' else 'h'
        added = line + f'\n    if l == 15:\n        _TERMINAL_REP_TAP["{name}"] = {value}'
        if name == 'hidden':
            added += '\n        _TERMINAL_REP_TAP["sigma"] = sigma'
        changed = changed.replace(line, added, 1)
        edits.append((line, added))
    restored = changed
    for old, new in reversed(edits):
        restored = restored.replace(new, old, 1)
    assert restored == original
    ast.parse(changed)
    return changed, dict(original_gnn_sha256=hashlib.sha256(original.encode()).hexdigest(),
                         observed_gnn_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                         inverse_byte_exact=True, extra_numeric_operations=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--ranks', required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(t['num_threads'] == 1 for t in threadpool_info())
    observed, proof = observed_source()
    stub = types.ModuleType('whestbench')
    stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    spec = importlib.util.spec_from_file_location('terminal_rep_parent', CANDIDATE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    manifest = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    ranks = list(map(int, args.ranks.split(',')))
    baseline = (json.loads((D / 'read5_packet_v1/dev/result.json').read_text())['records']
                if ranks == [35, 100, 35] else None)
    records, previous = [], {}
    held = {}
    for call, rank in enumerate(ranks):
        wp = (STORE / manifest[rank]['path']).parent / 'weights.npy'
        weights = np.load(wp)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ingress:
            ww = [fnp.asarray(weights[l]) for l in range(16)]
        mlp = types.SimpleNamespace(width=1024, depth=16, weights=ww, seed=0, name=f'rank{rank}')
        held.clear()
        started = time.monotonic()
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            if call == 0:
                spec.loader.exec_module(mod)
                mod._TERMINAL_REP_TAP = held
                exec(compile(observed, str(HERE / 'observed_gnn_step.py'), 'exec'), mod.__dict__)
                estimator = mod.Estimator()
            prediction = estimator.predict(mlp, 2**41)
        parent_wall = time.monotonic() - started
        parent_flops = ctx.summary_dict()['flops_used']
        assert set(held) == {'input', 'silu1', 'silu2', 'hidden', 'sigma'}
        arrays = {k: np.asarray(v, dtype=np.float32).copy() for k, v in held.items()}
        arrays['prediction'] = np.asarray(prediction, dtype=np.float32).copy()
        for key, shape in [('input', (1024, 126)), ('silu1', (1024, 128)), ('silu2', (1024, 128)),
                           ('hidden', (1024, 64)), ('sigma', (1024,)), ('prediction', (16, 1024))]:
            assert arrays[key].shape == shape and np.isfinite(arrays[key]).all(), key
        assert (arrays['prediction'] >= 0).all() and (arrays['sigma'] > 0).all()
        if baseline is not None:
            parent = baseline[call]
            assert rank == parent['rank'] and sha(wp) == parent['weights_sha256']
            assert np.array_equal(arrays['prediction'], np.load(parent['output']))
            assert parent_flops == parent['flops']
        if rank in previous:
            assert all(np.array_equal(a, previous[rank][k]) for k, a in arrays.items())
        previous[rank] = arrays
        target = args.out / f'call_{call:02d}.npz'
        np.savez(target, **arrays)
        rec = dict(call=call, rank=rank, weights_sha256=sha(wp), candidate_sha256=SHA,
                    capture=str(target.resolve()), capture_sha256=sha(target), parent_flops=parent_flops,
                    parent_multiplier=max(.1, parent_flops/2**41), includes_initialization=call == 0,
                    parent_wall_s=parent_wall, input_flops=ingress.summary_dict()['flops_used'],
                    truth_accessed=False, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        records.append(rec)
        print(json.dumps(rec), flush=True)
        held.clear()
        del prediction, weights, ww, mlp
    result = dict(completed=True, source_sha256=sha(__file__), protocol_sha256=sha(HERE / 'PROTOCOL.md'),
                   source_proof=proof, records=records, repeat_byte_exact=ranks == [35, 100, 35],
                   parent_all16_and_flops_exact=baseline is not None, truth_accessed=False)
    with (args.out / 'completed.json').open('x') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
