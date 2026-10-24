"""Tiny actual-native helper fixture; root runs with a candidate path argument."""
from pathlib import Path
import hashlib
import json
import sys
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE = Path(__file__).resolve().parent
BLOCKS = ((0, 2), (4, 1))


def pin(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def paired(common, difference):
    F = np.empty((len(common), 6), dtype=np.float32)
    b = 0
    for a, h in BLOCKS:
        F[:, a:a+h] = common[:, b:b+h]+difference[:, b:b+h]
        F[:, a+h:a+2*h] = common[:, b:b+h]-difference[:, b:b+h]
        b += h
    return F


def close(actual, expected):
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-6, atol=2e-6)


def main():
    candidate = Path(sys.argv[1]).resolve()
    assert not (HERE/'unit_result.json').exists()
    fragment = (HERE/'fragment.py').read_bytes()
    source = candidate.read_bytes()
    assert source.endswith(fragment), 'fixture must exercise the unchanged appended fragment'
    stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    mod = types.ModuleType('common_energy_candidate'); mod.__file__ = str(candidate)
    sys.modules[mod.__name__] = mod
    exec(compile(source, str(candidate), 'exec'), mod.__dict__)
    mod._shared_old384_mm = lambda a, b: a @ b
    original_put = mod.fnp.put; writes = []
    def observed_put(a, indices, values, *args, **kwargs):
        writes.append((id(a), int(indices.size)))
        return original_put(a, indices, values, *args, **kwargs)
    mod.fnp.put = observed_put
    cases = []
    def case(name, common, expected_indices, ru):
        rank = len(common)
        difference = (np.arange(rank*3).reshape(rank, 3) % 7-3).astype(np.float32)/32
        F = paired(common, difference)
        truncated = np.zeros_like(common)
        selected = list(range(rank)) if expected_indices is None else expected_indices
        truncated[selected] = common[selected]
        explicit = paired(truncated, difference)
        U = (np.arange(5*rank).reshape(5, rank) % 11-5).astype(np.float32)/16
        left = (np.arange(30).reshape(5, 6) % 9).astype(np.float32)/8
        lam = np.asarray([.125, -.25, -.125, .25, .5, -.5], dtype=np.float32)
        mod._UVP_RU = ru
        with flops.BudgetContext(flop_budget=2**28, quiet=True) as budget:
            f, u, left_sq, coeff = map(fnp.asarray, (F, U, left*left, lam))
            uv = mod._uvp_split(f, BLOCKS)
            assert uv[2] == BLOCKS and uv[3] == min(ru, rank)
            if expected_indices is None:
                assert uv[4] is None
                old = mod._UCE_ORIGINAL_SPLIT(f, BLOCKS)
                for new_part, old_part in zip(uv[:2], old[:2]):
                    assert np.asarray(new_part).tobytes() == np.asarray(old_part).tobytes()
            else:
                assert np.asarray(uv[4]).tolist() == expected_indices
            close(uv[0], common[selected]); close(uv[1], difference)
            recon = mod._uvp_recon(u, uv)
            before = len(writes); inner = mod._uvp_inner(left_sq, coeff, uv)
            assert writes[before:] == ([] if expected_indices is None else [(id(inner), 5*ru)])
            if expected_indices is None:
                assert np.asarray(recon).tobytes() == np.asarray(mod._UCE_ORIGINAL_RECON(u, old)).tobytes()
                assert np.asarray(inner).tobytes() == np.asarray(mod._UCE_ORIGINAL_INNER(left_sq, coeff, old)).tobytes()
        close(recon, U @ explicit)
        close(inner, (left*left*lam[None, :]) @ explicit.T)
        assert np.asarray(f).tobytes() == F.tobytes()
        assert not mod._BH_HOLD
        cases.append(dict(name=name, passed=True, indices=expected_indices,
                          flops=int(budget.summary_dict()['flops_used'])))
        return uv, explicit, left*left, lam
    try:
        common = np.asarray([[.25,0,0], [2,1,0], [0,0,4], [.5,.5,.5]], dtype=np.float32)
        base, Fb, Lb, lb = case('nonprefix common selection', common, [1,2], 2)
        ties = np.asarray([[0,0,0], [2,0,0], [0,2,0], [0,0,2]], dtype=np.float32)
        case('stable equal-energy ties', ties, [1,2], 2)
        case('full-rank exact original path', common, None, 8)
        recent_common = np.asarray([[0,4,0], [.25,0,0], [0,.5,0], [0,0,.25], [4,0,0], [.5,0,.5]], dtype=np.float32)
        recent, Fj, Lj, lj = case('independent lazy recent selection', recent_common, [0,4], 2)
        U = (np.arange(30).reshape(5,6) % 11-5).astype(np.float32)/16
        mapping = (np.arange(24).reshape(6,4) % 7-3).astype(np.float32)/8
        with flops.BudgetContext(flop_budget=2**28, quiet=True) as budget:
            u, m = fnp.asarray(U), fnp.asarray(mapping)
            reconstruction = fnp.concatenate([mod._uvp_recon(u @ m, base), mod._uvp_recon(u, recent)], axis=1)
            inner = mod._uvp_inner(fnp.asarray(Lb), fnp.asarray(lb), base) @ m.T
            inner = inner + mod._uvp_inner(fnp.asarray(Lj), fnp.asarray(lj), recent)
        combined = np.concatenate([mapping @ Fb, Fj], axis=1)
        close(reconstruction, U @ combined)
        close(inner, np.concatenate([Lb*lb[None,:], Lj*lj[None,:]], axis=1) @ combined.T)
        cases.append(dict(name='lazy base and join coordinates', passed=True,
                          flops=int(budget.summary_dict()['flops_used'])))
        with flops.BudgetContext(flop_budget=2**28, quiet=True):
            target = fnp.zeros((2,4), dtype=fnp.float32)
            positions = fnp.asarray([1,6], dtype=fnp.int64); values = fnp.asarray([3,5], dtype=fnp.float32)
        with flops.BudgetContext(flop_budget=2**28, quiet=True) as budget:
            original_put(target, positions, values)
        assert budget.summary_dict()['flops_used'] == 2
        np.testing.assert_array_equal(np.asarray(target), [[0,3,0,0], [0,0,5,0]])
    finally:
        mod.fnp.put = original_put
    result = dict(passed=True, candidate=pin(candidate), fragment=pin(HERE/'fragment.py'),
        unit=pin(Path(__file__).resolve()), cases=cases, paid_flat_scatter_flops=2,
        small_products_only=True, estimator_setup_called=False, no_targets=True)
    with (HERE/'unit_result.json').open('x') as output:
        json.dump(result, output, indent=2); output.write('\n')
    print(json.dumps(dict(passed=True, cases=len(cases), candidate=result['candidate'])))


if __name__ == '__main__':
    main()
