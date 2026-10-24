"""Fixed literal retirement/representation/gradient prerequisite. Inert without --run."""
import argparse
from datetime import datetime, timezone
import os
import resource
import signal
import time
import traceback
from common import HERE, OLD, bindings, sha, unchanged, write


def fixture(np):
    n = 8
    positive, negative = np.arange(3), np.arange(3, 6)
    oldbase = np.full((n, 3), .25, dtype=np.float32)
    oldp, oldq = oldbase.copy(), oldbase.copy()
    oldp[np.arange(3), np.arange(3)] = .5
    oldq[np.arange(3), np.arange(3)] = -.5
    p = np.concatenate((oldp, oldq), axis=1)
    selected = np.array([3, 5, 7], dtype=np.int64)
    base = np.full((n, 3), .25, dtype=np.float32)
    base[selected, np.arange(3)] = 0
    newborn = (6, selected, base, np.full(3, .5, np.float32), np.full(3, -.5, np.float32))
    lam = np.array([8, 4, 1, -8, -4, -1, 16, 2, .5, -16, -2, -.5], dtype=np.float32)
    live = np.array([0, 1, 3, 5, 7], dtype=np.int64)
    return p, lam, newborn, positive, negative, live


def checks(np, torch, model):
    native = dict(fnp=np, _mm=lambda a, b: a@b, _cp_diagonal_cube=lambda x: (x*x)*x)
    exec(compile((OLD/'core.py').read_text(), str(OLD/'core.py')+':numpy_fixture', 'exec'), native)
    T = lambda x: torch.tensor(x)
    p, lam, nb, pos, neg, live = fixture(np)
    physical = native['_paircap_materialize'](p, nb)
    tensor = lambda x, w: np.einsum('ir,jr,kr,r->ijk', x.astype(float), x.astype(float), x.astype(float), w.astype(float))
    original = tensor(physical, lam)
    plane = np.einsum('iij->ij', original)
    allowed = np.zeros((8, 8), dtype=bool)
    allowed[np.ix_(live, live)] = True
    np.fill_diagonal(allowed, True)
    records = []

    def compare(label, pp, ll, newborn, positive, negative, cap, expected=None):
        full = native['_paircap_materialize'](pp, newborn)
        # Independent actual-cube repeated plane, not native helper's output.
        fulltensor = tensor(full, ll)
        cp = np.einsum('iij->ij', fulltensor)
        original_cp = np.where(allowed, cp, 0)
        count = 0 if newborn is None else len(newborn[1])
        got = native['_paircap_retire'](pp, ll, newborn, positive, negative, original_cp, live, cap)
        dense = native['_paircap_materialize'](got[0], got[2])
        tg = model.retire(T(full), T(ll), T(positive), T(negative), count,
                          T(original_cp), T(live), cap)
        assert np.array_equal(dense, tg[0].detach().numpy())
        assert np.array_equal(got[1], tg[1].detach().numpy())
        assert np.array_equal(got[3], tg[2].numpy()) and np.array_equal(got[4], tg[3].numpy())
        assert np.max(np.abs(got[5]-tg[4].detach().numpy())) <= 1e-12
        if expected is not None:
            assert tg[5]['chosen'] == expected, (label, tg[5]['chosen'])
        retained = tensor(dense, got[1])
        removed = fulltensor-retained
        repeated = np.einsum('iij->ij', removed)
        assert np.max(np.abs((original_cp-got[5])[allowed]-repeated[allowed])) <= 1e-12
        assert np.count_nonzero((original_cp-got[5])[~allowed]) == 0
        # Star restoration repairs current live repeated faces and every scalar
        # diagonal, not all-distinct entries or masked cold off-diagonal faces.
        repair = original_cp-got[5]
        star = 3*repair.T-2*np.diag(np.diag(repair))
        starplane = (star.T+2*np.diag(np.diag(star)))/3
        assert np.max(np.abs((retained[np.arange(8), np.arange(8)]+starplane-cp)[allowed])) <= 1e-12
        w = ((np.arange(64).reshape(8, 8)%11)-5).astype(np.float32)/16
        pushed = native['_paircap_push_bank_shared_newborn'](w, got[0], np.arange(8), got[2])
        assert np.max(np.abs(pushed.astype(float)-w.astype(float).T@dense.astype(float))) <= 1e-12
        if not tg[5]['retired']:
            assert got[0] is pp and got[1] is ll and got[2] is newborn and got[5] is original_cp
        records.append(dict(case=label, **tg[5], physical_and_maps_exact=True,
                            literal_repeated_restoration=True, shared_push_real_exact=True))
        return got

    mixed = compare('mixed_old_new', p, lam, nb, pos, neg, 4, [0, 1, 3, 4])
    compare('below_cap_no_rewrite', p, lam, nb, pos, neg, 6, list(range(6)))
    tied = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1], np.float32)
    compare('stable_equal_energies', p, tied, nb, pos, neg, 4, [0, 1, 2, 3])
    for label, old, new, expected in (
            ('all_old_removed', 1, 16, [3, 4, 5]),
            ('all_new_removed', 16, 1, [0, 1, 2])):
        ll = np.array([old]*3+[-old]*3+[new]*3+[-new]*3, np.float32)
        compare(label, p, ll, nb, pos, neg, 3, expected)
    prior = native['_paircap_materialize'](mixed[0], mixed[2])
    later = (prior.shape[1], nb[1][:2], nb[2][:, :2], nb[3][:2], nb[4][:2])
    laterlam = np.concatenate((mixed[1], np.array([32, .25, -32, -.25], np.float32)))
    compare('second_retirement_non32_partition', prior, laterlam, later, mixed[3], mixed[4], 4)

    # Smooth fixed-cell CPU64 algebra: storage-rounding derivative is NOT
    # certified by a finite difference through a quantizing FP32 cast.
    full = physical.astype(float)
    x = torch.tensor(full, dtype=torch.float64, requires_grad=True)
    l = torch.tensor(lam.astype(float), dtype=torch.float64, requires_grad=True)
    cp = torch.tensor(np.where(allowed, plane, 0), dtype=torch.float64)
    cot = torch.tensor(((np.arange(64).reshape(8, 8)%13)-6)/32, dtype=torch.float64)
    def objective(xx, ll):
        result = model.retire(xx, ll, T(pos), T(neg), 3, cp, T(live), 4)
        # Include live downstream cubes and the restored star independently.
        kept, weights, adjusted = result[0], result[1], result[4]
        projected = kept.T@torch.arange(1, 9, dtype=torch.float64)/8
        return (projected**3*weights).sum()*.01+(adjusted*cot).sum(), result[5]['chosen']
    value, chosen = objective(x, l)
    value.backward()
    dx = torch.tensor(((np.arange(96).reshape(8, 12)%7)-3)/64, dtype=torch.float64)
    # Opposite weights are a state invariant. Perturb pairs, never one sign.
    dl = torch.tensor(np.array([1, -2, 3, -1, 2, -3, 2, -1, 1, -2, 1, -1])/64, dtype=torch.float64)
    analytic = float((x.grad*dx).sum()+(l.grad*dl).sum())
    eps = 1e-5
    plus, ids_plus = objective(x.detach()+eps*dx, l.detach()+eps*dl)
    minus, ids_minus = objective(x.detach()-eps*dx, l.detach()-eps*dl)
    assert ids_plus == chosen == ids_minus
    finite = float((plus-minus)/(2*eps))
    relative = abs(finite-analytic)/max(abs(analytic), 1e-12)
    assert relative <= 1e-6 and abs(analytic) > 1e-10
    dropped = [2, 5, 8, 11]
    dropped_gradient = float(torch.linalg.vector_norm(x.grad[:, dropped]))
    assert dropped_gradient > 1e-10, 'Dropped-plane feedback was detached'
    assert torch.isfinite(x.grad).all() and torch.isfinite(l.grad).all()
    return dict(cases=records, directional_gradient_relative=relative,
        gradient_analytic=analytic, gradient_finite=finite, dropped_factor_gradient_norm=dropped_gradient,
        gradient_cell_fixed=True, numerical_cast_derivative_not_claimed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    out = HERE/'tiny'
    assert not out.exists()
    pins = bindings((__file__, HERE/'common.py', HERE/'model.py', HERE/'PROTOCOL.md', HERE/'IMPLEMENTATION.md'))
    out.mkdir()
    start = time.monotonic()
    result = dict(passed=False, schema='current_paircap4096_e2e_tiny_v1', source_bindings=pins,
        truth_accessed=False, network_accessed=False, hard_wall_limit_s=60,
        rss_limit_bytes=2000000000, started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'started.json', result)
    def watchdog(signum, frame):
        if time.monotonic()-start >= 60:
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise TimeoutError('Fixed tiny certificate exceeded60s, including numerical imports')
        if int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) >= 2000000000:
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise MemoryError('Fixed tiny certificate exceeded2GB RSS')
    previous_handler = signal.signal(signal.SIGALRM, watchdog)
    signal.setitimer(signal.ITIMER_REAL, .1, .1)
    try:
        import numpy as np
        import torch
        import model
        torch.set_num_threads(1)
        result['checks'] = checks(np, torch, model)
        result['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert result['peak_rss_bytes'] < 2000000000
        unchanged(pins)
        assert time.monotonic()-start < 60
        result['passed'] = True
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    result['wall_s'] = time.monotonic()-start
    write(out/'result.json', result)
    write(out/'completion.json', dict(passed=result['passed'], result_sha256=sha(out/'result.json')))
    print(result, flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
