"""Literal signed-pair permutation/gradient certificate; inert without --run.

This supplements, never replaces or reruns, the completed original tiny test.
It uses no network, head payload, truth, fitting, or new random fixture.
"""
import argparse
from datetime import datetime, timezone
import os
import resource
import signal
import time
import traceback
from common import HERE, OLD, bindings, completed, module, sha, unchanged, write

TINY_SOURCE_SHA = '69c2e2299a2ca3728058b6a1fb3bd135159b8c216758fc8a91faba687bf4e433'


def certify(np, torch, model, original_tiny):
    assert sha(HERE/'check_tiny.py') == TINY_SOURCE_SHA
    old = module(HERE/'check_tiny.py', 'cap4096_original_tiny_fixture')
    p, lam, nb, pos, neg, live = old.fixture(np)
    native = dict(fnp=np, _mm=lambda a, b: a@b, _cp_diagonal_cube=lambda x: (x*x)*x)
    exec(compile((OLD/'core.py').read_text(), str(OLD/'core.py')+':identity_fixture', 'exec'), native)
    # Fixed within-old and within-new pair permutations, both signs together.
    oldperm, newperm = np.array([1, 0, 2]), np.array([2, 0, 1])
    permutation = np.concatenate((oldperm, oldperm+3, newperm+6, newperm+9))
    changed_nb = (6, nb[1][newperm], nb[2][:, newperm], nb[3][newperm], nb[4][newperm])
    changed_p, changed_lam = p[:, np.concatenate((oldperm, oldperm+3))], lam[permutation]
    physical = native['_paircap_materialize'](p, nb)
    changed_physical = native['_paircap_materialize'](changed_p, changed_nb)
    assert np.array_equal(changed_physical, physical[:, permutation])
    labels = [(0, i) for i in range(3)]+[(1, int(i)) for i in nb[1]]
    changed_labels = [labels[i] for i in oldperm]+[labels[3+i] for i in newperm]
    full_tensor = lambda x, w: np.einsum('ir,jr,kr,r->ijk', x.astype(float), x.astype(float), x.astype(float), w.astype(float))
    tensor = full_tensor(physical, lam)
    allowed = np.zeros((8, 8), bool)
    allowed[np.ix_(live, live)] = True
    np.fill_diagonal(allowed, True)
    cp = np.where(allowed, np.einsum('iij->ij', tensor), 0)
    T = lambda x: torch.tensor(x)

    def native_state(pp, ll, newborn):
        return native['_paircap_retire'](pp, ll, newborn, pos, neg, cp, live, 4)
    first = native_state(p, lam, nb)
    second = native_state(changed_p, changed_lam, changed_nb)
    retained = native['_paircap_materialize'](first[0], first[2])
    retained_changed = native['_paircap_materialize'](second[0], second[2])
    left = model.retire(T(physical), T(lam), T(pos), T(neg), 3, T(cp), T(live), 4)
    right = model.retire(T(changed_physical), T(changed_lam), T(pos), T(neg), 3, T(cp), T(live), 4)
    assert left[5]['minimum_relative_margin'] > 0 and right[5]['minimum_relative_margin'] > 0
    identities = sorted(labels[i] for i in left[5]['chosen'])
    changed_identities = sorted(changed_labels[i] for i in right[5]['chosen'])
    assert identities == changed_identities
    assert left[5]['chosen'] != right[5]['chosen'], 'Fixture must expose the old local-slot comparison defect'
    errors = {}
    def close(name, a, b):
        error = float(np.max(np.abs(a-b)))
        errors[name] = error
        assert error <= 1e-12, (name, error)
    close('retained_literal_tensor', full_tensor(retained, first[1]), full_tensor(retained_changed, second[1]))
    close('removed_live_plane_all_diagonals', cp-first[5], cp-second[5])
    close('torch_native_first_plane', left[4].numpy(), first[5])
    close('torch_native_permuted_plane', right[4].numpy(), second[5])
    repaired_star = lambda plane: 3*(cp-plane).T-2*np.diag(np.diag(cp-plane))
    close('repaired_star', repaired_star(first[5]), repaired_star(second[5]))
    w = ((np.arange(64).reshape(8, 8)%11)-5).astype(np.float32)/16
    pushed = native['_paircap_push_bank_shared_newborn'](w, first[0], np.arange(8), first[2])
    pushed_changed = native['_paircap_push_bank_shared_newborn'](w, second[0], np.arange(8), second[2])
    close('next_pushed_literal_tensor', full_tensor(pushed, first[1]), full_tensor(pushed_changed, second[1]))

    # Same smooth functional on each physical representation, with exact
    # locally constant choice and a pulled-back gradient. No FD of FP32 casts.
    cot = torch.tensor(((np.arange(64).reshape(8, 8)%13)-6)/32, dtype=torch.float64)
    direction = torch.arange(1, 9, dtype=torch.float64)/8
    def functional(x, ll):
        answer = model.retire(x, ll, T(pos), T(neg), 3, T(cp), T(live), 4)
        kept, weights, adjusted = answer[0], answer[1], answer[4]
        removal = T(cp)-adjusted
        star = 3*removal.T-2*torch.diag(torch.diagonal(removal))
        return ((kept.T@direction)**3*weights).sum()/64+(star*cot).sum(), answer[5]['chosen']
    x = torch.tensor(physical.astype(float), requires_grad=True)
    ll = torch.tensor(lam.astype(float), requires_grad=True)
    xx = torch.tensor(changed_physical.astype(float), requires_grad=True)
    lll = torch.tensor(changed_lam.astype(float), requires_grad=True)
    value, ids = functional(x, ll)
    changed_value, changed_ids = functional(xx, lll)
    value.backward(); changed_value.backward()
    close('functional_value', np.array(float(value.detach())), np.array(float(changed_value.detach())))
    close('factor_gradient_pullback', x.grad.numpy()[:, permutation], xx.grad.numpy())
    close('lambda_gradient_pullback', ll.grad.numpy()[permutation], lll.grad.numpy())
    dx = torch.tensor(((np.arange(96).reshape(8, 12)%7)-3)/64, dtype=torch.float64)
    dl = torch.tensor(np.array([1, -2, 3, -1, 2, -3, 2, -1, 1, -2, 1, -1])/64, dtype=torch.float64)
    analytic = float((x.grad*dx).sum()+(ll.grad*dl).sum())
    changed_analytic = float((xx.grad*dx[:, permutation]).sum()+(lll.grad*dl[permutation]).sum())
    close('directional_derivative_pullback', np.array(analytic), np.array(changed_analytic))
    eps = 1e-5
    plus, ip = functional(x.detach()+eps*dx, ll.detach()+eps*dl)
    minus, im = functional(x.detach()-eps*dx, ll.detach()-eps*dl)
    assert ip == ids == im
    finite = float((plus-minus)/(2*eps))
    relative = abs(finite-analytic)/max(abs(analytic), 1e-12)
    assert abs(analytic) > 1e-10 and relative <= 1e-6
    assert float(torch.linalg.vector_norm(x.grad[:, [2, 5, 8, 11]])) > 1e-10

    # Deliberate cutoff tie: stable local ordering MAY change the physical
    # winners. This must remain a rejected full-width identity comparison.
    tied = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1], np.float32)
    tie_a = model.retire(T(physical), T(tied), T(pos), T(neg), 3, T(cp), T(live), 4)[5]
    tie_b = model.retire(T(changed_physical), T(tied[permutation]), T(pos), T(neg), 3, T(cp), T(live), 4)[5]
    assert tie_a['minimum_relative_margin'] == tie_b['minimum_relative_margin'] == 0
    assert tie_a['chosen'] == tie_b['chosen'] == [0, 1, 2, 3]
    tie_ids_a = sorted(labels[i] for i in tie_a['chosen'])
    tie_ids_b = sorted(changed_labels[i] for i in tie_b['chosen'])
    assert tie_ids_a != tie_ids_b
    comparator = module(HERE/'check_wide_v3.py', 'cap4096_identity_comparator_for_literal_test')
    def comparison_fixture(native_chosen, torch_chosen):
        native_record = dict(selections=[], retirement=[dict(pairs_before=6, chosen=native_chosen)])
        torch_record = dict(selections={}, retirement={})
        for layer in range(16):
            for slot in range(2 if layer == 15 else 3):
                nids, tids = live.tolist(), live.tolist()
                if slot == 2:
                    nids = list(range(3)) if layer == 0 else nb[1].tolist() if layer == 1 else []
                    tids = oldperm.tolist() if layer == 0 else nb[1][newperm].tolist() if layer == 1 else []
                native_record['selections'].append(nids)
                torch_record['selections'][str((layer, slot))] = tids
            if layer < 15:
                torch_record['retirement'][str(layer)] = dict(
                    pairs_before=3 if layer == 0 else 6 if layer == 1 else 4,
                    chosen=list(range(3)) if layer == 0 else torch_chosen if layer == 1 else list(range(4)),
                    retired=(layer == 1))
        return comparator.identity_comparison(torch_record, native_record, cap=4)
    strict_comparison = comparison_fixture(left[5]['chosen'], right[5]['chosen'])
    tie_comparison = comparison_fixture(tie_a['chosen'], tie_b['chosen'])
    assert strict_comparison['passed'] and not tie_comparison['passed']
    assert any(x['type'] == 'retained_pair_identity' for x in tie_comparison['mismatches'])
    old_tie = next(x for x in original_tiny['checks']['cases'] if x['case'] == 'stable_equal_energies')
    assert old_tie['chosen'] == [0, 1, 2, 3] and old_tie['physical_and_maps_exact']
    return dict(errors=errors, strict_gap_physical_identities=identities,
        local_slots_original=left[5]['chosen'], local_slots_permuted=right[5]['chosen'],
        strict_relative_margin=left[5]['minimum_relative_margin'], gradient_relative=relative,
        cutoff_tie_changes_physical_identities=True, tie_identities_original=tie_ids_a,
        tie_identities_permuted=tie_ids_b, original_stable_tie_completion_preserved=True,
        production_identity_comparator_strict_case_passed=True,
        production_identity_comparator_cutoff_tie_rejected=True,
        exact_real_permutation_identity=True, FP32_operation_bit_identity_not_claimed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--tiny-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    out = HERE/'identity_v3'
    assert not out.exists()
    original_tiny = completed(HERE/'tiny', args.tiny_sha256)
    pins = bindings((__file__, HERE/'common.py', HERE/'model.py', HERE/'check_tiny.py',
        HERE/'check_wide_v3.py', HERE/'PROTOCOL.md', HERE/'IDENTITY_COMPARISON_V3.md',
        HERE/'tiny/result.json', HERE/'tiny/completion.json'))
    out.mkdir()
    start = time.monotonic()
    result = dict(passed=False, schema='cap4096_identity_certificate_v3', source_bindings=pins,
        original_tiny_sha256=args.tiny_sha256, network_accessed=False, truth_accessed=False,
        started_utc=datetime.now(timezone.utc).isoformat(), hard_wall_limit_s=60, rss_limit_bytes=2000000000)
    write(out/'started.json', result)
    def watchdog(signum, frame):
        if time.monotonic()-start >= 60 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >= 2000000000:
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise RuntimeError('Identity certificate60s/2GB resource limit')
    previous = signal.signal(signal.SIGALRM, watchdog)
    signal.setitimer(signal.ITIMER_REAL, .1, .1)
    try:
        import numpy as np
        import torch
        import model
        torch.set_num_threads(1)
        result['checks'] = certify(np, torch, model, original_tiny)
        unchanged(pins)
        result['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert result['peak_rss_bytes'] < 2000000000 and time.monotonic()-start < 60
        result['passed'] = True
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    result['wall_s'] = time.monotonic()-start
    write(out/'result.json', result)
    write(out/'completion.json', dict(passed=result['passed'], result_sha256=sha(out/'result.json')))
    print(result, flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
