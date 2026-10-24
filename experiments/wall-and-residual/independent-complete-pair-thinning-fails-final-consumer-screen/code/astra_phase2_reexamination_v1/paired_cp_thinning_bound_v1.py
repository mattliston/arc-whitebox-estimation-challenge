"""Target-free best independent-pair thinning variance at the final consumer."""
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from centered_bank_autodiff_v3 import ExplicitBoundBankFullCard, marginals, scalar_features, final_read
from full_carrier_autodiff_v1 import diagonal

HERE = Path(__file__).resolve().parent
OUT = HERE/'paired_cp_thinning_bound_v1.json'


def arr(x):
    return x.detach().cpu().double().numpy().copy()


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def optimal_probabilities(norms, budget):
    assert 0 < budget < len(norms)
    lo, hi = 0., max(float(np.max(norms))*len(norms)/budget, 1e-30)
    for _ in range(80):
        tau = (lo+hi)/2
        if np.minimum(1., norms/tau).sum() > budget:
            lo = tau
        else:
            hi = tau
    p = np.minimum(1., norms/hi)
    # Zero-norm atoms have no consumer variance and may be discarded.
    variance = float(np.sum((1/p[p > 0]-1)*norms[p > 0]**2))
    return p, variance


def enumeration_certificate():
    rng = np.random.Generator(np.random.PCG64DXSM(930196))
    atoms = rng.normal(size=(5, 7))
    direct, left, right = rng.normal(size=(3, 5))
    effects = direct[:, None]*atoms+left[:, None]*(right@atoms)[None, :]
    norms = np.sqrt(np.mean(effects**2, axis=0))
    probabilities, formula = optimal_probabilities(norms, 3.5)
    enumerated = 0.
    for values in itertools.product((0, 1), repeat=7):
        keep = np.asarray(values)
        mass = np.prod(np.where(keep, probabilities, 1-probabilities))
        error = effects@(keep/probabilities-1)
        enumerated += mass*np.mean(error**2)
    error = abs(enumerated-formula)/formula
    result = dict(formula=formula, enumeration=enumerated, relative_error=error,
                  expected_count=float(probabilities.sum()), passed=bool(error <= 1e-11))
    assert result['passed']
    return result


def capture(rank):
    card = ExplicitBoundBankFullCard(rank, 'mps', audit_selections=True)
    with torch.no_grad():
        state = card.initial()
        zero = torch.zeros(card.n, device='mps')
        means = []
        for layer in range(15):
            state = card.step(layer, state, zero, card.P0)
            means.append(np.maximum(arr(state[0]), 0))
        m, C, p, lam, star, b, alive, radial, hidden = state
        w = card.w[15]
        mu = w.T@m
        pushed = C@w
        v = (w*pushed).sum(0)
        metric = diagonal(C, torch.diagonal(C)*alive)
        metric_push = metric@w
        bv = (w*metric_push).sum(0)
        pp = w.T@p
        k3cp = (pp**3*lam).sum(1)
        k3 = k3cp+((w.T@star)*w.T*w.T).sum(1)+3*bv*(w.T@b)
        tape = {name: arr(value) for name, value in
                dict(mu=mu, v=v, k3=k3, pp=pp, lam=lam, radial=radial,
                     hidden=hidden, fixed=card.fixed[15], w=w).items()}
        tape['counts'] = [len(card.selections[2*l+1][0]) for l in range(15)]
        final = card.step(15, state, zero, card.P0)[0]
        means.append(np.maximum(arr(final), 0))
        gpu = int(torch.mps.driver_allocated_memory())
    baseline = np.load(HERE/'centered_parent_numerics_v1.npz')[f'baseline_{rank}']
    maximum = float(np.max(np.abs(np.stack(means)-baseline)))
    masks = np.load(HERE/f'full_carrier_local_cell_r{rank}_v1.npz')['baseline_masks'][:30]
    equal = bool(np.array_equal(masks, np.stack([x[1] for x in card.selections])))
    assert maximum <= 2e-6 and equal
    tape['baseline'] = means[-1]
    del card, state, m, C, p, lam, star, b, alive, radial, hidden, w
    gc.collect()
    torch.mps.empty_cache()
    return tape, dict(maximum_mean_error=maximum, memberships_equal=equal), gpu


def final_jacobian(tape, rank):
    # CPU final-only differentiation: no CPU f32 carrier and no threadpoolctl.
    mu, v = [torch.tensor(tape[k], dtype=torch.float64) for k in ('mu', 'v')]
    k4 = torch.tensor(tape['radial'], dtype=torch.float64)*v*v
    base_k3 = torch.tensor(tape['k3'], dtype=torch.float64)
    fixed, w, hidden = [torch.tensor(tape[k], dtype=torch.float32) for k in ('fixed', 'w', 'hidden')]
    P = {k: torch.tensor(v, dtype=torch.float32) for k, v in dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz')).items()}
    Q = {k: torch.tensor(v, dtype=torch.float32) for k, v in dict(np.load(HERE/'payload_fullpair_final_head_v1.npz')).items()}

    def forward(k3, independent_global=None):
        host, aux = marginals(mu, v, k3, k4)
        x = scalar_features(15, host, aux, fixed)
        if independent_global is not None:
            x = torch.cat((x[:, :29], independent_global[:, None].float()), dim=1)
        value = host+final_read(x, aux['sigma'], w, hidden, P, Q).double()
        return torch.where(value < 0, torch.zeros_like(value), value), host, aux

    k3 = base_k3.clone().requires_grad_()
    host0, aux0 = marginals(mu, v, k3, k4)
    global0 = (host0/aux0['sigma']).mean().detach().repeat(len(k3)).requires_grad_()
    value, host, aux = forward(k3, global0)
    direct, left = torch.autograd.grad(value.sum(), (k3, global0), retain_graph=True)
    right, = torch.autograd.grad((host/aux['sigma']).mean(), k3)
    direct, left, right = [arr(x) for x in (direct, left, right)]
    rng = np.random.Generator(np.random.PCG64DXSM(930296+rank))
    cot = rng.normal(size=len(k3)); cot /= np.linalg.norm(cot)
    actual, _, _ = forward(k3)
    reverse, = torch.autograd.grad((actual*torch.tensor(cot)).sum(), k3)
    reconstructed = direct*cot+right*(left@cot)
    jac_error = relative(reconstructed, arr(reverse))
    final_error = float(np.max(np.abs(arr(actual)-tape['baseline'])))
    assert jac_error <= 1e-5 and final_error <= 2e-6
    return direct, left, right, dict(reverse_relative_error=jac_error,
                                     final_replay_max_error=final_error)


def main():
    assert not OUT.exists() and not OUT.with_suffix('.npz').exists()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0'
    assert 'threadpoolctl' not in sys.modules
    assert json.loads((HERE/'centered_bank_baseline_mps_f32_v3.json').read_text())['passed']
    torch.set_num_threads(3)
    start = time.time()
    tiny = enumeration_certificate()
    print(json.dumps(dict(stage='enumeration', **tiny)), flush=True)
    records, saved = [], {}
    max_gpu = 0
    for rank in (35, 100):
        tape, fidelity, gpu = capture(rank)
        max_gpu = max(max_gpu, gpu)
        direct, left, right, derivative = final_jacobian(tape, rank)
        print(json.dumps(dict(rank=rank, stage='fidelity', **fidelity, **derivative)), flush=True)
        norms, cohorts, cp_sum = [], [], np.zeros(1024)
        offset = 0
        for layer, count in enumerate(tape['counts']):
            plus = tape['pp'][:, offset:offset+count]
            minus = tape['pp'][:, offset+count:offset+2*count]
            weight = tape['lam'][offset:offset+count]
            assert np.array_equal(weight, -tape['lam'][offset+count:offset+2*count])
            common, difference = (plus+minus)/2, plus-minus
            atoms = weight*(3*common*common*difference+.25*difference**3)
            cp_sum += atoms.sum(1)
            effect = direct[:, None]*atoms+left[:, None]*(right@atoms)[None, :]
            norms.append(np.sqrt(np.mean(effect*effect, axis=0)))
            cohorts.extend([layer]*count)
            offset += 2*count
        assert offset == len(tape['lam'])
        original_cp = (tape['pp']**3*tape['lam']).sum(1)
        pair_error = relative(cp_sum, original_cp)
        assert pair_error <= 1e-10
        norms = np.concatenate(norms)
        probability, variance = optimal_probabilities(norms, len(norms)/2)
        uniform_variance = float(np.sum(norms*norms))
        assert abs(probability.sum()-len(norms)/2) <= 1e-8
        cohorts = np.asarray(cohorts)
        row = dict(rank=rank, pair_count=len(norms), expected_retained_pairs=float(probability.sum()),
                   budget_fraction=.5, optimal_linearized_added_mse=variance,
                   uniform_half_linearized_added_mse=uniform_variance,
                   threshold=1e-9, passed=bool(variance <= 1e-9),
                   pair_sum_relative_error=pair_error, forward_fidelity=fidelity,
                   jacobian_certificate=derivative,
                   expected_pairs_by_cohort=[float(probability[cohorts == l].sum()) for l in range(15)])
        records.append(row)
        for name, value in dict(norms=norms, probabilities=probability, cohorts=cohorts,
                                direct=direct, left=left, right=right).items():
            saved[f'{name}_{rank}'] = value
        print(json.dumps(row), flush=True)
        del tape, atoms, effect, plus, minus, common, difference
        gc.collect()
    np.savez(OUT.with_suffix('.npz'), **saved)
    result = dict(passed=all(r['passed'] for r in records), records=records,
                  enumeration_certificate=tiny, target_access=False, actual_sampling=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest(),
                  utc_finished=datetime.now(timezone.utc).isoformat(), wall_seconds=time.time()-start,
                  peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  largest_observed_gpu_driver_bytes=max_gpu,
                  qualification='Exact variance for the local frozen final-consumer Jacobian and optimal independent Bernoulli full-pair retention. All prior-layer errors and probability acquisition/cost are free. No finite-perturbation, coadaptation, universal compression limit or score claim; process/GPU values are not independent additive peaks.')
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
