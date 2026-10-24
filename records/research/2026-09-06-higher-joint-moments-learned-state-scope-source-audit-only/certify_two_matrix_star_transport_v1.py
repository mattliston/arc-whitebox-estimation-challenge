"""Tiny target-free identity/cost audit for a proposed CP-free matrix student.

This does not test its missing all-distinct tensor information, train it,
run a wide network, or assert that it achieves any raw/adjusted score.
It verifies the proposed standardized THREE-product star transport only.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'two_matrix_star_transport_certificate_v1.json'
SEED = 930117


def dense_star(plane):
    n = len(plane)
    edge = plane.copy()
    np.fill_diagonal(edge, np.diag(plane) / 3)
    identity = np.eye(n)
    return (np.einsum('ab,ac->abc', identity, edge)
            + np.einsum('ac,ab->abc', identity, edge)
            + np.einsum('bc,ba->abc', identity, edge))


def three_product(plane, transform):
    edge = plane.copy()
    np.fill_diagonal(edge, np.diag(plane) / 3)
    pushed = edge @ transform
    return ((transform * transform).T @ pushed
            + 2 * (transform * pushed).T @ transform)


def relative(actual, reference):
    return float(np.linalg.norm(actual-reference) / max(np.linalg.norm(reference), 1.))


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserve existing certificate: {OUT}')
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError('Launch at nice >=10')
    start = time.time()
    rng = np.random.default_rng(SEED)
    rows = []
    for old_n, new_n in ((7, 5), (5, 7), (7, 7)):
        for kind in ('zero', 'diagonal', 'off_diagonal', 'general'):
            d = rng.standard_normal((old_n, old_n))
            if kind == 'zero':
                d *= 0
            elif kind == 'diagonal':
                d = np.diag(np.diag(d))
            elif kind == 'off_diagonal':
                np.fill_diagonal(d, 0)
            w = rng.standard_normal((old_n, new_n)) / np.sqrt(old_n)
            factor = rng.standard_normal((old_n, old_n))
            covariance = factor @ factor.T + np.eye(old_n)
            old_sigma = np.sqrt(np.diag(covariance))
            new_sigma = np.sqrt(np.diag(w.T @ covariance @ w))
            v = old_sigma[:, None] * w / new_sigma[None, :]
            tensor = dense_star(d)
            reference = np.einsum('abc,ai,bi,cj->ij', tensor, v, v, v)
            actual = three_product(d, v)
            recovered = tensor[np.arange(old_n), np.arange(old_n), :]
            p_old = rng.permutation(old_n)
            p_new = rng.permutation(new_n)
            permuted = three_product(d[np.ix_(p_old, p_old)], v[np.ix_(p_old, p_new)])
            scale_old = np.exp(rng.uniform(-2, 2, old_n))
            scale_new = np.exp(rng.uniform(-2, 2, new_n))
            rescaled_w = w * scale_new[None, :] / scale_old[:, None]
            rescaled_v = ((old_sigma * scale_old)[:, None] * rescaled_w
                          / (new_sigma * scale_new)[None, :])
            # Check physical units against the raw dense tensor transport.
            physical_tensor = tensor * np.einsum('a,b,c->abc', old_sigma, old_sigma, old_sigma)
            physical_reference = np.einsum('abc,ai,bi,cj->ij', physical_tensor, w, w, w)
            physical_actual = actual * new_sigma[:, None]**2 * new_sigma[None, :]
            errors = dict(three_product=relative(actual, reference),
                          repeated_reconstruction=relative(recovered, d),
                          permutation=relative(permuted, actual[np.ix_(p_new, p_new)]),
                          positive_scale_normalization=relative(rescaled_v, v),
                          physical_units=relative(physical_actual, physical_reference))
            rows.append(dict(old_n=old_n, new_n=new_n, fixture=kind, errors=errors))
    worst = max(value for row in rows for value in row['errors'].values())
    width = 1024
    gemm = 2 * width**3
    transitions = 15
    cost = dict(width=width, transitions=transitions,
                conventional_dense_product_flops=gemm,
                covariance_products_per_transition=2,
                coordinate_star_products_per_transition=3,
                latent_even_memory_products_per_transition=2,
                seven_product_transition_total=7*transitions*gemm,
                first_covariance_gram=gemm,
                optional_one_product_psd_extra=transitions*gemm,
                seven_product_plus_first_utilization=(7*transitions+1)*gemm/(2**41),
                eight_product_plus_first_utilization=(8*transitions+1)*gemm/(2**41),
                excludes='All nonlinear maps, pointwise updates, normalization, feature/head work, '
                         'storage/communication, and any unlisted state/source computation. '
                         'Not native metering, a final design bill, or a runtime certificate.')
    record = dict(schema='two-matrix-star-transport-certificate-v1', passed=worst <= 1e-12,
                  seed=SEED, fixtures=rows, worst_relative_error=worst,
                  cost=cost, target_access=False, wide_network_execution=False,
                  torch_imported=False, wall_seconds=time.time()-start,
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  qualification='Exact coordinate-star identity, NOT a faithful reconstruction of '
                  'the omitted all-distinct third tensor. Even memory is latent, not kappa3. '
                  'This certificate does not reopen the closed standalone coordinate-star estimator.')
    with OUT.open('x') as f:
        json.dump(record, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(passed=record['passed'], worst_relative_error=worst,
                          cost=cost, output=str(OUT)), indent=2), flush=True)
    if not record['passed']:
        raise RuntimeError('Tiny fixed identity certificate failed')


if __name__ == '__main__':
    main()
