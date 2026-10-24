"""Offline target-free physical-product test; no predictor and no truth access.

Compare a deployed-form row-sampled range finder with the best Dv row subspace
and the best physical output subspace. Oracle spectra are diagnostics only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def norm2(x):
    return float(np.sum(np.square(x, dtype=np.float64), dtype=np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rank', type=int, required=True, choices=(35, 150))
    a = ap.parse_args()
    assert os.environ.get('ARC_BLAS_THREADS') in ('1', '2', '4')
    a.out.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    candidates = sorted(a.capture.glob('part*/old_dv_capture_v1/CAPTURE_COMPLETE_v1.json'))
    assert candidates
    source = '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
    records = []
    for receipt in candidates:
        done = json.loads(receipt.read_text())
        assert done['complete'] and done['target_free'] and done['source_sha256'] == source
        if a.rank not in done['ranks']:
            continue
        for rec in done['files']:
            path = receipt.parent / rec['path']
            assert sha(path) == rec['sha256']
            x = json.loads(path.read_text())
            if x['rank'] == a.rank:
                records.append((path, x))
    assert {x['layer'] for _, x in records} == set(range(7, 15))
    reports = []
    for path, rec in records:
        def read(key):
            r = rec[key]
            p = path.parent / r['path']
            assert p.stat().st_size == r['bytes'] and sha(p) == r['sha256']
            x = np.fromfile(p, dtype=np.dtype(r['dtype'])).reshape(r['shape'])
            assert np.isfinite(x).all()
            return x
        D = read('Dv').astype(np.float64)
        F = read('FvT').astype(np.float64)
        native = read('native_inner').astype(np.float64)
        U = read('pre_basis').astype(np.float64)
        live = read('live_indices').astype(np.int64)
        R = U[live].T
        if 'lazy_map' in rec:
            R = read('lazy_map').astype(np.float64).T @ R
        assert D.shape[1] == F.shape[0] and F.shape[1] == native.shape[1] == R.shape[0]
        I = D @ F
        m = D.mean(axis=0, keepdims=True)
        A = D - m
        Im = m @ F
        B = I - Im
        RR = R @ R.T
        rv, rq = np.linalg.eigh((RR + RR.T) * .5)
        Rroot = rq * np.sqrt(np.maximum(rv, 0))[None, :]
        physical = I @ Rroot
        phys_den = max(norm2(physical), np.finfo(float).tiny)
        inner_den = max(norm2(I), np.finfo(float).tiny)
        def error(estimate):
            delta = estimate - I
            return dict(inner_relative_l2=float(np.sqrt(norm2(delta) / inner_den)),
                        physical_relative_l2=float(np.sqrt(norm2(delta @ Rroot) / phys_den)))
        # Best Dv rank is diagnostic only; constructing this Gram is not proposed
        # for deployment. The sampled row QR below is the priced algorithm.
        G = A @ A.T
        ev, eu = np.linalg.eigh((G + G.T) * .5)
        order = np.argsort(ev)[::-1]
        ev = np.maximum(ev[order], 0)
        eu = eu[:, order]
        _, ps, _ = np.linalg.svd(B @ Rroot, full_matrices=False)
        total_d = max(norm2(D), np.finfo(float).tiny)
        full_dense_bill = 2 * D.shape[0] * D.shape[1] * F.shape[1]
        row_report = dict(rank=a.rank, layer=rec['layer'], part=rec['part'],
                          input_shapes=dict(Dv=list(D.shape), FvT=list(F.shape), R=list(R.shape)),
                          Dv_min=float(D.min()), Dv_max=float(D.max()),
                          Dv_negative_fraction=float(np.mean(D < 0)),
                          mean_fraction=1 - norm2(A) / total_d,
                          native_vs_exact=error(native), exact_mean_only=error(np.broadcast_to(Im, I.shape)),
                          mean_bill_dense=int(D.shape[0] * D.shape[1] + 2 * D.shape[1] * F.shape[1]),
                          full_dense_bill=int(full_dense_bill), rank_results=[])
        for k in (4, 8, 16, 32, 64, 128):
            if k >= min(A.shape):
                continue
            # Uniform response rows: fixed by dimensions, never target outcomes.
            rows = np.linspace(0, A.shape[0] - 1, k, dtype=np.int64)
            assert len(np.unique(rows)) == k
            Q, qrR = np.linalg.qr(A[rows].T, mode='reduced')
            C = A @ Q
            projectedF = Q.T @ F
            sampled = Im + C @ projectedF
            oracle_d = Im + eu[:, :k] @ (eu[:, :k].T @ B)
            dense = (D.shape[0] * D.shape[1] * 2
                     + 2 * D.shape[1] * F.shape[1]
                     + 2 * D.shape[1] * k * k - 2 * k ** 3 / 3
                     + 2 * D.shape[0] * D.shape[1] * k
                     + 2 * k * D.shape[1] * F.shape[1]
                     + 2 * D.shape[0] * k * F.shape[1]
                     + D.shape[0] * F.shape[1])
            row_report['rank_results'].append(dict(
                rank=k, sampled=error(sampled), best_Dv_subspace=error(oracle_d),
                oracle_physical_relative_l2=float(np.sqrt(np.sum(ps[k:] ** 2) / phys_den)),
                Dv_centered_tail_fraction=float(ev[k:].sum() / max(ev.sum(), 1e-300)),
                sample_R_condition=float(np.linalg.cond(qrR)),
                arithmetic_dense_estimate=int(dense), arithmetic_dense_ratio=float(dense / full_dense_bill),
                tariff_warning='Unmetered algebraic counts; incumbent uses fast kernels. Not a score estimate.'))
        for k in (224, 256, 320):
            if k < I.shape[1]:
                cut = I.copy()
                cut[:, k:] = 0
                row_report['trailing_coordinate_cut_%d' % k] = error(cut)
        reports.append(row_report)
        out = a.out / (path.stem + '_ANALYSIS.json')
        with out.open('x') as f:
            json.dump(row_report, f, indent=2, allow_nan=False)
            f.write('\n')
        print(json.dumps(dict(layer=rec['layer'], part=rec['part'],
                              mean_fraction=row_report['mean_fraction'],
                              k16=next(x for x in row_report['rank_results'] if x['rank'] == 16)['sampled'])), flush=True)
        del D, F, native, U, live, R, I, m, A, Im, B, RR, rv, rq, Rroot, physical, G, ev, eu, ps
    result = dict(utc=utc(), status='TARGET_FREE_PHYSICAL_PRODUCT_DIAGNOSTIC_COMPLETE',
                  source_sha256=source, analysis_sha256=sha(__file__), rank=a.rank,
                  native_launches=0, truth_access=False, score_gain_claim=False,
                  elapsed_s=time.monotonic() - begin, records=reports)
    with (a.out / 'RESULT_v1.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
