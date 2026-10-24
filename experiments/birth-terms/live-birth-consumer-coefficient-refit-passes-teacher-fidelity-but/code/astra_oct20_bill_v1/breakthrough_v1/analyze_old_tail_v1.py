"""Target-free old-read tail experiments; no estimator launch or score claim."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from analyze_old_dv_v3 import sha, utc, norm2

SOURCE = '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'


def records(capture, rank):
    name = 'SALVAGED_CAPTURE_v1.json' if rank == 35 else 'CAPTURE_COMPLETE_v1.json'
    matches = sorted(capture.glob('part*/old_dv_capture_v*/' + name))
    assert len(matches) == 1
    receipt = matches[0]
    done = json.loads(receipt.read_text())
    assert done['target_free'] and done['source_sha256'] == SOURCE and done['ranks'] == [rank]
    if rank == 35:
        assert done['diagnostic_only'] and done['resource_checks_passed']
        assert done['all_prediction_bytes_equal'] and done['traceback_only_wrong_finish_layer_assert']
    else:
        assert done['complete']
    result = []
    for item in done['files']:
        path = receipt.parent / item['path']
        assert sha(path) == item['sha256']
        rec = json.loads(path.read_text())
        assert rec['rank'] == rank
        result.append((path, rec))
    assert sorted((r['layer'], r['part']) for _, r in result) == [
        (10, 'materialized'), (11, 'materialized'), (12, 'materialized'),
        (13, 'base'), (13, 'join'), (14, 'base'), (14, 'join')]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, required=True, choices=(35, 150))
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get('ARC_BLAS_THREADS') == '2'
    a.out.mkdir(exist_ok=False, parents=True)
    start = time.monotonic()
    reports = []
    for path, rec in records(a.capture, a.rank):
        def read(key):
            info = rec[key]
            file = path.parent / info['path']
            assert file.stat().st_size == info['bytes'] and sha(file) == info['sha256']
            val = np.fromfile(file, dtype=np.dtype(info['dtype'])).reshape(info['shape'])
            assert np.isfinite(val).all()
            return val
        D = read('Dv').astype(np.float64)
        F = read('FvT').astype(np.float64)
        U = read('pre_basis').astype(np.float64)
        live = read('live_indices').astype(np.int64)
        R = U[live].T
        if 'lazy_map' in rec:
            R = read('lazy_map').astype(np.float64).T @ R
        I = D @ F
        Im = D.mean(axis=0, keepdims=True) @ F
        B = I - Im
        RR = R @ R.T
        val, vec = np.linalg.eigh((RR + RR.T) * .5)
        root = vec * np.sqrt(np.maximum(val, 0))[None, :]
        den = max(norm2(I @ root), 1e-300)
        rrdiag = np.maximum(np.diag(RR), 0)
        def error(delta):
            return float(np.sqrt(norm2(delta @ root) / den))
        report = dict(layer=rec['layer'], part=rec['part'], shape=list(D.shape),
                      tail_lowrank=[], selection=[])
        for keep in (224, 256, 288, 320, 352):
            Bt = B[:, keep:]
            Rt = root[keep:]
            s = np.linalg.svd(Bt @ Rt, compute_uv=False)
            mean_error = float(np.sqrt(norm2(Bt @ Rt) / den))
            item = dict(keep=keep, tail_width=I.shape[1] - keep,
                        zero_tail_error=float(np.sqrt(norm2(I[:, keep:] @ Rt) / den)),
                        exact_mean_tail_error=mean_error, approximations=[])
            for width in (4, 8, 16, 32):
                if width >= Bt.shape[1]:
                    continue
                rows = np.linspace(0, D.shape[0] - 1, width, dtype=np.int64)
                Q, _ = np.linalg.qr(Bt[rows].T, mode='reduced')
                delta = ((Bt @ Q) @ Q.T - Bt) @ Rt
                item['approximations'].append(dict(
                    width=width,
                    sampled_output_error=float(np.sqrt(norm2(delta) / den)),
                    oracle_physical_error=float(np.sqrt(np.sum(s[width:] ** 2) / den)),
                    extra_dense_arithmetic_vs_omitting_tail=int(
                        D.shape[0] * D.shape[1] + 2 * D.shape[1] * Bt.shape[1]
                        + 4 * width * D.shape[1] * Bt.shape[1]
                        + 2 * D.shape[0] * D.shape[1] * width
                        + 2 * D.shape[0] * width * Bt.shape[1]
                        + 2 * Bt.shape[1] * width * width)))
            report['tail_lowrank'].append(item)
            for center in (False, True):
                base = B if center else I
                scores = dict(
                    prefix=np.arange(I.shape[1], 0, -1, dtype=np.float64),
                    exact_column_energy=np.mean(base * base, axis=0) * rrdiag,
                    factor_energy=np.sum(F * F, axis=0) * rrdiag)
                for sample in (8, 32):
                    rows = np.linspace(0, D.shape[0] - 1, sample, dtype=np.int64)
                    scores['sample%d_column_energy' % sample] = np.mean(base[rows] ** 2, axis=0) * rrdiag
                for label, score in scores.items():
                    selected = np.sort(np.argsort(score, kind='stable')[-keep:])
                    omitted = np.setdiff1d(np.arange(I.shape[1]), selected)
                    delta = np.zeros_like(I)
                    delta[:, omitted] = -base[:, omitted]
                    report['selection'].append(dict(keep=keep, exact_omitted_mean=center,
                        method=label, physical_error=error(delta),
                        prefix_exchanges=int(np.count_nonzero(selected >= keep))))
        reports.append(report)
        print(json.dumps(dict(layer=rec['layer'], part=rec['part'],
            tails=[dict(keep=t['keep'], mean=t['exact_mean_tail_error'],
                        rank8=next((v['sampled_output_error'] for v in t['approximations'] if v['width'] == 8), None))
                   for t in report['tail_lowrank']])), flush=True)
    result = dict(utc=utc(), source_sha256=SOURCE, analysis_sha256=sha(__file__),
                  rank=a.rank, elapsed_s=time.monotonic() - start, records=reports,
                  truth_access=False, native_launches=0, score_gain_claim=False,
                  status='TARGET_FREE_TAIL_DIAGNOSTIC_COMPLETE')
    with (a.out / 'RESULT_v1.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
