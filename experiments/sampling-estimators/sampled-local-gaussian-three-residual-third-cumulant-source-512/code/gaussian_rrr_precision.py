"""Research-only local Gaussian RRR variance; never imported by an estimator."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
CAPTURE = P2 / 'astra_phase2_reexamination_v1/current_comp_joint_cell_attribution_v1/capture_v1'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
OUT = HERE / 'gaussian_rrr_precision.json'


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not OUT.exists()
    started = time.monotonic()
    metadata = json.loads((CAPTURE / 'restart_packet.json').read_text())
    assert metadata['candidate_sha256'] == 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
    assert metadata['rank'] == 35 and metadata['layer'] == 8 and not metadata['truth_accessed']
    spans = {}

    def read_span(packet, key, aux=False):
        item = metadata['packets'][packet]
        if aux:
            item = item['aux']['mapping']
        desc = item[key]['array']
        with (CAPTURE / 'restart_packet.bin').open('rb') as f:
            f.seek(desc['offset'])
            blob = f.read(desc['bytes'])
        assert hashlib.sha256(blob).hexdigest() == desc['sha256']
        array = np.frombuffer(blob, dtype=desc['dtype']).reshape(desc['shape']).copy()
        assert np.isfinite(array).all()
        spans[f'{packet}.{key}'] = desc['sha256']
        return array

    mu = read_span('pre8', 'mu')
    cov = read_span('pre8', 'Cpre')
    alive = read_span('pre8', 'alive')
    saved_sigma = read_span('pair8', 'sigma', True)
    saved_gate = read_span('pair8', 'gate', True)
    saved_mean = read_span('pair8', 'gaussian_mean', True)
    alpha9 = read_span('pair9', 'alpha', True)
    pdf9 = read_span('pair9', 'pdf', True)
    sigma9 = read_span('pair9', 'sigma', True)
    manifest = json.loads((STORE / 'manifest.json').read_text())
    wp = (STORE / manifest['roles']['observer_fit']['row_manifests'][35]['path']).parent / 'weights.npy'
    weights_sha = file_sha(wp)
    assert weights_sha == '14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2'
    weights = np.load(wp, mmap_mode='r')
    assert weights.shape == (16, 1024, 1024)
    q = weights[9].astype(np.float64)
    q2, q3 = q * q, q * q * q
    asym = float(np.max(np.abs(cov - cov.T)))
    cov = (cov + cov.T) * .5
    sigma = np.sqrt(np.diag(cov))
    assert np.all(sigma > 0) and np.all(sigma9 > 0)
    alpha = mu / sigma
    gate = .5 * (1 + np.array([math.erf(x / math.sqrt(2)) for x in alpha]))
    pdf = np.exp(-alpha**2 / 2) / math.sqrt(2 * math.pi)
    mg = mu * gate + sigma * pdf
    response = -alpha9 * pdf9 / (6 * sigma9**2)
    n = mu.size
    rec = dict(schema='local_gaussian_rrr_precision_v1', research_only=True,
               source_sha256=file_sha(Path(__file__)),
               protocol_sha256=file_sha(HERE / 'GAUSSIAN_RRR_PRECISION_PROTOCOL.md'),
               capture_metadata_sha256=file_sha(CAPTURE / 'restart_packet.json'),
               source_candidate_sha256=metadata['candidate_sha256'],
               selected_span_sha256=spans, weights_sha256=weights_sha,
               rank=35, layer=8, consumer=9, truth_accessed=False,
               covariance_max_asymmetry=asym, covariance_diagonal_min=float(sigma.min()**2),
               alive_count=int(alive.sum()),
               marginal_max_differences=dict(sigma=float(np.max(np.abs(sigma-saved_sigma))),
                   gate=float(np.max(np.abs(gate-saved_gate))), mean=float(np.max(np.abs(mg-saved_mean)))),
               pairs_per_batch=512, primary_gate_variance=1e-10,
               complete_deployment_bill=False, batches=[])
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        rec.update(covariance_positive_definite=False, necessary_precision_gate_passed=False,
                   conclusion='Captured covariance has no positive-definite Gaussian law without a repair; no sampling performed.')
    else:
        rec['covariance_positive_definite'] = True
        rec['cholesky_min_diagonal'] = float(np.diag(chol).min())
        means = []
        for seed in (1909198, 1909199):
            z = np.random.default_rng(seed).standard_normal((512, n)) @ chol.T
            outputs = []
            for sign in (1., -1.):
                dz = sign * z
                r = (np.maximum(mu + dz, 0) - mg - gate * dz) * alive
                qr = r @ q.T
                qr2 = (r*r) @ q2.T
                qr3 = (r*r*r) @ q3.T
                y = (qr*qr*qr - 3*qr*qr2 + 2*qr3) * response
                assert np.isfinite(y).all()
                outputs.append(y)
            paired = (outputs[0] + outputs[1]) * .5
            mean = np.mean(paired, axis=0)
            pair_variance = float(np.mean(np.var(paired, axis=0, ddof=1)))
            iid_variance = float(np.mean(np.var(outputs[0], axis=0, ddof=1)))
            means.append(mean)
            rec['batches'].append(dict(seed=seed, pair_variance=pair_variance,
                mean_variance_512_pairs=pair_variance/512,
                iid_mean_variance_512_draws=iid_variance/512,
                measured_source_mean_square=float(np.mean(mean*mean)),
                precision_gate_passed=bool(pair_variance/512 <= 1e-10)))
        rec['two_batch_difference_mean_square'] = float(np.mean((means[0]-means[1])**2))
        rec['two_batch_expected_noise_mean_square'] = sum(b['mean_variance_512_pairs'] for b in rec['batches'])
        rec['independent_batch_cross_mean'] = float(np.mean(means[0]*means[1]))
        rec['necessary_precision_gate_passed'] = all(b['precision_gate_passed'] for b in rec['batches'])
        # Gaussian acquisition once and three dense projections for each sign.
        # Does not include elementary work, RNG, carrying the CP source, or future consumers.
        rec['optimistic_dense_matrix_flops_per_512_pair_cell'] = (n**3/3 + 14*512*n*n)
        rec['optimistic_15_cell_matrix_flops'] = 15*rec['optimistic_dense_matrix_flops_per_512_pair_cell']
        rec['conclusion'] = ('Necessary local precision passes; ownership, signed accuracy and complete cost remain unqualified.'
            if rec['necessary_precision_gate_passed'] else
            'The fixed512-pair local Gaussian RRR constructor fails its necessary precision gate; no expansion or tuning.')
    rec['elapsed_s'] = time.monotonic() - started
    with OUT.open('x') as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
