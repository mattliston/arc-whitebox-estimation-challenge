"""One fixed input-only Dirichlet row; no truth paths or bank mutations."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from threadpoolctl import threadpool_limits

from dirichlet_gram_observer_v1 import observer, final_features

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    rank = parser.parse_args().rank
    assert rank == 100 or 35 <= rank < 59
    out = HERE / 'rows_dirichlet_gram_probe_v1' / f'row_{rank:04d}'
    tape = HERE / 'rows_current_transport_shadow_v2' / f'row_{rank:04d}'
    bank_dir = HERE / 'rows_matrix_error_observer_v1' / f'row_{rank:04d}'
    paths = [
        HERE / 'dirichlet_gram_observer_v1.py', Path(__file__).resolve(),
        HERE / 'DIRICHLET_GRAM_INFORMATION_DESIGN_V1.md',
        HERE / 'DIRICHLET_GRAM_ROOT_ACCEPTANCE_V1.md',
        HERE / 'dirichlet_gram_small_certificate_v1.json',
        HERE / 'payload_fullpair_final_head_v1.npz',
        bank_dir / 'result.json', bank_dir / 'final.npz',
        tape / 'result.json', tape / 'weights.npy', tape / 'pre_cov.npy',
        tape / 'x.npy', tape / 'sigma.npy',
    ]
    before = {str(p.relative_to(HERE)): digest(p) for p in paths}
    if out.exists():
        record = json.loads((out / 'result.json').read_text())
        assert record['passed'] and record['input_source_sha256'] == before
        assert digest(out / 'data.npz') == record['data_sha256']
        print(json.dumps(dict(rank=rank, existing_sealed_row=True, passed=True)), flush=True)
        return
    certificate = json.loads((HERE / 'dirichlet_gram_small_certificate_v1.json').read_text())
    assert certificate['passed']
    assert before['dirichlet_gram_observer_v1.py'] == certificate['source_sha256']
    assert before['DIRICHLET_GRAM_INFORMATION_DESIGN_V1.md'] == '1ddd561a782b19cdb81c95c4c3d92960b16e311114d7118a9ed061de7e9d302c'
    bank_record = json.loads((bank_dir / 'result.json').read_text())
    assert bank_record['passed'] and bank_record['target_access'] is False
    assert all(bank_record['prefix_feature_sigma_byte_parity'])
    assert all(bank_record['existing_current_scalar_final_byte_parity'].values())
    assert json.loads((tape / 'result.json').read_text())['target_access'] is False
    out.mkdir(parents=True)
    started = time.time()
    try:
        with np.load(bank_dir / 'final.npz') as archive:
            bank = {key: archive[key].copy() for key in archive.files}
        assert set(bank) == {'x', 'sigma', 'host', 'delta', 'h', 'prediction'}
        assert all(np.all(np.isfinite(value)) for value in bank.values())
        weights = np.load(tape / 'weights.npy', mmap_mode='r')
        pre_cov = np.load(tape / 'pre_cov.npy', mmap_mode='r')
        raw_features = np.load(tape / 'x.npy', mmap_mode='r')
        sigmas = np.load(tape / 'sigma.npy', mmap_mode='r')
        with threadpool_limits(limits=2):
            pre_gram, history = observer(weights, pre_cov, raw_features, sigmas)
            sources = final_features(pre_gram, pre_cov[-1], bank)
            deterministic = None
            if rank in (35, 100):
                repeated, repeated_history = observer(weights, pre_cov, raw_features, sigmas)
                repeated_sources = final_features(repeated, pre_cov[-1], bank)
                deterministic = bool(np.array_equal(pre_gram, repeated)
                                     and np.array_equal(sources, repeated_sources)
                                     and history == repeated_history)
                assert deterministic
        saved = {key: value.copy() for key, value in bank.items()}
        saved['sources'] = sources
        parity = {key: bool(np.array_equal(bank[key], saved[key])) for key in bank}
        assert all(parity.values())
        np.savez(out / 'data.npz', **saved)
        after = {str(p.relative_to(HERE)): digest(p) for p in paths}
        assert after == before
        record = dict(
            rank=rank, passed=True, target_access=False, byte_parity=parity,
            deterministic_replay=deterministic, history=history,
            feature_mean=sources.mean(axis=0).tolist(),
            feature_std=sources.std(axis=0).tolist(),
            source_dtypes=dict(prefix_sigma=str(sigmas.dtype), prefix_cov=str(pre_cov.dtype),
                               final_sigma=str(bank['sigma'].dtype), final_host=str(bank['host'].dtype)),
            wall_s=time.time() - started,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            input_source_sha256=before, inputs_unchanged=True,
            data_sha256=digest(out / 'data.npz'),
            qualification='Input-only approximate observer; no true deep Jacobian, wide PSD, native bill, or accuracy claim.',
        )
        (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({key: value for key, value in record.items()
                          if key not in ('history', 'input_source_sha256')}, indent=2), flush=True)
    except BaseException as exc:
        failure = dict(rank=rank, passed=False, target_access=False,
                       error_type=type(exc).__name__, error=str(exc),
                       wall_s=time.time() - started, input_source_sha256=before)
        (out / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
