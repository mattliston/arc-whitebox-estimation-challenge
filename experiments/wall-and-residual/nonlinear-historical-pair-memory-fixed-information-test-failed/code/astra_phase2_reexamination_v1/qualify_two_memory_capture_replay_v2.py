"""Fixed v1/v2 resource replay qualification; no predictive model is evaluated."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
PINS = {
    'capture_two_memory_teacher_action_v1.py': '4eb6a294651800c137614cbc1ad8a1bab99a206ad220d76e86f74b184056d608',
    'capture_two_memory_teacher_action_replay_v2.py': '82103b513f517cd555803260dd6f2c54aeeeb07a3a9e55254b12162baf646d7e',
    'TWO_MEMORY_RESOURCE_REPLAY_V2.md': 'f32c9e8097d1845b4882930f9d45ae8aaab89499d9f4493f964123a763a05d4c',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    modes = {'--execute-first': ([35], 'two_memory_capture_replay_v2_rank35_qualification.json'),
             '--execute-three': ([35, 36, 37], 'two_memory_capture_replay_v2_qualification.json')}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        print('Inert. Requires --execute-first or --execute-three after source review.')
        return
    ranks, output = modes[sys.argv[1]]
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Launch nice >=10')
    require('/anaconda3/' not in sys.executable, 'Use NumPy-only repo interpreter')
    require(not (HERE/output).exists(), 'Preserve existing qualification receipt')
    source_hashes = {name: sha(HERE/name) for name in [*PINS, Path(__file__).name]}
    require(all(source_hashes[k] == v for k, v in PINS.items()), 'Frozen source changed')
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits
    started = time.time()
    checked = {}
    rows = []
    with threadpool_limits(limits=2):
        for rank in ranks:
            roots = [HERE/f'rows_two_memory_teacher_action_v{version}'/f'row_{rank:04d}'
                     for version in (1, 2)]
            receipts = []
            for root in roots:
                path = root/'result.json'
                checked[str(path.relative_to(HERE))] = sha(path)
                receipt = json.loads(path.read_text())
                require(receipt['passed'] and receipt['rank'] == rank,
                        f'Capture numerical fidelity failed: {root}')
                for name, digest in receipt['source_hashes'].items():
                    require(sha(HERE/name) == digest, f'Capture source changed: {name}')
                receipts.append(receipt)
            require(receipts[1]['peak_rss_bytes'] <= 6*(1 << 30), 'v2 exceeds fixed RSS cap')
            require(receipts[0]['input_hashes'] == receipts[1]['input_hashes'], 'Inputs differ')
            require(set(receipts[0]['artifacts']) == set(receipts[1]['artifacts']), 'Artifacts differ')
            comparisons = []
            for name in sorted(receipts[0]['artifacts']):
                paths = [root/name for root in roots]
                for p, receipt in zip(paths, receipts):
                    digest = sha(p)
                    require(digest == receipt['artifacts'][name]['sha256'], 'Captured artifact changed')
                    checked[str(p.relative_to(HERE))] = digest
                with np.load(paths[0], allow_pickle=False) as old, np.load(paths[1], allow_pickle=False) as new:
                    require(set(old.files) == set(new.files), 'Array keys differ')
                    for key in sorted(old.files):
                        a, b = old[key], new[key]
                        require(a.shape == b.shape and a.dtype == b.dtype, 'Shape/dtype changed')
                        require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite parity input')
                        if name.startswith('post_') and key == 'physical_K21':
                            relative = float(np.linalg.norm(a-b)/max(1., float(np.linalg.norm(a))))
                            require(relative <= 1e-10, 'Blocked physical plane parity failed')
                            comparisons.append(dict(artifact=name, array=key, relative_l2=relative))
                        else:
                            require(a.tobytes() == b.tobytes(), f'Bank array byte parity failed: {name}/{key}')
                            comparisons.append(dict(artifact=name, array=key, byte_equal=True))
                        del a, b
            rows.append(dict(rank=rank, v1_receipt_sha256=sha(roots[0]/'result.json'),
                             v2_receipt_sha256=sha(roots[1]/'result.json'),
                             v1_peak_rss_bytes=receipts[0]['peak_rss_bytes'],
                             v2_peak_rss_bytes=receipts[1]['peak_rss_bytes'], comparisons=comparisons))
        backend = threadpool_info()
    require(all(sha(HERE/name) == digest for name, digest in checked.items()), 'Artifact changed while checking')
    require(all(sha(HERE/name) == digest for name, digest in source_hashes.items()), 'Source changed while checking')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 6*(1 << 30), 'Qualification RSS cap exceeded')
    result = dict(passed=True, ranks=ranks, rows=rows, source_hashes=source_hashes,
                  checked_files=checked, arrays_read_for_numerical_parity_only=True,
                  predictive_features_evaluated=False, current_action_loss_evaluated=False,
                  requested_blas_threads=2, backend=backend,
                  effective_threads_independently_verified=bool(backend),
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  wall_seconds=time.time()-started, utc_finished=datetime.now(timezone.utc).isoformat())
    with (HERE/output).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    worst = max(c.get('relative_l2', 0.) for row in rows for c in row['comparisons'])
    print(json.dumps(dict(passed=True, ranks=ranks, worst_post_plane_relative_l2=worst,
                          bank_arrays_byte_identical=True, output=output,
                          wall_seconds=result['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
