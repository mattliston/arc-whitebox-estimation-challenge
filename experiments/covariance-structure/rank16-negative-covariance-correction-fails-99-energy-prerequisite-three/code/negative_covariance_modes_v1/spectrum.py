import hashlib
import resource
import time

from common import *
from geometry import negative_modes


def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads'] == 1 for x in threadpool_info())
    verify();assert read(HERE/'unit.json')['passed']
    rows = [];started = time.monotonic()
    for rec in captures():
        rank = rec['rank'];payload = {};layers = []
        with np.load(rec['capture']['path'], allow_pickle=False) as z:
            for layer in LAYERS:
                key = f'C_post_{layer:02d}';C = z[key]
                assert hashlib.sha256(C.tobytes()).hexdigest() == rec['array_digests'][key]['sha256']
                values, Q, d, info = negative_modes(np, C.astype(np.float64))
                payload[f'values_{layer}'] = values;payload[f'Q_{layer}'] = Q;payload[f'd_{layer}'] = d
                row = dict(layer=layer, **info);layers.append(row)
                print(json.dumps(dict(rank=rank, **row)), flush=True)
        dest = HERE/f'modes_{rank:04d}.npz';np.savez(dest, **payload)
        rows.append(dict(rank=rank, layers=layers, file=pin(dest), capture=rec['capture'],
                         passed=all(x['spectral_gate_passed'] for x in layers)))
    result = dict(completed=True, passed=all(r['passed'] for r in rows), rows=rows,
                  target_arrays_opened=False, candidate_created=False, wall_s=time.monotonic()-started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'spectrum_result.json', result)
    print(json.dumps(dict(spectrum_passed=result['passed'])), flush=True)


if __name__ == '__main__':
    main()
