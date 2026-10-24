"""Additive diagnostic-only norm scaling repair; first-jet arithmetic frozen."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import certify_pair_jet_coalescence_v1 as frozen

HERE = Path(__file__).resolve().parent
OUT = HERE/'pair_jet_coalescence_tiny_cost_v2.json'


def main():
    assert not OUT.exists()
    start = time.time()
    certificate = frozen.tiny()
    certificate['original_v1_composite_passed'] = certificate['passed']
    for row in certificate['records']:
        ratio = row['relative_tensor_error']
        error_squared_scale = ratio*ratio
        row['error_norm_relative_to_error_squared'] = (
            row['error_norm_error']/error_squared_scale
            if error_squared_scale else row['error_norm_error'])
        row['eligible_first_jet'] = ratio <= .1
        norm_pass = (row['error_norm_error'] <= 1e-11 if row['eligible_first_jet']
                     else row['error_norm_relative_to_error_squared'] <= 1e-11)
        identity_pass = all(row[k] <= 1e-11 for k in
                            ('tensor_identity_error', 'tensor_norm_error',
                             'repeated_plane_error', 'next_plane_error',
                             'restored_next_plane_error'))
        row['passed'] = bool(norm_pass and identity_pass)
    certificate['passed'] = all(r['passed'] for r in certificate['records'])
    assert certificate['passed'], certificate
    projection = frozen.cost()
    result = dict(passed=certificate['passed'] and projection['passed'], tiny=certificate,
                  cost=projection, target_access=False, wide_execution=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  frozen_v1_source_sha256=hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest(),
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.time()-start,
                  peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  qualification='Root-approved additive v2 normalization repair only. Original failed v1 retained. Exact same fixtures, first-jet arithmetic, .1 eligibility and one static cost schedule. For eligible pairs the original error-norm-to-T gate remains; huge ineligible error norms checked relative to themselves. No network execution or score.')
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'tiny':{'passed':certificate['passed']},
                     'cost':{**projection,'rows':[{k:v for k,v in r.items() if k not in ('cohorts','original_layers','reduced_layers')} for r in projection['rows']]}},indent=2))


if __name__ == '__main__':
    main()
