"""Pure analytic kernel audit; reads archived measurements, never reruns them."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'public_quadrature_derivation_v1'
OUT = HERE/'nngp_centering_audit_v1.json'
assert not OUT.exists()
started = time.time()


def raw(t):
    return (np.sqrt(1-t*t)+(np.pi-np.arccos(t))*t)/np.pi


def cauchy(radius, grid):
    z = radius*np.exp(2j*np.pi*np.arange(grid)/grid)
    rows = []
    c = 0.
    for l in range(1, 17):
        z = raw(z)
        c = float(raw(c))
        coeff = np.fft.fft((z-c)/(1-c))/grid
        rows.append(coeff[:65]/radius**np.arange(65))
    return np.array(rows)


a = cauchy(.85, 2048)
b = cauchy(.90, 4096)
maximum_difference = float(np.max(np.abs(a-b)))
maximum_imaginary = float(np.max(np.abs(b.imag)))
old = json.loads((OLD/'theory_branching.json').read_text())
old_curves = json.loads((OLD/'curve_v2.json').read_text())
verification = json.loads((OLD/'quad_verify.json').read_text())['e1']
c, derivative = 0., 1.
records = []
for l in range(1, 17):
    derivative *= 1-np.arccos(c)/np.pi
    c = float(raw(c))
    linear = float(derivative/(1-c))
    coeff = b[l-1].real
    rows = dict(depth=l, raw_constant=c, normalized_linear_mass=linear,
                cauchy_linear_mass=float(coeff[1]),
                chain_cauchy_difference=abs(linear-float(coeff[1])),
                normalized_mean_degree=1/(1-c),
                cumulative_degree_mass={str(k): float(coeff[1:k+1].sum()) for k in (1, 2, 4, 8, 16, 32, 64)},
                wrong_iterated_centered_linear=old['degree_table'][str(l)]['P_deg_eq_1'])
    if l <= 6:
        rows.update(archived_n64_ensemble_linear=verification['ensemble_mean_linear_mass'][l-1],
                    archived_n64_ensemble_se=verification['ensemble_se'][l-1])
    records.append(rows)
# The archived schema is recorded explicitly rather than guessing a field.
curve_metadata = {k: v for k, v in old_curves.items()
                  if 'linear' in k or 'richardson' in k or 'm1' in k.lower()}
passed = (maximum_difference <= 1e-8 and maximum_imaginary <= 1e-8
          and max(r['chain_cauchy_difference'] for r in records) <= 1e-9
          and float(np.min(b.real[:, 1:])) >= -1e-8)
result = dict(passed=passed, target_access=False, sampling_runs=0,
              max_independent_cauchy_difference=maximum_difference,
              max_coefficient_imaginary=maximum_imaginary,
              records=records, existing_rank35_linear_measurement_fields=curve_metadata,
              archived_input_hashes={s: hashlib.sha256((OLD/s).read_bytes()).hexdigest() for s in
                                     ('theory_branching.py', 'theory_branching.json', 'curve_v2.json', 'quad_verify.json')},
              primary_sources=['https://arxiv.org/abs/1711.00165',
                               'https://github.com/google/neural-tangents/blob/main/neural_tangents/_src/stax/elementwise.py'],
              wall_s=time.time()-started,
              qualification='Corrected ideal raw-kernel composition followed by input centering. Not exact fixed-net/finite-input Gaussian Hermite masses. Old MC/CV/rules data unchanged; no new accuracy, cost or submission claim.')
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(passed=passed, max_independent_cauchy_difference=maximum_difference,
                      max_imaginary=maximum_imaginary, records=[records[0], records[5], records[-1]],
                      archived_rank35_linear=curve_metadata), indent=2), flush=True)
assert passed, result
