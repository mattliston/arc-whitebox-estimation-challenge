"""Inspect only sealed tiny output/cotangent arrays; no Torch/cards/targets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PINS = {
    HERE/'stage_cpu_forward_v1/result.json': '6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607',
    HERE/'stage_mps_forward_v1/result.json': '91d1f3b56d4ff6f365b6216e3832226ccf71913d4b5618fb8697e4906f1cae7d',
}


def sha(path):
    path = Path(path).resolve()
    assert path.is_relative_to(HERE), 'Only current saved diagnostic artifacts'
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    pins = {str(path): digest for path, digest in PINS.items()}
    for path, digest in pins.items():
        assert sha(path) == digest
    records = {key: json.loads((HERE/('stage_'+key+'_forward_v1')/'result.json').read_text())
               for key in ('cpu', 'mps')}
    assert records['cpu']['passed'] and not records['mps']['passed']
    assert all(record['alltruth_accessed'] is False for record in records.values())
    assert records['mps']['error'] == "RuntimeError('Actual-FP32 finite-direction backend gate')"
    for key in ('cpu', 'mps'):
        seal = HERE/('stage_'+key+'_forward_v1')/'completion.json'
        pins[str(seal)] = sha(seal)
        assert json.loads(seal.read_text())['result_sha256'] == PINS[seal.with_name('result.json')]
    out = HERE/'saved_scalar_audit_v1'
    out.mkdir(exist_ok=False)
    import numpy as np
    assert not any(key in sys.modules for key in ('torch', 'flopscope', 'threadpoolctl'))

    def array(item):
        path = Path(item['path']).resolve()
        assert path.name != 'weights.npy' and sha(path) == item['sha256']
        pins[str(path)] = item['sha256']
        value = np.load(path, allow_pickle=False)
        assert value.shape == (16, 1024) and value.dtype == np.float32 and np.isfinite(value).all()
        assert hashlib.sha256(value.tobytes()).hexdigest() == item['raw_sha256']
        return value.astype(np.float64)

    cot = array(records['cpu']['cotangents']['35'])
    def common(value):
        return math.fsum((value[-1]*cot[-1]).ravel())/math.sqrt(1024) + \
            .05*math.fsum((value[:-1]*cot[:-1]).ravel())/math.sqrt(15*1024)

    predictions, scalars, cases = {}, {}, []
    for key in ('cpu', 'mps'):
        predictions[key], scalars[key] = {}, {}
        for case in ('null', 'plus025', 'minus025', 'plus050', 'minus050'):
            entry = records[key]['outputs']['35'][case]
            values = array(entry['prediction'])
            saved = entry['synthetic_loss']
            current = common(values)
            predictions[key][case], scalars[key][case] = values, current
            cases.append(dict(backend=key, case=case, original_scalar=saved,
                              common_fsum_scalar=current, original_minus_common=saved-current))
    amplitudes = []
    for suffix, amplitude in (('025', .025), ('050', .05)):
        ds, original, vectors = {}, {}, {}
        for key in ('cpu', 'mps'):
            ds[key] = (scalars[key]['plus'+suffix] - scalars[key]['minus'+suffix])/(2*amplitude)
            original[key] = next(x['symmetric_secant'] for x in records[key]['secants']
                                 if x['rank'] == 35 and x['amplitude'] == amplitude)
            vectors[key] = (predictions[key]['plus'+suffix] - predictions[key]['minus'+suffix])/(2*amplitude)
        original_gap = original['mps'] - original['cpu']
        primal_gap = ds['mps'] - ds['cpu']
        rounding_gap = (original['mps']-ds['mps']) - (original['cpu']-ds['cpu'])
        assert abs(original_gap-primal_gap-rounding_gap) <= 1e-14
        delta = vectors['mps'] - vectors['cpu']
        norm = max(float(np.linalg.norm(vectors[key])) for key in ('cpu', 'mps'))
        amplitudes.append(dict(amplitude=amplitude, original_secants=original,
            common_fsum_secants=ds, original_relative_gap=abs(original_gap)/max(map(abs, original.values())),
            common_relative_gap=abs(primal_gap)/max(map(abs, ds.values())),
            signed_original_gap=original_gap, signed_primal_component=primal_gap,
            signed_scalar_evaluation_component=rounding_gap,
            vector_secant_relative_l2=float(np.linalg.norm(delta))/norm,
            layer_vector_relative_l2=[float(np.linalg.norm(delta[l]))/
                max(float(np.linalg.norm(vectors['cpu'][l])), float(np.linalg.norm(vectors['mps'][l])),1e-300)
                for l in range(16)]))
    selectors = {}
    for case in ('null', 'plus025', 'minus025', 'plus050', 'minus050'):
        a = records['cpu']['outputs']['35'][case]['diagnostics']['selection_cells']
        b = records['mps']['outputs']['35'][case]['diagnostics']['selection_cells']
        assert set(a) == set(b)
        selectors[case] = dict(membership_changed_cells=[key for key in a if a[key]['membership'] != b[key]['membership']],
                              order_changed_cells=[key for key in a if a[key]['ordered_ids'] != b[key]['ordered_ids']])
    pins[str(Path(__file__).resolve())] = sha(__file__)
    pins[str(HERE/'SAVED_SCALAR_AUDIT_PLAN_V1.md')] = sha(HERE/'SAVED_SCALAR_AUDIT_PLAN_V1.md')
    for path, digest in pins.items():
        assert sha(path) == digest
    result = dict(completed=True, passed=True, diagnostic_only=True,
        original_qualification_still_failed=True, fit_admitted=False, truths_accessed=False,
        card_calls=0, gradient_calls=0, source_bindings=pins, rank=35, cases=cases,
        amplitudes=amplitudes, selectors=selectors, numpy_version=np.__version__,
        created_utc=datetime.now(timezone.utc).isoformat())
    write(out/'result.json', result)
    write(out/'completion.json', dict(completed=True, passed=True, result_sha256=sha(out/'result.json')))
    print(json.dumps(dict(amplitudes=amplitudes, selectors=selectors), sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
