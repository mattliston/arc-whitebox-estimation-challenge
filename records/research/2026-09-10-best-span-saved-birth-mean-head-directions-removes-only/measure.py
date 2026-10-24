"""One fixed saved-output projection diagnostic. No estimator or truth reads."""
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import struct
import sys
import time
import ast
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PINS = {
    'current_p1_node_composition_v1/score_v3/result.json': '17bd92bb65abb5d113cf3cd57df175b865f03295a7e0d96eadfeb133b6658390',
    'current_p1_node_composition_v1/score_v3/completion.json': '23d4e593af714034e7208a3a5947bef47386cf87a47dd5872c568a22c65415b4',
    'current_d50_native_objective_p1_v1/score2_v1/result.json': '4f60a8def29f1a9326d68d5c25bbe79b3f26289af8cf46de81609fb91cf055e5',
    'current_d50_native_objective_p1_v1/score2_v1/completion.json': 'c72831702f18a70a4bc40a8d88a655879be3e816e3edd3351cce9ea75e8cd587',
    'current_p1_node_composition_v1/native_terminal_v2/result.json': 'a9071e5afa6dbf35e6a6871d6997a7dbc8a1105d2bf3e97bd172c7a1b0e0180f',
    'current_p1_node_composition_v1/native_terminal_v2/completion.json': '6c050c04fd565d840273f2a917f2a106182cfed1b69f59a09cd5392497706772',
    'current_d50_native_objective_p1_v1/native2_terminal_v1/result.json': '7a1f37d80a7ed32850b10047408d85eed050220b4a6271dde90bdd21c07062e3',
    'current_d50_native_objective_p1_v1/native2_terminal_v1/completion.json': 'b73a957bf78e45da0db2741cea0b33a2cd98317726edaaa0954eaece054e5957',
    'birth_direction_fit_v1/validation_terminal_v2/result.json': 'ef7f4fcd8c6bc9275e434560ac5f514fd5f99d94077441d757ad34fb3a8df17d',
    'birth_direction_fit_v1/validation_terminal_v2/completion.json': 'fbbe37f97f39210f9e60a9dc3eeaa2d647529cfef875f72fb31942a7e724a079',
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def resolve(p):
    p = Path(p)
    p = p if p.is_absolute() else BASE / p
    p = p.resolve()
    if not p.is_relative_to(BASE):
        raise ValueError('input outside phase2 experiment root')
    return p


def checked(p, expected, bindings):
    p = resolve(p)
    if p.stat().st_size > 10_000_000 or sha(p) != expected:
        raise ValueError(f'input mismatch {p}')
    bindings[str(p)] = expected
    return p


def read_npy(p):
    data = p.read_bytes()
    if data[:8] != b'\x93NUMPY\x01\x00':
        raise ValueError('npy format')
    n = struct.unpack('<H', data[8:10])[0]
    head = ast.literal_eval(data[10:10+n].decode('ascii'))
    if head != {'descr': '<f4', 'fortran_order': False, 'shape': (16, 1024)}:
        raise ValueError('npy dtype/shape/order')
    body = data[10+n:]
    if len(body) != 16 * 1024 * 4:
        raise ValueError('npy size')
    vals = struct.unpack('<16384f', body)
    if not all(math.isfinite(x) and x >= 0 for x in vals):
        raise ValueError('output domain')
    return vals


def guard():
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 512 * 2**20:
        raise MemoryError('512 MiB RSS ceiling')


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b)) / 1024


def one(base, linear, gram):
    if not (base > 0 and gram > 0):
        raise ValueError('invalid quadratic')
    alpha = -linear / gram
    gain = linear * linear / gram
    if not 0 <= gain <= base * (1 + 1e-10):
        raise ValueError('projection violates Cauchy-Schwarz')
    return {'base_error': base, 'oracle_alpha_not_deployable': alpha,
            'optimistic_minimum_error': base - gain,
            'fractional_error_reduction': gain / base}


def aggregate(rows, weighted):
    def avg(key):
        return math.fsum(r[key] * (r['reference_multiplier'] if weighted else 1)
                         for r in rows) / len(rows)
    e0, e1, e2 = (avg(k) for k in ('d50_mse', 'p1_mse', 'comp_mse'))
    a, b, c = (avg(k) for k in ('g11', 'g12', 'g22'))
    q1, q2 = (avg(k) for k in ('e0_d1', 'e0_d2'))
    det = a*c - b*b
    if not det > 1e-8 * a*c:
        raise ValueError('ill-conditioned two-direction span')
    x1 = (-c*q1 + b*q2) / det
    x2 = (b*q1 - a*q2) / det
    gain = -(q1*x1 + q2*x2)
    if not 0 <= gain <= e0 * (1 + 1e-10):
        raise ValueError('span violates Cauchy-Schwarz')
    return {'basis': 'fixed COMP per-net multiplier' if weighted else 'raw MSE',
            'p1_minus_d50': one(e0, q1, a),
            'comp_minus_p1': one(e1, q2+b, c),
            'two_direction_span': {
                'base_error': e0, 'optimistic_minimum_error': e0-gain,
                'fractional_error_reduction': gain/e0,
                'oracle_coefficients_not_deployable': [x1, x2],
                'normalized_gram_determinant': det/(a*c)},
            'actual_original_errors_at_reference_weights': [e0, e1, e2]}


def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60s cap')))
    signal.alarm(60)
    start = time.monotonic()
    out = HERE / 'diagnostic_v1'
    out.mkdir(exist_ok=False)
    bindings = {str(Path(__file__).resolve()): sha(Path(__file__)),
                str(HERE/'PROTOCOL.md'): sha(HERE/'PROTOCOL.md')}
    result = {'passed': False, 'official_score': False, 'candidate_score': False,
              'truth_arrays_accessed': False, 'estimator_executed': False,
              'fit_or_refit': False, 'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        objs = {}
        for p, h in PINS.items():
            objs[p] = json.loads(checked(p, h, bindings).read_text())
        for p, d in objs.items():
            if not d.get('passed'):
                raise ValueError(f'unpassed input {p}')
            if p.endswith('/completion.json'):
                rp = p.removesuffix('completion.json') + 'result.json'
                if d['result_sha256'] != PINS[rp]:
                    raise ValueError('completion binding')
        s1 = objs['current_d50_native_objective_p1_v1/score2_v1/result.json']
        s2 = objs['current_p1_node_composition_v1/score_v3/result.json']
        natives = [objs[p] for p in (
            'birth_direction_fit_v1/validation_terminal_v2/result.json',
            'current_d50_native_objective_p1_v1/native2_terminal_v1/result.json',
            'current_p1_node_composition_v1/native_terminal_v2/result.json')]
        if len(s1['records']) != 40 or len(s2['records']) != 40:
            raise ValueError('score panel size')
        outputs = []
        for d in natives:
            if len(d['trials']) != 41 or d.get('truth_accessed'):
                raise ValueError('native acquisition')
            paths = []
            for t in d['trials']:
                paths.append(checked(t['output'], t['output_sha256'], bindings))
            if paths[0].read_bytes() != paths[40].read_bytes():
                raise ValueError('repeat mismatch')
            outputs.append(paths)
        # All output files authenticated before output arithmetic.
        rows = []
        for i, (r1, r2) in enumerate(zip(s1['records'], s2['records'])):
            guard()
            if r1['call'] != i or r2['call'] != i or r1['rank'] != r2['rank']:
                raise ValueError('score rank mapping')
            if r1['per_role']['terminal'] != r2['per_role']['P1']:
                raise ValueError('common P1 truth/bill frame mismatch')
            stats = [r1['per_role']['current_d50'], r1['per_role']['terminal'],
                     r2['per_role']['composition']]
            for d, stat in zip(natives, stats):
                t = d['trials'][i]
                if t['rank'] != r1['rank'] or t['call'] != i or t['flops'] != stat['flops']:
                    raise ValueError('native score/bill mapping')
                if stat['multiplier'] != max(.1, stat['flops']/2**41):
                    raise ValueError('multiplier mismatch')
                if abs(stat['adjusted']-stat['raw_mse']*stat['multiplier']) > 1e-23:
                    raise ValueError('objective mismatch')
            y = [read_npy(paths[i]) for paths in outputs]
            if not y[0][:1024] == y[1][:1024] == y[2][:1024]:
                raise ValueError('first layer control mismatch')
            v = [[x-z for x,z in zip(y[1][-1024:],y[0][-1024:])],
                 [x-z for x,z in zip(y[2][-1024:],y[1][-1024:])]]
            g11, g12, g22 = dot(v[0],v[0]), dot(v[0],v[1]), dot(v[1],v[1])
            e0, e1, e2 = (s['raw_mse'] for s in stats)
            b1 = (e1-e0-g11)/2
            b2 = (e2-e1-g22)/2-g12
            rows.append({'rank': r1['rank'], 'call': i, 'd50_mse': e0,
                         'p1_mse': e1, 'comp_mse': e2, 'g11': g11,
                         'g12': g12, 'g22': g22, 'e0_d1': b1, 'e0_d2': b2,
                         'reference_multiplier': stats[2]['multiplier']})
        raw = aggregate(rows, False)
        weighted = aggregate(rows, True)
        for p, h in bindings.items():
            if sha(Path(p)) != h:
                raise ValueError('input changed')
        guard()
        result.update(passed=True, records=rows, raw_projection_diagnostic=raw,
                      reference_weighted_projection_diagnostic=weighted,
                      simple_existing_direction_span_reaches_reference_sub3=
                      weighted['two_direction_span']['optimistic_minimum_error'] < 3e-9,
                      hashes_unchanged=True, no_deployable_coefficient_selected=True)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result.update(elapsed_s=time.monotonic()-start, input_bindings=bindings,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    target = out/'result.json'
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed': result['passed'],
        'result_sha256': sha(target)}, sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_bindings','records')}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
