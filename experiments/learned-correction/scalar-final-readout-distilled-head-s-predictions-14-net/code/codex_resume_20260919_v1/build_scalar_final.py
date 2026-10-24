"""Current I95 prefix plus a teacher-distilled scalar terminal read."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / 'fable_wall_repair_v1/cand_c2_wallrepair_I95.py'
EMITTER = P2 / 'astra_phase2_reexamination_v1/current_shared384_anchored_p_proposal_v1/build_native.py'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def once(source, old, new):
    assert source.count(old) == 1, (source.count(old), old[:100])
    return source.replace(old, new)


def main():
    spec = importlib.util.spec_from_file_location('literal_helper', EMITTER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original_pack = helper.pack_literal
    helper.pack_literal = lambda v: original_pack(list(v) if isinstance(v, tuple) else v)
    base = BASE.read_bytes()
    assert sha(base) == 'dd64dfc367fbaf154fe956f335c21aa51c1589e7f19a731be74c29ca41bc29b3'
    values = helper.surface(base)[0]
    with np.load(HERE / 'scalar_distill_pilot_payload.npz') as payload:
        head = values['head'][1][:-64*4] + payload['head'].astype('<f4').tobytes()
        bias = values['head_b'][1][:-4] + payload['head_b'].astype('<f4').tobytes()
    values['head'] = (values['head'][0], head)
    values['head_b'] = (values['head_b'][0], bias)
    literal_source, literal_proof = helper.emit(base, values)
    source = literal_source.decode()
    edits = []

    def edit(old, new):
        nonlocal source
        source = once(source, old, new)
        edits.append((old, new))

    mask = [0 if i in (13,14,15,22,23,24,25,26,27,28) else 1 for i in range(30)]
    old = '    x = fnp.clip((x.astype(fnp.float32)-P["centers"][l])/P["scales"][l],-8,8)\n'
    edit(old, old + f'    if l == 15:\n        x = x * fnp.asarray({mask!r}, dtype=fnp.float32)[None, :]\n')
    old = '    return _predict_binary(mlp,hook=hook,coefficient_hook=coefficient_hook)\n'
    new = '''    def scalar_terminal_hook(l, m, aux, w):
        x = final_scalar_features(l, m, aux, w)
        delta, _ = gnn_step(l, x, aux['sigma'], w, state, P)
        return fnp.asarray(delta, dtype=fnp.float64)
    return _predict_binary(mlp,hook=hook,coefficient_hook=coefficient_hook,
                           final_hook=scalar_terminal_hook)
'''
    edit(old, new)
    # Limit insertion to the active, last _predict_binary definition. Its
    # prefix through layer14 and its learned/node/birth parameters are intact.
    start = source.rindex('def _predict_binary(')
    anchor = '        mu = w.T @ m\n'
    before, active = source[:start], source[start:]
    assert active.count(anchor) == 1
    branch = '''        if l == depth - 1 and final_hook is not None:
            v, active_v = _cov_compact_diagonal(w32, C, previous, previous_alive)
            bpre = w.T @ b
            pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)
            k3 = fnp.sum(_cp_diagonal_cube(pp.astype(fnp.float64)) * lam.astype(fnp.float64), axis=1)
            k3 = k3 + _star_compact_diagonal(w32, star, previous, previous_alive)
            k3 = k3 + 3 * active_v * bpre
            m, aux = nonlinear_scalar(mu, v, k3, radial * v * v)
            m = m + final_hook(l, m, aux, W[l])
            means.append(m.copy())
            break
'''
    source = before + once(active, anchor, anchor + branch)
    # Independent inverse: every executable change and both head fields can
    # be removed to recover the entire submitted source byte for byte.
    inverse = once(source, anchor + branch, anchor)
    for old, new in reversed(edits):
        inverse = once(inverse, new, old)
    assert inverse.encode() == literal_source
    old_values, old_spans, _, _ = helper.surface(base)
    _, new_spans, _, _ = helper.surface(inverse.encode())
    raw = inverse.encode()
    for key in ('head_b', 'head'):
        a,b = new_spans[key]
        c,d = old_spans[key]
        # Span positions may shift after the first replacement: recalculate.
        a,b = helper.surface(raw)[1][key]
        raw = raw[:a] + base[c:d] + raw[b:]
    assert raw == base
    ast.parse(source)
    candidate = source.encode()
    out = HERE / 'scalar_distill_I95.py'
    with out.open('xb') as stream:
        stream.write(candidate)
    proof = dict(parent_sha256=sha(base), candidate_sha256=sha(candidate),
                 payload_sha256=sha((HERE / 'scalar_distill_pilot_payload.npz').read_bytes()),
                 builder_sha256=sha(Path(__file__).read_bytes()), whole_source_inverse_verified=True,
                 earlier_layer_parameters_unchanged=True, prefix_arithmetic_unchanged=True,
                 modified_head_fields=literal_proof['modified_keys'],
                 scalar_terminal_approximation=True, native_parity_not_yet_tested=True)
    with (HERE / 'scalar_build.json').open('x') as stream:
        json.dump(proof, stream, indent=2)
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
