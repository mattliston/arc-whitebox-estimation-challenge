"""Export the extended+gate on-policy fit (705 columns) onto the exact 49bd/f938 parents.
Tensor part: head/head_b/hidden_head/first_hidden_head rows 1..15 += coef[:, :321] (float32, fitter deploy() arithmetic);
two NEW tensors sq_hidden_head (16,128) and sq_first_head (16,128) with row 0 zero. Code part: two accessor functions and ONE
override of _gop2_gnn_step whose only change is keeping the first hidden h, adding the two squared readouts before the bias.
usage: export_extended_v1.py <fit_receipts_dir> <out_dir> [scale]
"""
import ast, base64, hashlib, json, math, subprocess, sys, zlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
NC = HERE.parent / 'runtime_first128_frozen_census_v1'
PARENTS = {'science': NC / 'candidate_first128_paid_nocensus_science_v1.py', 'release': NC / 'candidate_first128_paid_nocensus_release_v1.py'}
PINS = {'science': '49bd2ba31f0bce983f44cc001746e63ffaa84e530d3b31efa6424f69a0298c80', 'release': 'f9381d0982f5d24e76d38884188ba7c998548e4a3f306825e7e80fb9fa0fc3b6'}
COLUMNS = (('head', 0, 64), ('head_b', 64, 65), ('hidden_head', 65, 193), ('first_hidden_head', 193, 321))
SHAPES = {'head': (16, 64), 'head_b': (16,), 'hidden_head': (16, 128), 'first_hidden_head': (16, 128)}
NEW = (('sq_hidden_head', 321, 449), ('sq_first_head', 449, 577), ('gate_first_head', 577, 705))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()

def literal(text):
    tree = ast.parse(text)
    rows = [n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_GNN_LITERAL' for t in n.targets)]
    return ast.literal_eval(rows[-1]), tree

def unpack(item):
    shape, data = item; raw = zlib.decompress(base64.b64decode(data)); assert len(raw) == 4 * math.prod(shape)
    values = np.frombuffer(raw, dtype='<f4').reshape(shape); assert np.isfinite(values).all(); return raw, values

def pack(values): return (tuple(values.shape), base64.b64encode(zlib.compress(values.astype('<f4').tobytes(), 9)).decode())

def functions(tree): return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

def step_override(text):
    tree = ast.parse(text)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_gop2_gnn_step']
    node = defs[-1]; lines = text.splitlines(keepends=True); src = ''.join(lines[node.lineno - 1:node.end_lineno])
    a = "    first_scalar = _e2_H(h @ _fh90_first_head(k, l))\n"
    b = "    state = (_e2_append_inline((_e2_value_inline := (h @ k['enc4_wT']))) or _e2_value_inline) + P['enc4_b']\n"
    c = "+ first_scalar))) or _e2_value_inline) + c['head_b'])"
    assert src.count(a) == 1 and src.count(b) == 1 and src.count(c) == 1, (src.count(a), src.count(b), src.count(c))
    src = src.replace(a, a + "    sq_first = _e2_H(_e2_H(h * h) @ _sq90_first_head(k, l))\n    gate_first = _e2_H(_e2_H(h * _e2_H(fnp.clip(_e2_H(_e2_H(Hx[:, 5:6] * c['scales'][5:6]) + c['centers'][5:6]), 0, 1))) @ _g90_gate_first_head(k, l))\n")
    src = src.replace(b, "    sq_hidden = _e2_H(_e2_H(h * h) @ _sq90_hidden_head(k, l))\n" + b)
    src = src.replace(c, "+ first_scalar) + sq_hidden) + sq_first) + gate_first))) or _e2_value_inline) + c['head_b'])")
    # the two added parentheses must balance: original '((... + first_scalar)))' -> '((((... + first_scalar) + sq_hidden) + sq_first)))' needs two more '(' at the start of the inner group
    d = "((_e2_H(state @ c['head']) + _e2_H(h @ _hc90_hidden_head(k, l))) + first_scalar)"
    assert src.count(d) == 1
    src = src.replace(d, "(((" + d)
    compile(src, '<override>', 'exec')
    return src

ACCESSORS = '''
def _sq90_hidden_head(k,l):
    rows=k.get('_sq90_hidden_rows')
    if rows is None:
        rows=k['_sq90_hidden_rows']=fnp.unstack(k['P']['sq_hidden_head'])
    return rows[l]

def _sq90_first_head(k,l):
    rows=k.get('_sq90_first_rows')
    if rows is None:
        rows=k['_sq90_first_rows']=fnp.unstack(k['P']['sq_first_head'])
    return rows[l]

def _g90_gate_first_head(k,l):
    rows=k.get('_g90_gate_first_rows')
    if rows is None:
        rows=k['_g90_gate_first_rows']=fnp.unstack(k['P']['gate_first_head'])
    return rows[l]

'''

def main():
    receipts, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(); scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    fit = read(receipts / 'FITTED_v1.json'); seal = read(receipts / 'full90_COEFFICIENTS_v1.json')
    assert fit['successful'] and fit['passed'] and fit['status'] == 'EXTENDED_PASS_NATIVE_REQUIRED' and fit['width'] == 705
    cv = fit['cross_validation']; assert cv['mean_incremental_relative_pct'] <= -.15 and cv['incrementally_lower'] >= 54 and fit['negative_blocks'] >= 4
    assert fit['full90']['physical']['passed'] and fit['full90']['coefficient_seal_sha256'] == sha(receipts / 'full90_COEFFICIENTS_v1.json')
    coef = np.asarray(seal['coef_by_layer'], dtype='f8') * scale; assert coef.shape == (15, 705) and np.isfinite(coef).all() and scale > 0
    original = {role: path.read_text() for role, path in PARENTS.items()}
    for role, path in PARENTS.items(): assert sha(path) == PINS[role]
    parent = {role: literal(text)[0] for role, text in original.items()}; assert parent['science'] == parent['release']
    prior = parent['science']; updated = dict(prior); proofs = {}
    for key, lo, hi in COLUMNS:
        raw, values = unpack(prior[key]); assert tuple(prior[key][0]) == SHAPES[key]
        new = values.copy(); new[1:] = (values[1:].astype('f8') + coef[:, lo:hi].reshape(values[1:].shape)).astype('f4')
        assert np.array_equal(new[0], values[0]); updated[key] = pack(new); assert unpack(updated[key])[1].tobytes() == new.astype('<f4').tobytes()
        proofs[key] = dict(max_abs_change=float(np.abs(new.astype('f8') - values.astype('f8')).max()))
    for key, lo, hi in NEW:
        assert key not in prior; t = np.zeros((16, 128), 'f4'); t[1:] = coef[:, lo:hi].astype('f4'); updated[key] = pack(t)
        proofs[key] = dict(new_tensor=True, row0_zero=True, rms=float(np.sqrt(np.mean(t[1:].astype('f8') ** 2))))
    override = step_override(original['science']); assert override == step_override(original['release'])
    suffix = ('\n# Extended TRAIN90 on-policy readout (strength scale %r): squared hidden2/first128 and gate x first128 readouts; tensor payload plus one step override.\n' % scale
              + 'assert _GNN_PARAMS is None and not _E2_GNN_STATE\nassert not _FAST_RESCUE_FRAC\n' + ACCESSORS + override + '\n_GNN_LITERAL = ' + repr(updated) + '\n_FAST_RESCUE_FRAC = {}\n')
    assert not out.exists(); out.mkdir(parents=True); files = {}
    for role, text in original.items():
        target = out / ('candidate_first128_onpolicy_extgate_' + role + '_v1.py'); target.write_text(text + suffix)
        written = target.read_text(); decoded, tree = literal(written); assert decoded == updated
        fo, fn = functions(literal(text)[1]), functions(tree)
        changed = {k for k in fo if fo[k] != fn.get(k)}; added = set(fn) - set(fo)
        assert changed == {'_gop2_gnn_step'} and added == {'_sq90_hidden_head', '_sq90_first_head', '_g90_gate_first_head'}, (changed, added)
        compile(written, str(target), 'exec')
        files[role] = dict(path=str(target), sha256=sha(target), bytes=target.stat().st_size, parent_sha256=PINS[role])
    (out / 'EXPORTED_v1.json').write_text(json.dumps(dict(utc=utc(), successful=True, strength_scale=scale, files=files, cross_validation=cv, negative_blocks=fit['negative_blocks'], tensor_proofs=proofs, step_override=override), indent=2) + '\n')
    print(json.dumps(dict(successful=True, files=files, cv=cv)))

if __name__ == '__main__': main()
