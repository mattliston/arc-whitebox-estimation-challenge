"""Export a blocks fit (gate design + pv/hg/pvg feature blocks, TRAIN90 seal from fit_blocks_full90_v1) onto the exact 49bd/f938 parents.
Tensor part: head/head_b/hidden_head/first_hidden_head rows 1..15 += coef[:, :321]; NEW (16,128) tensors (row 0 zero):
sq_hidden_head, sq_first_head, gate_first_head, then pv_hidden_head / gate_hidden_head / pv_gate_head per the seal's design.
Code part: accessor functions and ONE override of _gop2_gnn_step: gate computed once, squared/gate readouts, previous layer's
hidden2 kept in the per-network constants dict (k['_pv90_prev']; zero contribution at layer 1).
usage: export_blocks_v1.py <seal.json> <out_dir> [scale] | --self-test
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
BASE_NEW = (('sq_hidden_head', 321, 449), ('sq_first_head', 449, 577), ('gate_first_head', 577, 705))
BLOCK_TENSOR = {'pv': 'pv_hidden_head', 'hg': 'gate_hidden_head', 'pvg': 'pv_gate_head', 'fh': 'first_hidden_prod_head'}
BLOCK_ORDER = ('pv', 'hg', 'pvg')
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

def accessor(name, key, rowskey):
    return "def %s(k,l):\n    rows=k.get('%s')\n    if rows is None:\n        rows=k['%s']=fnp.unstack(k['P']['%s'])\n    return rows[l]\n\n" % (name, rowskey, rowskey, key)

ACC = {'sq_hidden_head': '_sq90_hidden_head', 'sq_first_head': '_sq90_first_head', 'gate_first_head': '_g90_gate_first_head',
       'pv_hidden_head': '_pv90_prev_head', 'gate_hidden_head': '_g90_gate_hidden_head', 'pv_gate_head': '_pv90_gate_head', 'first_hidden_prod_head': '_fh90_prod_head'}

def step_override(text, design):
    tree = ast.parse(text)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_gop2_gnn_step']
    node = defs[-1]; lines = text.splitlines(keepends=True); src = ''.join(lines[node.lineno - 1:node.end_lineno])
    a = "    first_scalar = _e2_H(h @ _fh90_first_head(k, l))\n"
    b = "    state = (_e2_append_inline((_e2_value_inline := (h @ k['enc4_wT']))) or _e2_value_inline) + P['enc4_b']\n"
    c = "+ first_scalar))) or _e2_value_inline) + c['head_b'])"
    d = "((_e2_H(state @ c['head']) + _e2_H(h @ _hc90_hidden_head(k, l))) + first_scalar)"
    assert src.count(a) == 1 and src.count(b) == 1 and src.count(c) == 1 and src.count(d) == 1
    gate = "    gate90 = _e2_H(fnp.clip(_e2_H(_e2_H(Hx[:, 5:6] * c['scales'][5:6]) + c['centers'][5:6]), 0, 1))\n"
    after_a = "    sq_first = _e2_H(_e2_H(h * h) @ _sq90_first_head(k, l))\n    gate_first = _e2_H(_e2_H(h * gate90) @ _g90_gate_first_head(k, l))\n"
    before_b = "    sq_hidden = _e2_H(_e2_H(h * h) @ _sq90_hidden_head(k, l))\n"; terms = ['sq_hidden', 'sq_first', 'gate_first']
    if 'hg' in design: before_b += "    gate_hidden = _e2_H(_e2_H(h * gate90) @ _g90_gate_hidden_head(k, l))\n"; terms.append('gate_hidden')
    if 'fh' in design: after_a += "    first90 = h\n"; before_b += "    fh_prod = _e2_H(_e2_H(first90 * h) @ _fh90_prod_head(k, l))\n"; terms.append('fh_prod')
    if 'pv' in design or 'pvg' in design: before_b += "    pv90 = k.get('_pv90_prev')\n"
    if 'pv' in design: before_b += "    pv_hidden = _e2_H(pv90 @ _pv90_prev_head(k, l)) if (l > 1 and pv90 is not None) else 0\n"; terms.append('pv_hidden')
    if 'pvg' in design: before_b += "    pv_gate = _e2_H(_e2_H(pv90 * gate90) @ _pv90_gate_head(k, l)) if (l > 1 and pv90 is not None) else 0\n"; terms.append('pv_gate')
    if 'pv' in design or 'pvg' in design: before_b += "    k['_pv90_prev'] = h\n"
    src = src.replace(a, gate + a + after_a).replace(b, before_b + b)
    src = src.replace(c, "+ first_scalar)" + ''.join(" + %s)" % t for t in terms) + ")) or _e2_value_inline) + c['head_b'])")
    src = src.replace(d, "(" * len(terms) + d)
    compile(src, '<override>', 'exec')
    return src

def tensors_for(design, order=None):
    names = [n for n, _, _ in BASE_NEW] + [BLOCK_TENSOR[b] for b in (order if order is not None else [b for b in BLOCK_ORDER if b in design])]
    return names

def parse_scale(spec):
    """'1.5' -> 15 x 1.5; '1.5x8,2.0x7' -> rows 1..8 at 1.5 and rows 9..15 at 2.0 (per-layer strength)."""
    if 'x' not in spec and ',' not in spec: return [float(spec)] * 15
    out = []
    for part in spec.split(','):
        v, n = part.split('x'); out += [float(v)] * int(n)
    assert len(out) == 15, out
    return out

def main():
    if sys.argv[1] == '--self-test':
        for dsg in ((), ('pv',), ('hg',), ('pv', 'hg'), ('pv', 'hg', 'pvg'), ('hg', 'fh')):
            text = PARENTS['science'].read_text(); o = step_override(text, dsg); assert o == step_override(PARENTS['release'].read_text(), dsg)
            print('design', dsg, 'override lines', o.count('\n'), 'terms ok')
        print(step_override(PARENTS['science'].read_text(), ('hg', 'fh')))
        return
    sealp, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(); scale = parse_scale(sys.argv[3] if len(sys.argv) > 3 else '1.0')
    seal = read(sealp); design = tuple(seal['design']); assert seal['family'] == 'full90' and seal['physical']['passed'], seal['physical']
    coef = np.asarray(seal['coef_by_layer'], dtype='f8') * np.asarray(scale, dtype='f8')[:, None]; W = seal['width']; assert coef.shape == (15, W) and np.isfinite(coef).all() and min(scale) > 0
    names = tensors_for(design, seal.get('block_order')); assert W == 321 + 128 * len(names), (W, names)
    bspec = sys.argv[4] if len(sys.argv) > 4 else ''   # per-block strength, e.g. 'fh:2.0,hg:1.5' (applied on top of the per-layer scale)
    for part in [x for x in bspec.split(',') if x]:
        b, v = part.split(':'); key = BLOCK_TENSOR[b]; j = names.index(key); lo = 321 + 128 * j
        coef[:, lo:lo + 128] *= float(v); print('block', b, key, 'cols', lo, lo + 128, 'x', float(v))
    original = {role: path.read_text() for role, path in PARENTS.items()}
    for role, path in PARENTS.items(): assert sha(path) == PINS[role]
    parent = {role: literal(text)[0] for role, text in original.items()}; assert parent['science'] == parent['release']
    prior = parent['science']; updated = dict(prior); proofs = {}
    for key, lo, hi in COLUMNS:
        raw, values = unpack(prior[key]); assert tuple(prior[key][0]) == SHAPES[key]
        new = values.copy(); new[1:] = (values[1:].astype('f8') + coef[:, lo:hi].reshape(values[1:].shape)).astype('f4')
        assert np.array_equal(new[0], values[0]); updated[key] = pack(new); assert unpack(updated[key])[1].tobytes() == new.astype('<f4').tobytes()
        proofs[key] = dict(max_abs_change=float(np.abs(new.astype('f8') - values.astype('f8')).max()))
    for j, key in enumerate(names):
        lo = 321 + 128 * j; assert key not in prior; t = np.zeros((16, 128), 'f4'); t[1:] = coef[:, lo:lo + 128].astype('f4'); updated[key] = pack(t)
        proofs[key] = dict(new_tensor=True, row0_zero=True, rms=float(np.sqrt(np.mean(t[1:].astype('f8') ** 2))))
    override = step_override(original['science'], design); assert override == step_override(original['release'], design)
    acc = ''.join(accessor(ACC[key], key, '_%s_rows' % ACC[key].strip('_')) for key in names)
    suffix = ('\n# Blocks TRAIN90 on-policy readout (design %r, strength scale %r): squared/gate readouts plus %s; tensor payload plus one step override.\n' % (list(design), scale, ', '.join(names[3:]) or 'no extra blocks')
              + 'assert _GNN_PARAMS is None and not _E2_GNN_STATE\nassert not _FAST_RESCUE_FRAC\n\n' + acc + override + '\n_GNN_LITERAL = ' + repr(updated) + '\n_FAST_RESCUE_FRAC = {}\n')
    assert not out.exists(); out.mkdir(parents=True); files = {}
    for role, text in original.items():
        target = out / ('candidate_first128_onpolicy_blocks_' + role + '_v1.py'); target.write_text(text + suffix)
        written = target.read_text(); decoded, tree = literal(written); assert decoded == updated
        fo, fn = functions(literal(text)[1]), functions(tree)
        changed = {k for k in fo if fo[k] != fn.get(k)}; added = set(fn) - set(fo)
        assert changed == {'_gop2_gnn_step'} and added == {ACC[k] for k in names}, (changed, added)
        compile(written, str(target), 'exec')
        files[role] = dict(path=str(target), sha256=sha(target), bytes=target.stat().st_size, parent_sha256=PINS[role])
    (out / 'EXPORTED_v1.json').write_text(json.dumps(dict(utc=utc(), successful=True, strength_scale=scale, strength_spec=(sys.argv[3] if len(sys.argv) > 3 else '1.0'), block_spec=(sys.argv[4] if len(sys.argv) > 4 else ''), design=list(design), seal_sha256=sha(sealp), files=files, tensor_proofs=proofs, step_override=override), indent=2) + '\n')
    print(json.dumps(dict(successful=True, files=files, design=list(design))))

if __name__ == '__main__': main()
