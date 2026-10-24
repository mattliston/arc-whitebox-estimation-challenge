"""Lean exporter: add a sealed 705-column correction (scaled) to the seven readout heads of the LAST _GNN_LITERAL of a parent pair.
usage: export_iter2_v1.py --sci SCI --rel REL --seal SEAL.json --out DIR --name NAME [--scale 1.0]
"""
import argparse, ast, base64, hashlib, json, math, subprocess, zlib
from pathlib import Path
import numpy as np
COLS = (('head', 0, 64), ('head_b', 64, 65), ('hidden_head', 65, 193), ('first_hidden_head', 193, 321), ('sq_hidden_head', 321, 449), ('sq_first_head', 449, 577), ('gate_first_head', 577, 705))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def literal(text):
    tree = ast.parse(text); rows = [n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_GNN_LITERAL' for t in n.targets)]
    return ast.literal_eval(rows[-1]), tree
def unpack(item):
    shape, data = item; raw = zlib.decompress(base64.b64decode(data)); assert len(raw) == 4 * math.prod(shape); return np.frombuffer(raw, dtype='<f4').reshape(shape)
def pack(v): return (tuple(v.shape), base64.b64encode(zlib.compress(v.astype('<f4').tobytes(), 9)).decode())
def functions(tree): return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
def main():
    ap = argparse.ArgumentParser()
    for n in ('sci', 'rel', 'seal', 'out', 'name'): ap.add_argument('--' + n, required=True)
    ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--allow-teacher-mismatch', action='store_true'); a = ap.parse_args()
    seal = json.loads(Path(a.seal).read_text()); coef = np.asarray(seal['coef_by_layer'], dtype='f8') * a.scale; assert coef.shape == (15, 705) and np.isfinite(coef).all()
    texts = {'science': Path(a.sci).read_text(), 'release': Path(a.rel).read_text()}
    lits = {k: literal(t)[0] for k, t in texts.items()}; assert lits['science'] == lits['release']; prior = lits['science']
    assert a.allow_teacher_mismatch or seal['teacher_sha256'] == sha(a.sci), 'seal was fit on a different teacher'
    updated = dict(prior); proofs = {}
    for key, lo, hi in COLS:
        v = unpack(prior[key]); new = v.copy(); new[1:] = (v[1:].astype('f8') + coef[:, lo:hi].reshape(v[1:].shape)).astype('f4')
        assert np.array_equal(new[0], v[0]) and np.isfinite(new).all(); updated[key] = pack(new); assert np.array_equal(unpack(updated[key]), new)
        proofs[key] = dict(max_abs_change=float(np.abs(new.astype('f8') - v.astype('f8')).max()))
    out = Path(a.out); assert not out.exists(); out.mkdir(parents=True); files = {}
    suffix = '\n# iter2 on-policy readout correction (scale %r) on the fired trajectory; tensor-only.\nassert _GNN_PARAMS is None and not _E2_GNN_STATE\nassert not _FAST_RESCUE_FRAC\n_GNN_LITERAL = %s\n_FAST_RESCUE_FRAC = {}\n' % (a.scale, repr(updated))
    for role, text in texts.items():
        assert text.endswith('_FAST_RESCUE_FRAC = {}\n')
        target = out / ('candidate_%s_%s_v1.py' % (a.name, role)); target.write_text(text + suffix); written = target.read_text()
        dec, tree = literal(written); assert dec == updated and functions(tree) == functions(ast.parse(text)); compile(written, str(target), 'exec')
        files[role] = dict(path=str(target), sha256=sha(target), bytes=target.stat().st_size, parent_sha256=sha(a.sci if role == 'science' else a.rel))
    (out / 'EXPORTED_v1.json').write_text(json.dumps(dict(utc=subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip(), scale=a.scale, seal_sha256=sha(a.seal), files=files, proofs=proofs), indent=1) + '\n')
    print(json.dumps(dict(files=files, proofs=proofs)))
if __name__ == '__main__': main()
