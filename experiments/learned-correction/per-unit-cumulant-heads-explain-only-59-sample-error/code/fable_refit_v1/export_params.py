#!/usr/bin/env python3
"""TAG fable-refit (RF-5): write refit parameters (params_epochN.pt from refit.py: p, a, c, e) back into a native candidate's literals
(_GNN_LITERAL packed base64/zlib f32, _BD_COEFFICIENT_LITERAL 15x4, _BDNH_A_LITERAL 64x4, _EDGE_LITERAL nested lists/floats,
_MESSAGE_EDGE_LITERAL left/right 48x16), preserving the original key order and nesting.  usage: export_params.py <in_cand.py> <params.pt> <out_cand.py>"""
import sys, ast, base64, zlib, json
from pathlib import Path
import numpy as np, torch
src_path, pt_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
src = src_path.read_text(); tree = ast.parse(src)
saved = torch.load(pt_path)
p = {k: v.numpy() for k, v in saved['p'].items()}; a = saved['a'].numpy(); c = saved['c'].numpy(); e = {k: v.numpy() for k, v in saved['e'].items()}
def pack(arr):
    arr = np.ascontiguousarray(arr.astype(np.float32))
    return (tuple(int(x) for x in arr.shape), base64.b64encode(zlib.compress(arr.tobytes())).decode())
def nest(orig, arr):
    """rebuild the original nesting (list/tuple/float) from an array"""
    if isinstance(orig, float): return float(np.asarray(arr).reshape(()))
    if isinstance(orig, (list, tuple)):
        rows = [nest(o, arr[i]) for i, o in enumerate(orig)]
        return rows if isinstance(orig, list) else tuple(rows)
    raise TypeError(type(orig))
edits = []
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name == '_GNN_LITERAL':
            orig = ast.literal_eval(node.value); assert set(orig) == set(p), (set(orig) ^ set(p))
            new = {k: pack(p[k]) for k in orig}; assert all(new[k][0] == tuple(orig[k][0]) for k in orig)
        elif name == '_BD_COEFFICIENT_LITERAL':
            orig = ast.literal_eval(node.value); assert np.asarray(orig).shape == c.shape; new = nest(orig, c)
        elif name == '_BDNH_A_LITERAL':
            orig = ast.literal_eval(node.value); assert np.asarray(orig).shape == a.shape; new = nest(orig, a)
        elif name == '_EDGE_LITERAL':
            orig = ast.literal_eval(node.value); new = {k: nest(orig[k], e[k]) for k in orig}
        elif name == '_MESSAGE_EDGE_LITERAL':
            orig = ast.literal_eval(node.value); new = {k: nest(orig[k], e['context_' + k]) for k in orig}
        else:
            continue
        edits.append((node.lineno, node.end_lineno, name, new))
assert len(edits) == 5, [x[2] for x in edits]
lines = src.split('\n')
for lineno, end, name, new in sorted(edits, key=lambda x: -x[0]):
    lines[lineno - 1:end] = [f"{name} = {new!r}"]
out_path.write_text('\n'.join(lines)); ast.parse(out_path.read_text())
print(json.dumps(dict(out=str(out_path), edited=[x[2] for x in edits], bytes=len(out_path.read_text()))))
