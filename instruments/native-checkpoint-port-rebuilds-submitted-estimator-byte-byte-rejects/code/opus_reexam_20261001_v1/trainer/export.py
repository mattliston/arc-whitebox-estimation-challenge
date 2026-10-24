"""TAG opus-trainer (2026-10-01): plain exporter. Writes a native candidate.py = the research parent RRPI source with ONLY the
parameter values replaced: the five learned literals (_GNN_LITERAL, _BDNH_A_LITERAL, _BD_COEFFICIENT_LITERAL, _EDGE_LITERAL,
_MESSAGE_EDGE_LITERAL), the last assignments of _BD_J1_PHYS, _S19_BD_ADD, _S19_BSCALE, _S19_LAM, _S19_K4BAND and the (c0, c1) tuple
inside the last _K31R_TABLE comprehension. Only the VALUE EXPRESSION of each statement is replaced (byte span from the parser), so
comments and every other byte stay as they are; an array whose bytes are unchanged keeps its original encoded text.
Checked on every export: frozen arrays/dials unchanged; the new file compiles; every other top-level statement is AST-identical;
putting the old value text back gives the parent byte for byte; the arrays and dials decoded from the new file equal the input
bit for bit. Plain rewrite of H/hill_climb_rrpi_detached_star_v1/export_endpoint.py without its protocol ceremony.
Physics shifts stored with a parameter file ('pl', otr_pl.py families; master --extra) are exported too:
  rc, edge   as new values of the native per-layer tables _GN_RC / _GN_EDGE (literals only);
  bsrc       (one scale s for all layers) needs ONE native code line: the b-carrier source `radial d v / 6` becomes
             `radial d v / _S19_BSRC_DIV` (both branches of that line) and a new literal _S19_BSRC_DIV = 6 / s is added after
             _S19_K4BAND. Same operations, one constant changed; the engineer must confirm bill and rounding natively.
  iso, k21n, k31   (one shift for all layers) as new values of the native literal _GN_KNOBS (the setup pre-binds iso and k31
             from that literal, so nothing else moves).
  Any other family with a non-zero shift is refused.
usage: python -B export.py <params.pt> <new candidate.py>            (parameter file as written by master.py)
       python -B export.py --roundtrip <new candidate.py>            (untrained parameters; must reproduce the parent exactly)"""
import ast
import base64
import sys
import zlib
from pathlib import Path

import numpy as np

from otr_common import DIAL_KEYS, FROZEN_DIALS, FROZEN_P, LITERALS, SOURCE, SOURCE_SHA, publish, read_source, sha256, source_nodes


def nest(original, value):
    if isinstance(original, (tuple, list)):
        assert len(original) == len(value)
        return type(original)(nest(old, new) for old, new in zip(original, value))
    return float(value)


KNOBS = ('iso', 'k21n', 'k31')
NATIVE = {'rc', 'edge', 'bsrc'} | set(KNOBS)
BSRC_OLD = b'_e2_H(_e2_H(_e2_H(radial * d) * vv) / 6)'
BSRC_NEW = b'_e2_H(_e2_H(_e2_H(radial * d) * vv) / _S19_BSRC_DIV)'


def export(arrays, dials, out, pl=None):
    """arrays: 28 float32 numpy arrays; dials: 12 floats; pl: optional physics shifts {family: 16 floats}.
    Returns the list of statements whose value changed."""
    pl = {family: [float(x) for x in values] for family, values in (pl or {}).items() if any(values)}
    assert set(pl) <= NATIVE, 'no native form yet for: %s' % sorted(set(pl) - NATIVE)
    assert sha256(SOURCE) == SOURCE_SHA
    raw = SOURCE.read_bytes()
    base, base_dials = read_source(np)
    assert set(arrays) == set(base) and tuple(dials) == DIAL_KEYS
    for key, value in arrays.items():
        assert value.dtype == np.float32 and value.shape == base[key].shape and np.isfinite(value).all(), key
    for key in FROZEN_P:
        assert arrays['p.' + key].tobytes() == base['p.' + key].tobytes()
    assert all(type(value) is float and np.isfinite(value) for value in dials.values())
    assert all(dials[key] == base_dials[key] for key in FROZEN_DIALS)
    nodes = source_nodes(raw.decode())
    literal = {name: ast.literal_eval(nodes[name][0].value) for name in LITERALS}
    packed = {}
    for key, (shape, encoded) in literal['_GNN_LITERAL'].items():
        value = np.ascontiguousarray(arrays['p.' + key], dtype=np.float32)
        if value.tobytes() != base['p.' + key].tobytes():
            encoded = base64.b64encode(zlib.compress(value.tobytes())).decode()
        packed[key] = (tuple(value.shape), encoded)
    values = dict(
        _GNN_LITERAL=packed,
        _BDNH_A_LITERAL=nest(literal['_BDNH_A_LITERAL'], arrays['a'].tolist()),
        _BD_COEFFICIENT_LITERAL=nest(literal['_BD_COEFFICIENT_LITERAL'], arrays['c'].tolist()),
        _EDGE_LITERAL={key: nest(value, arrays['e.' + key].tolist()) for key, value in literal['_EDGE_LITERAL'].items()},
        _MESSAGE_EDGE_LITERAL={key: nest(value, arrays['e.context_' + key].tolist())
                               for key, value in literal['_MESSAGE_EDGE_LITERAL'].items()},
        _BD_J1_PHYS=dials['X1'], _S19_BD_ADD=(dials['X0'], 0., dials['X2'], dials['X3']),
        _S19_BSCALE=dials['bscale'], _S19_LAM=(dials['s_carry'], dials['s_new']),
        _S19_K4BAND=(dials['k4a'], dials['k4b'], dials['k4c']))
    for family, name in (('rc', '_GN_RC'), ('edge', '_GN_EDGE')):
        if family in pl:
            assert len(nodes[name]) == 1
            table = ast.literal_eval(nodes[name][0].value)
            values[name] = {layer: old + pl[family][layer] for layer, old in table.items()}
    if any(family in pl for family in KNOBS):
        assert len(nodes['_GN_KNOBS']) == 1
        knobs = ast.literal_eval(nodes['_GN_KNOBS'][0].value)
        for family in KNOBS:
            if family in pl:
                assert all(x == pl[family][2] for x in pl[family][2:15]), '%s is exported as one shift for all layers' % family
                knobs[family] = knobs[family] + pl[family][2]
        values['_GN_KNOBS'] = knobs
    spans = {name: nodes[name][-1].value for name in values}        # the expression to replace, per statement
    table = nodes['_K31R_TABLE'][-1].value
    assert isinstance(table, ast.DictComp)
    spans['_K31R_TABLE'], values['_K31R_TABLE'] = table.value, (dials['c0'], dials['c1'])
    assert len(spans) == 11 + sum(family in pl for family in ('rc', 'edge')) + any(family in pl for family in KNOBS)
    lines = raw.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    edits = sorted((starts[node.lineno - 1] + node.col_offset, starts[node.end_lineno - 1] + node.end_col_offset, name)
                   for name, node in spans.items())
    pieces, undo, cursor, changed = [], [], 0, []
    for start, stop, name in edits:
        new = repr(values[name]).encode()
        assert ast.literal_eval(raw[start:stop].decode()) is not None
        if new != raw[start:stop]:
            changed.append(name)
        pieces += [raw[cursor:start], new]
        undo += [raw[cursor:start], raw[start:stop]]
        cursor = stop
    candidate = b''.join(pieces + [raw[cursor:]])
    assert b''.join(undo + [raw[cursor:]]) == raw             # old values back in place = the parent, byte for byte
    literal_only = candidate
    if 'bsrc' in pl:
        scale = 1.0 + pl['bsrc'][1]
        assert all(1.0 + x == scale for x in pl['bsrc'][1:15]) and scale > 0, 'bsrc is exported as one scale for all layers'
        assert candidate.count(BSRC_OLD) == 2 and b'_S19_BSRC_DIV' not in candidate
        anchor = candidate.index(b'\n_S19_K4BAND = ')
        end = candidate.index(b'\n', anchor + 1) + 1
        line = ('_S19_BSRC_DIV = %r   # trainer: b-carrier source scale %r (radial d v / 6 x scale)\n' % (6.0 / scale, scale)).encode()
        candidate = (candidate[:end] + line + candidate[end:]).replace(BSRC_OLD, BSRC_NEW)
        assert candidate.replace(BSRC_NEW, BSRC_OLD).replace(line, b'') == literal_only   # exactly two substitutions and one new line
        changed.append('_S19_BSRC_DIV (new) and the b source line of _predict_binary')
    compile(candidate, str(out), 'exec')
    old_tree, new_tree = ast.parse(raw), ast.parse(literal_only)
    assert len(old_tree.body) == len(new_tree.body)
    for old, new in zip(old_tree.body, new_tree.body):
        if ast.dump(old) != ast.dump(new):
            assert isinstance(old, ast.Assign) and isinstance(old.targets[0], ast.Name)
            name = old.targets[0].id
            assert name in changed and nodes[name][-1].lineno == old.lineno and new.targets[0].id == name
    publish(out, lambda stream: stream.write(candidate), 'wb')
    back, back_dials = read_source(np, out)
    assert set(back) == set(arrays) and all(back[key].tobytes() == arrays[key].tobytes() for key in arrays)
    assert back_dials == dials
    written = source_nodes(out.read_text())
    for family, name in (('rc', '_GN_RC'), ('edge', '_GN_EDGE')):
        if family in pl:
            assert ast.literal_eval(written[name][0].value) == values[name]
    if 'bsrc' in pl:
        assert ast.literal_eval(written['_S19_BSRC_DIV'][0].value) == 6.0 / (1.0 + pl['bsrc'][1])
    if '_GN_KNOBS' in values:
        assert ast.literal_eval(written['_GN_KNOBS'][0].value) == values['_GN_KNOBS']
    return changed


if __name__ == '__main__':
    import json
    out = Path(sys.argv[2])
    if sys.argv[1] == '--roundtrip':
        arrays, dials = read_source(np)
    else:
        import torch
        from otr_common import load_params, load_pl
        arrays, dials = load_params(torch, sys.argv[1])
        arrays = {key: value.numpy() for key, value in arrays.items()}
        pl = load_pl(torch, sys.argv[1])
    changed = export(arrays, dials, out, pl if sys.argv[1] != '--roundtrip' else None)
    print(json.dumps(dict(candidate=str(out), sha256=sha256(out), bytes=out.stat().st_size, parent_sha256=SOURCE_SHA,
                          changed_statements=changed, identical_to_parent=out.read_bytes() == SOURCE.read_bytes(),
                          reload_bit_exact=True)), flush=True)
