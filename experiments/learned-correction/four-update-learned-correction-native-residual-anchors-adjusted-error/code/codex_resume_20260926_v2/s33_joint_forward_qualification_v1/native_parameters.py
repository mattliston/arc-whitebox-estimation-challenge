"""Parameter-only S33 diagnostic overrides; call only inside a root guard."""
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent
H = HERE.parent
P2 = H.parent
SOURCE = H / 'hill_climb_precision_g3_symdiag128_v1/candidate.py'
SOURCE_SHA = '88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6'
DIALS = P2 / 'fable_physicist_v1/refit6b/s6b_loopR8s_joint_3ep/dials_epoch1.json'
DIALS_SHA = 'e881c6da752d60845d583ef98e916fd21f809d98d6d416959ec8e450b0621992'
LITERALS = ('_GNN_LITERAL', '_BDNH_A_LITERAL', '_BD_COEFFICIENT_LITERAL',
            '_EDGE_LITERAL', '_MESSAGE_EDGE_LITERAL')
FROZEN_P = ('centers', 'scales', 'rms', 'gain', 'first')
CACHES = ('_GNN_PARAMS', '_BD_COEFFICIENTS', '_BDNH_PARAMETERS',
          '_EDGE_PARAMETERS', '_MESSAGE_EDGE_PARAMETERS')


def _source_values():
    source = SOURCE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
    nodes = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            nodes.setdefault(node.targets[0].id, []).append(node)
    result = {}
    for name in LITERALS:
        assert len(nodes[name]) == 1
        result[name] = ast.literal_eval(nodes[name][0].value)
    return result, nodes


def load_s33_arrays(np):
    """Return all 28 canonical native FP32 arrays and their 12 matched dials."""
    literals, nodes = _source_values()
    arrays = {}
    for key, (shape, encoded) in literals['_GNN_LITERAL'].items():
        raw = zlib.decompress(base64.b64decode(encoded, validate=True))
        arrays['p.' + key] = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    arrays['a'] = np.asarray(literals['_BDNH_A_LITERAL'], dtype=np.float32)
    arrays['c'] = np.asarray(literals['_BD_COEFFICIENT_LITERAL'], dtype=np.float32)
    for key, value in literals['_EDGE_LITERAL'].items():
        arrays['e.' + key] = np.asarray(value, dtype=np.float32)
    for key, value in literals['_MESSAGE_EDGE_LITERAL'].items():
        arrays['e.context_' + key] = np.asarray(value, dtype=np.float32)
    assert len(arrays) == 28
    assert all(value.dtype == np.float32 and np.isfinite(value).all()
               for value in arrays.values())
    raw_dials = DIALS.read_bytes()
    assert hashlib.sha256(raw_dials).hexdigest() == DIALS_SHA
    dials = json.loads(raw_dials)
    expected = dict(_BD_J1_PHYS=dials['X1'],
        _S19_BD_ADD=(dials['X0'], 0.0, dials['X2'], dials['X3']),
        _S19_BSCALE=dials['bscale'], _S19_LAM=(dials['s_carry'], dials['s_new']),
        _S19_K4BAND=(dials['k4a'], dials['k4b'], dials['k4c']))
    for name, value in expected.items():
        assert ast.literal_eval(nodes[name][-1].value) == value
    table = nodes['_K31R_TABLE'][-1].value
    assert isinstance(table, ast.DictComp)
    assert ast.literal_eval(table.value) == (dials['c0'], dials['c1'])
    return arrays, dials


def _nest(original, value):
    if isinstance(original, (tuple, list)):
        assert len(original) == len(value)
        return type(original)(_nest(old, new) for old, new in zip(original, value))
    return float(value)


def install_on_module(np, mod, arrays, dials):
    """Install one fixed diagnostic arm before setup; do not edit source files."""
    assert hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() == SOURCE_SHA
    assert all(getattr(mod, name) is None for name in CACHES)
    assert mod._SCRATCH is None
    baseline, baseline_dials = load_s33_arrays(np)
    assert set(arrays) == set(baseline) and set(dials) == set(baseline_dials)
    for key, value in arrays.items():
        assert value.dtype == np.float32 and value.shape == baseline[key].shape
        assert np.isfinite(value).all()
    for key in FROZEN_P:
        assert arrays['p.' + key].tobytes() == baseline['p.' + key].tobytes()
    for key in ('p.head', 'p.head_b'):
        assert arrays[key][[0, 15]].tobytes() == baseline[key][[0, 15]].tobytes()
    assert arrays['c'][:, [1, 3]].tobytes() == baseline['c'][:, [1, 3]].tobytes()
    assert all(math.isfinite(float(value)) for value in dials.values())
    assert dials['X3'] == baseline_dials['X3']
    original, _ = _source_values()
    for name in LITERALS:
        assert getattr(mod, name) == original[name]
    changes = {}
    packed = {}
    for key in original['_GNN_LITERAL']:
        arr = np.ascontiguousarray(arrays['p.' + key], dtype=np.float32)
        packed[key] = (tuple(arr.shape), base64.b64encode(zlib.compress(arr.tobytes())).decode())
    changes['_GNN_LITERAL'] = packed
    changes['_BDNH_A_LITERAL'] = _nest(original['_BDNH_A_LITERAL'], arrays['a'].tolist())
    changes['_BD_COEFFICIENT_LITERAL'] = _nest(original['_BD_COEFFICIENT_LITERAL'], arrays['c'].tolist())
    changes['_EDGE_LITERAL'] = {key: _nest(value, arrays['e.' + key].tolist())
        for key, value in original['_EDGE_LITERAL'].items()}
    changes['_MESSAGE_EDGE_LITERAL'] = {key: _nest(value, arrays['e.context_' + key].tolist())
        for key, value in original['_MESSAGE_EDGE_LITERAL'].items()}
    changes.update(_BD_J1_PHYS=float(dials['X1']),
        _S19_BD_ADD=(float(dials['X0']), 0.0, float(dials['X2']), float(dials['X3'])),
        _S19_BSCALE=float(dials['bscale']),
        _S19_LAM=(float(dials['s_carry']), float(dials['s_new'])),
        _S19_K4BAND=(float(dials['k4a']), float(dials['k4b']), float(dials['k4c'])),
        _K31R_TABLE={key: (float(dials['c0']), float(dials['c1']))
                     for key in mod._K31R_TABLE})
    before = dict(vars(mod))
    for key, value in changes.items():
        setattr(mod, key, value)
    assert set(vars(mod)) == set(before)
    assert all(value is before[key] for key, value in vars(mod).items() if key not in changes)
    assert hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() == SOURCE_SHA
    return dict(source_sha256=SOURCE_SHA, modified_globals=sorted(changes),
                other_global_identities_unchanged=True, source_bytes_unchanged=True,
                parameter_arrays=28, frozen_p=list(FROZEN_P),
                parameter_hashes={key: hashlib.sha256(value.tobytes()).hexdigest()
                                  for key, value in sorted(arrays.items())})
