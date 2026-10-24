"""Parameter-only S34 diagnostic overrides; call only inside a root guard."""
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
SOURCE = H / 'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
SOURCE_SHA = '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
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


def load_s34_arrays(np):
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

