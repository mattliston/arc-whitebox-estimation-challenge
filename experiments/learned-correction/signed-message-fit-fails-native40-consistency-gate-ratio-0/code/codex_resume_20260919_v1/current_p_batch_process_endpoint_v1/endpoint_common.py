from common import *
import ast
import base64
import zlib


def endpoint_verify():
    verify()


def literal_span(source):
    nodes = [n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_GNN_LITERAL']
    assert len(nodes) == 1
    n = nodes[0]
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:n.lineno - 1])) + n.col_offset
    end = sum(map(len, lines[:n.end_lineno - 1])) + n.end_col_offset
    value = source[start:end]
    assert value == ast.get_source_segment(source, n)
    return start, end, value


def decode(np, literal):
    result = {}
    for key, (shape, blob) in ast.literal_eval(literal).items():
        result[key] = np.frombuffer(zlib.decompress(base64.b64decode(blob, validate=True)),
            dtype=np.float32).reshape(shape).copy()
    return result


def serialize(np, values):
    assert all(v.dtype == np.float32 and v.flags.c_contiguous and np.isfinite(v).all() for v in values.values())
    return repr({k: (v.shape, base64.b64encode(zlib.compress(v.tobytes(), 9)).decode('ascii'))
                 for k, v in values.items()})


def exact(a, b):
    return set(a) == set(b) and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype
                                  and a[k].tobytes() == b[k].tobytes() for k in a)
