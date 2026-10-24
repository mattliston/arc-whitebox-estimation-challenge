"""Text-only EDBAND constructor; no estimator import/execution.
prepare(parent_text, values) -> (asserted edits, rendered fragment, metadata).
Caller owns unique output paths; used by versioned stack/build_edband_v1.py.
"""
import ast
import hashlib
import math
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent


def normalize(values):
    assert values.get('schema') == 'edge_band_v1'
    allowed = {'schema', 'edges', 'bias', 'output', 'source', 'parent_candidate_sha256', 'parent_native_sha256', 'note'}
    assert set(values) <= allowed, sorted(set(values) - allowed)
    edges = values.get('edges', [])
    assert isinstance(edges, list) and len(edges) <= 7
    previous = 7
    for edge in edges:
        assert isinstance(edge, (list, tuple)) and len(edge) == 2
        lo, hi = edge
        assert type(lo) is int and type(hi) is int and previous < lo <= hi <= 14
        previous = hi
    def rows(key):
        given = values.get(key)
        if given is None:
            return tuple(None for _ in edges)
        assert len(given) == len(edges), key
        result = []
        for vector in given:
            if vector is None:
                result.append(None)
                continue
            assert len(vector) == 16, (key, len(vector))
            row = []
            for value in vector:
                assert type(value) in (int, float) and math.isfinite(value), (key, value)
                rounded = struct.unpack('<f', struct.pack('<f', value))[0]
                assert math.isfinite(rounded)
                row.append(rounded)
            result.append(tuple(row))
        return tuple(result)
    return tuple(tuple(e) for e in edges), rows('bias'), rows('output')


def signatures(source):
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'einsum':
            assert node.args
            arg = node.args[0]
            found.add(arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                      else '<existing dynamic expression:' + ast.dump(arg, include_attributes=False) + '>')
    return sorted(found)


def prepare(parent_text, values):
    assert '_EDBAND_EDGES' not in parent_text, 'already EDBAND'
    psha = hashlib.sha256(parent_text.encode()).hexdigest()
    bindings = [key for key in ('parent_native_sha256', 'parent_candidate_sha256') if values.get(key)]
    assert bindings, 'edge weights require a direct-parent SHA binding'
    for key in bindings:
        assert values[key] == psha, key + ' mismatch (bind to actual builder PARENT)'
    edges, bias, output = normalize(values)
    nodes = [n for n in ast.parse(parent_text).body if isinstance(n, ast.FunctionDef) and n.name == '_g5_message_edge_base']
    assert len(nodes) == 1, len(nodes)
    function = ast.get_source_segment(parent_text, nodes[0])
    site = '    k=_e2_edge_consts(t,n)\n'
    assert function.count(site) == 1
    prefix = function[:function.index(site) + len(site)]
    assert parent_text.count(prefix) == 1
    edits = [(prefix, prefix + '    t = _edge_band_parameters(layer, t)   # EDBAND after shared R/S cache lookup\n', 1)]
    fragment = (HERE / 'fragment_edband_v1.py').read_text()
    for name, value in (('EDGES', edges), ('BIAS', bias), ('OUTPUT', output)):
        old = '_EDBAND_' + name + ' = ()\n'
        assert fragment.count(old) == 1
        fragment = fragment.replace(old, '_EDBAND_' + name + ' = ' + repr(value) + '\n')
    # Fragment has no einsum at all: no new signature or operand reuse is hidden.
    assert signatures(fragment) == []
    rendered = parent_text.replace(edits[0][0], edits[0][1]) + '\n' + fragment
    assert signatures(rendered) == signatures(parent_text)
    compile(rendered, '<EDBAND text-only candidate>', 'exec')
    metadata = dict(schema='edge_band_v1', parent_sha256=psha, edges=edges,
                    new_einsum_signatures=[], one_time_arrays=sum(v is not None for v in bias + output),
                    payload_bytes=64 * sum(v is not None for v in bias + output))
    return edits, fragment, metadata
