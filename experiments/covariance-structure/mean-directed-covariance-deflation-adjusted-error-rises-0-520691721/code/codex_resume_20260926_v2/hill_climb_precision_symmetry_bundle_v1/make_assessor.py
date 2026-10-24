"""Stdlib adaptation of the reviewed fixed-panel assessor, after source sealing."""
from pathlib import Path
import ast
import hashlib
import json

D = Path(__file__).resolve().parent
base = D.parent/'hill_climb_star32_q2_v1/assess.py'
assert hashlib.sha256(base.read_bytes()).hexdigest() == 'df446e2247650d50bd4793fecdc832dd70cb3e0a62c688bac049eb11aa37879d'
source = base.read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'build_check')
lines = source.splitlines(keepends=True)
replacement = '''def build_check():
    build = load(D/'build.json')
    for record in build['files']:
        checked(record)
    assert build['candidate'] == pin(D/'candidate.py') and build['candidate']['sha256'] == CANDIDATE_SHA
    assert build['parent']['sha256'] == PARENT_SHA and build['inverse_exact']
    assert pin(D/'PROTOCOL.md')['sha256'] == PROTOCOL_SHA
    parent, child = checked(build['parent']).decode(), checked(build['candidate']).decode()
    suffix = '\\n\\n' + checked(pin(D/'symmetry.py')).decode()
    assert child.endswith(suffix) and len(suffix.encode()) == build['symmetry_suffix_bytes']
    reconstructed = child[:-len(suffix)]
    edits = load(D/'edits.json')
    assert len(edits) == build['precision_edits']
    for edit in reversed(edits):
        assert reconstructed.count(edit['after']) == 1
        reconstructed = reconstructed.replace(edit['after'], edit['before'], 1)
    assert reconstructed == parent
    primitive = load(D/'symmetry_result.json')
    assert primitive['passed'] and primitive['source_sha256'] == CANDIDATE_SHA
    assert primitive['protocol_sha256'] == PROTOCOL_SHA and primitive['build_sha256'] == pin(D/'build.json')['sha256']
    assert primitive['probe_sha256'] == pin(D/'symmetry_probe.py')['sha256']
    guard(D/'symmetry_guard.json', 45, [str(NATIVE), '-B', str(D/'symmetry_probe.py')])
    checked(pin(D/'assess.py'))
    assert pin(DRIVER)['sha256'] == DRIVER_SHA
    checked(pin(DRIVER))
    return build['candidate']
'''
source = ''.join(lines[:node.lineno-1]) + replacement + ''.join(lines[node.end_lineno:])
source = source.replace('hill_climb_star32_q2', 'hill_climb_precision_symmetry_bundle')
build = json.loads((D/'build.json').read_text())
source = source.replace("CANDIDATE_SHA = 'ac27b278dbcbc3cbe79271b32d134d533451d8ebc629bb3b28a80f352000df2a'", 'CANDIDATE_SHA = ' + repr(build['candidate']['sha256']))
source = source.replace("PROTOCOL_SHA = '681cc05b57cf1d91129ad35dcb12daec9e1711f63964e3cb629332b669442290'", 'PROTOCOL_SHA = ' + repr(hashlib.sha256((D/'PROTOCOL.md').read_bytes()).hexdigest()))
ast.parse(source)
with (D/'assess.py').open('x') as stream:
    stream.write(source)
print(json.dumps(dict(assessor_sha256=hashlib.sha256(source.encode()).hexdigest(), candidate_sha256=build['candidate']['sha256'])))
