"""Freeze exact first128 physics parents with paid, fixed SMoff + NPoff routes."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1]
K = RUNS / 'fable_oct19_kc_v1'
A = K / 'agent_astra_joint_v1'
V = RUNS / 'astra_oct20_vm_frontier_v1'
D = RUNS / 'astra_oct20_bill_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()
PARENTS = {'first128': {
    'science': (D / 'breakthrough_v1/first_hidden_readout_v1/native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_science_v1.py', '4da68821a5db9212035531a25dc8ab0c197a44d84a1e5482168d4223d349032c'),
    'release': (D / 'breakthrough_v1/first_hidden_readout_v1/native_v1/edge8_complement_first128_local90/candidate_edge8_complement_first128_local90_release_v1.py', 'e491c4aed9fb0bca8d04f6dad3832144ae996ddf0086277b58eba75c77308bbc'),
}}
tail = V / 'runtime_current337466_v1/tail_smoff_v1.py'
nogov = A / 'tail_disable_fast_rescue_astra_v1.py'
assert sha(tail) == '520d8f4b90463b0ca218930f471f9869a1ea131b7e618816efc146c42a2eabc7'
assert sha(nogov) == 'be4293e44004415fb9faba1b42a26c8428a447ae226475a77a707054aac54607'
reference = K / 'candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py'
assert sha(reference) == '434b3a4730febb062b59a8a29c042d27e1f4bdbf8301cef3a72c90b2e798fcb4'
reference_tree = ast.parse(reference.read_text())
sm_tail = D / 'tail_sm_direct_v1.py'
tiny = V / 'runtime_current337466_v1/cloud_checks_v1/SM_CHECK_v1.json'
tiny_guard = V / 'runtime_current337466_v1/cloud_checks_v1/runtime_current337466_checks_v1_GUARD.json'
tiny_report = json.loads(tiny.read_text())
guard_report = json.loads(tiny_guard.read_text())
assert tiny_report['successful'] and len(tiny_report['tests']) == 36
assert tiny_report['source_sha256'] == sha(reference)
assert guard_report['successful'] and guard_report['whole_group_retired']


def sm_dependencies(tree):
    chosen = []
    selectors = [('_Sm40Prepared', 0, 1000000), ('_astra_sm40_linear_grouped', 20000, 20100),
                 ('_astra_sm40_linear_grouped', 20930, 21000), ('_sm40_program_key', 21420, 21450)]
    for name, lower, upper in selectors:
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name == name and lower <= n.lineno <= upper]
        assert len(nodes) == 1, (name, len(nodes))
        chosen.extend(nodes)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            (isinstance(t, ast.Name) and t.id == '_ASTRA_SM40_RECORDS') or
            (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == '_SMX_RECORDS')
            for t in node.targets):
            chosen.append(node)
    return ast.dump(ast.Module(body=chosen, type_ignores=[]), include_attributes=False)


reference_dependencies = sm_dependencies(reference_tree)
np_tail = V / 'runtime_npxoff_current337466_v1/tail_npxoff_v1.py'
assert sha(np_tail) == '28a225c520f4674b4f532397edf56f541cf10c2ba437124ee0f446f23ef8278c'
np_copy = D / 'tail_npx_copy_exact_v1.py'
np_tiny = V / 'runtime_npxoff_current337466_v1/cloud_checks_v1/NP_CHECK_v1.json'
np_proof = json.loads(np_tiny.read_text())
assert np_proof['successful'] and np_proof['numerical_and_auxiliary_exact'] and len(np_proof['records']) == 12
assert np_proof['source_sha256'] == sha(reference) and np_proof['tail_sha256'] == sha(np_copy)
def np_dependencies(source):
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and (t.id.startswith('_NP_') or t.id == '_S19_LAM') for t in node.targets):
            nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in ('_np_ages', '_np_key', '_np_tables', '_np_const', '_npx_left', '_astra_compute_left_v1'):
            nodes.append(node)
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
np_expected = np_dependencies(reference.read_text())
outputs = {}; cases = []
for key, roles in PARENTS.items():
    outputs[key] = {}
    for role, (parent, digest) in roles.items():
        assert sha(parent) == digest
        source = parent.read_text(); tree = ast.parse(source)
        assert ast.unparse(tree.body[-1]) == '_FAST_RESCUE_FRAC = {}'
        assert sm_dependencies(tree) == reference_dependencies
        # The entire original SMDIRECT implementation is inherited unchanged.
        assert sm_tail.read_text() in source
        assert np_copy.read_text() in source and np_dependencies(source) == np_expected
        target = HERE / f'candidate_{key}_smoff_npxoff_{role}_v1.py'
        payload = parent.read_bytes() + b'\n' + tail.read_bytes() + b'\n' + np_tail.read_bytes() + b'\n' + nogov.read_bytes()
        with target.open('xb') as stream:
            stream.write(payload)
        parsed = ast.parse(payload.decode())
        assert ast.dump(ast.Module(body=parsed.body[:len(tree.body)], type_ignores=[]), include_attributes=False) == ast.dump(tree, include_attributes=False)
        assert ast.unparse(parsed.body[-1]) == '_FAST_RESCUE_FRAC = {}'
        outputs[key][role] = dict(path=str(target), sha256=sha(target), parent_path=str(parent), parent_sha256=digest)
        cases.append(dict(key=key, role=role, exact_parent_prefix=True, sm_dependencies_ast_exact=True, sm_direct_tail_byte_exact=True, np_dependencies_ast_exact=True, np_copy_tail_byte_exact=True, final_empty_rescue_map=True))
report = dict(utc=utc(), status='SOURCE_BUILT_IMPORT_PENDING_NO_NATIVE', outputs=outputs, cases=cases,
              tail=dict(path=str(tail), sha256=sha(tail)), nogov=dict(path=str(nogov), sha256=sha(nogov)),
              tiny_reuse=dict(path=str(tiny), sha256=sha(tiny), guard_path=str(tiny_guard), guard_sha256=sha(tiny_guard), cases=36,
                              exact_extracted_dependencies_sha256=hashlib.sha256(reference_dependencies.encode()).hexdigest()),
              np_tiny_reuse=dict(path=str(np_tiny), sha256=sha(np_tiny), cases=12, dependency_sha256=hashlib.sha256(np_expected.encode()).hexdigest()), np_tail=dict(path=str(np_tail), sha256=sha(np_tail)),
              no_governor=True, no_fire=True, no_native=True,
              limitations=['The full first128 SMoff+NPoff bill must be measured and included in adjusted scores.',
                           'Previous434b instructions and bill do not qualify these stacks.',
                           'Final stack needs original paired runtime measurement.'])
with (HERE / 'BUILD_v1.json').open('x') as stream:
    json.dump(report, stream, indent=2); stream.write('\n')
print(json.dumps(dict(utc=report['utc'], outputs=outputs, status=report['status'])))
