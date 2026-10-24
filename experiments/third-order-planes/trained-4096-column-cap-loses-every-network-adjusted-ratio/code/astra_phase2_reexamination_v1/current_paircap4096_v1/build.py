"""Inert exclusive source builder for one fixed current409f pair cap.

Only --build writes source and metadata. It imports no numerical package,
reads no model/weight/target arrays, and executes no generated estimator.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'unused_materializations_v1/estimator.py'
PARENT_SHA = '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f'
MARKER = '\n\n# BEGIN INERT CURRENT409F SIGNED-PAIR-CAP4096 V1 APPENDIX\n'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'Source changed during hashing: ' + str(path))
    return dict(path=str(path), sha256=digest, bytes=after.st_size, mtime_ns=after.st_mtime_ns)


def last_function(source, name):
    found = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    require(bool(found), 'Missing original function: ' + name)
    node = found[-1]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def derived_predict(parent):
    old = last_function(parent, '_predict_binary')
    modifications = [
        ('    newborn = None\n',
         '    newborn = None\n'
         '    pair_positive = fnp.zeros(0, dtype=fnp.int64)\n'
         '    pair_negative = fnp.zeros(0, dtype=fnp.int64)\n', 1),
        ('        pp = _push_bank_shared_newborn(w32, p, previous, newborn)\n',
         '        pp = _paircap_push_bank_shared_newborn(w32, p, previous, newborn)\n', 1),
        ('        lam = fnp.concatenate([lam, a[selected].astype(fnp.float32), -a[selected].astype(fnp.float32)])\n',
         '        lam = fnp.concatenate([lam, a[selected].astype(fnp.float32), -a[selected].astype(fnp.float32)])\n'
         '        p, lam, newborn, pair_positive, pair_negative, cp_post = _paircap_retire(\n'
         '            p, lam, newborn, pair_positive, pair_negative, cp_post, live_indices, _PAIR_CAP)\n', 1),
    ]
    new = old
    for before, after, count in modifications:
        require(new.count(before) == count, 'Unexpected parent edit count')
        new = new.replace(before, after)
    inverse = new
    for before, after, count in reversed(modifications):
        require(inverse.count(after) == count, 'Ambiguous derived-function inverse')
        inverse = inverse.replace(after, before)
    require(inverse == old, 'Derived function changed outside the declared three sites')
    return old, new


def build():
    destination, receipt = HERE / 'estimator.py', HERE / 'build.json'
    require(not destination.exists() and not receipt.exists(), 'Exclusive new source build only')
    paths = (PARENT, HERE / 'core.py', HERE / 'PROTOCOL.md', Path(__file__).resolve())
    inputs = {str(path.resolve()): binding(path) for path in paths}
    require(inputs[str(PARENT.resolve())]['sha256'] == PARENT_SHA, 'Frozen current409f source pin')
    parent = PARENT.read_bytes().decode()
    core = (HERE / 'core.py').read_bytes().decode()
    require(MARKER not in parent and MARKER not in core, 'Ambiguous appendix boundary')
    core_tree = ast.parse(core)
    for node in core_tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.FunctionDef):
            continue
        require(isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == '_PAIR_CAP'
                and ast.literal_eval(node.value) == 4096, 'Core has an active/import/unknown top-level statement')
    caps = [node for node in core_tree.body if isinstance(node, ast.Assign)]
    require(len(caps) == 1, 'One fixed pair-cap literal required')
    old, derived = derived_predict(parent)
    source = parent + MARKER + core + '\n\n' + derived
    tree = ast.parse(source)
    require(source.split(MARKER)[0].encode() == PARENT.read_bytes(), 'Parent source prefix byte proof')
    require(last_function(source, '_predict_binary') == derived, 'Derived route is not effective')
    for name in ('features', '_nonlinear_pair_batch', '_nonlinear_terminal_live',
                 'omitted_radial_source', '_star_compact', '_cov_compact', '_mm'):
        require(last_function(source, name) == last_function(parent, name),
                'Frozen physical/head/kernel function changed: ' + name)
    parent_classes = [ast.dump(n, include_attributes=False) for n in ast.parse(parent).body
                      if isinstance(n, ast.ClassDef)]
    candidate_classes = [ast.dump(n, include_attributes=False) for n in tree.body
                         if isinstance(n, ast.ClassDef)]
    require(parent_classes == candidate_classes, 'Estimator/head class changed')
    require({str(path.resolve()): binding(path) for path in paths} == inputs,
            'Inputs changed before source creation')
    record = dict(schema='current_paircap4096_inert_source_v1', passed=True,
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  parent_sha256=PARENT_SHA, candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  candidate=str(destination.resolve()), source_bindings=inputs,
                  signed_pair_cap=4096, physical_column_cap=8192,
                  parent_source_prefix_byte_exact=True, predict_three_edit_inverse_byte_exact=True,
                  all_parent_payload_and_function_text_preserved=True,
                  effective_predict_replaced=True, head_model_and_all16_routing_unchanged=True,
                  new_imports=False, network_or_target_access=False, numerical_execution=False,
                  generated_candidate_executed=False, tiny_qualification_passed=False,
                  full_cost_measured=False, numerical_or_training_admission=False,
                  approximation='Whole-pair retirement into current repeated-plane star; future all-distinct tensor is changed',
                  note='An inert source artifact, not a qualified estimator, successful cost gate, or training permission')
    with destination.open('xb') as stream:
        stream.write(source.encode())
    require(sha(destination) == record['candidate_sha256'], 'Written candidate hash mismatch')
    with receipt.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    if args.build:
        build()
    else:
        parser.print_help()
