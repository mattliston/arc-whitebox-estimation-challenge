"""Add only the omitted pinned setup_clock record to frozen complete-cohort plans.

No truth or fixture arrays, estimator imports, scoring, or native dispatch.
The original plans/proofs and failed run remain immutable. Every inherited
truth proof, source, split, SDK pin, guard pin, and numeric setting is identical.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / 'cloud_fresh4_v1/PLAN_COMPRESSION_CONFIRM_FRESH4_MARKLINE_v1.json'
TEMPLATE_SHA = 'e0f3621a326995f48b539aa8cb3138be7a125334da0592aef880f3e5a4d10bff'
TRUTH_BUILDER_SHA = '676f4d0bef620b0886b9d69f298d54ca3b1d771e15a4f32f4893fd81ac30777b'
COHORTS = {
    'original8': ('fe9ed4bb88d0f5b27b2a44a458a8ae32f4a2624a51f8c33b23a1e994b38de1ce',
                  '16f19788bafa3d186049fae5ead4f93d6dc50758379258316366368d8e9c3008'),
    'extension4': ('c26d23e138ce771ce071df4d33ee2eb330e8390df2aec5ea6c4fab913b78ca8c',
                   'd76020d9c40b12198ada1b28168225f53b25366b2fadaf1ca4f32e877951f97a'),
}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def record(root, path):
    return dict(path=str(path.resolve().relative_to(root.resolve())), sha256=sha(path))


def inputs(root, cohort):
    parent = HERE / ('four_source_' + cohort + '_v1/PLAN_FOUR_SOURCE_v1.json')
    proof_path = parent.with_suffix('.PROOF.json')
    assert (sha(parent), sha(proof_path)) == COHORTS[cohort]
    assert sha(TEMPLATE) == TEMPLATE_SHA
    assert sha(HERE / 'build_four_source_plan_v1.py') == TRUTH_BUILDER_SHA
    old, proof = read(parent), read(proof_path)
    assert proof['successful'] and proof['plan_sha256'] == sha(parent)
    assert proof['builder_sha256'] == TRUTH_BUILDER_SHA and proof['cohort'] == cohort
    assert proof['all_cohort_complete_before_truth_values_opened'] and proof['no_fit']
    assert not proof['estimator_imported'] and not proof['native_launched']
    assert 'setup_clock' not in old
    clock = read(TEMPLATE)['setup_clock']
    assert sha(root / clock['path']) == clock['sha256']
    assert (root / clock['path']).stat().st_size == clock['bytes']
    return parent, proof_path, old, proof, clock


def verify_schema_repair(root, plan_path, proof):
    cohort = proof['cohort']
    parent, parent_proof, old, inherited, clock = inputs(root, cohort)
    plan = read(plan_path)
    expected = copy.deepcopy(old)
    expected['setup_clock'] = clock
    assert plan == expected, 'Only setup_clock may change in the frozen plan'
    assert proof['plan_sha256'] == sha(plan_path) and proof['builder_sha256'] == sha(__file__)
    assert proof['schema_parent_plan'] == record(root, parent)
    assert proof['schema_parent_proof'] == record(root, parent_proof)
    assert proof['original_truth_builder_sha256'] == inherited['builder_sha256']
    assert proof['schema_repair_only_keys'] == ['setup_clock']
    for key, value in inherited.items():
        if key not in ('plan_sha256', 'builder_sha256'):
            assert proof[key] == value, key
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cohort', choices=tuple(COHORTS), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists() and not a.out.with_suffix('.PROOF.json').exists()
    parent, parent_proof, old, inherited, clock = inputs(a.root, a.cohort)
    plan = copy.deepcopy(old)
    plan['setup_clock'] = clock
    # Audit every literal top-level plan subscription in every frozen worker,
    # including the inherited common implementation imported by common_v7.
    audited = []
    common = HERE.parents[2] / 'astra_oct20_bill_v1/cloud_linux_v1/common_linux_v1.py'
    assert sha(common) == '13aead70dfce132335df80829c63bbfacc039d823dd67b9276746e7046004f37'
    paths = [a.root / path for path in plan['worker_source_pins']] + [common]
    required = set()
    for path in paths:
        if str(path.relative_to(a.root)) in plan['worker_source_pins']:
            assert sha(path) == plan['worker_source_pins'][str(path.relative_to(a.root))]
        refs = sorted({n.slice.value for n in ast.walk(ast.parse(path.read_text()))
                       if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                       and n.value.id == 'plan' and isinstance(n.slice, ast.Constant)
                       and isinstance(n.slice.value, str)})
        required.update(refs)
        audited.append(dict(**record(a.root, path), required_plan_keys=refs))
    assert required <= set(plan), sorted(required - set(plan))
    assert plan['warmup_each'] is False and plan['first7'] == plan['held25'] == []
    assert all(len(group) == 1 for group in plan['groups'])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(plan, f, indent=2); f.write('\n')
    proof = copy.deepcopy(inherited)
    proof.update(plan_sha256=sha(a.out), builder_sha256=sha(__file__),
                 original_truth_builder_sha256=inherited['builder_sha256'],
                 schema_parent_plan=record(a.root, parent), schema_parent_proof=record(a.root, parent_proof),
                 schema_repair_only_keys=['setup_clock'], schema_repair_utc=utc(),
                 schema_template=record(a.root, TEMPLATE), schema_required_keys=sorted(required),
                 schema_workers=audited, schema_repair_no_array_reads=True)
    with a.out.with_suffix('.PROOF.json').open('x') as f:
        json.dump(proof, f, indent=2); f.write('\n')
    assert verify_schema_repair(a.root, a.out, proof)
    print(json.dumps(dict(plan=str(a.out), sha256=sha(a.out),
                         proof_sha256=sha(a.out.with_suffix('.PROOF.json')), cohort=a.cohort)))


if __name__ == '__main__':
    main()
