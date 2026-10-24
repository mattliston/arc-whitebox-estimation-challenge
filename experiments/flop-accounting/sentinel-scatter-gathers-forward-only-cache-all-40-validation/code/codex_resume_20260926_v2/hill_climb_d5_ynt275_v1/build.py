"""Stdlib-only construction of one fixed young-negative threshold pilot."""
from pathlib import Path
import ast
import copy
import difflib
import hashlib
import json

D = Path(__file__).resolve().parent
H = D.parent
P = H / 'hill_climb_e1_ynt_psdb_d5_v1'
Y = H / 'hill_climb_young_negative_trace_v1'
T = H / 'hill_climb_d5_late_common224_v1'
PARENT_SHA = '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
FRAGMENT_SHA = '85e75784afb7f3d79854cd5599f4e7f77a15bdef117f6edccb7f23fa85ec70c0'
TEMPLATE_SHA = '5381e5d072a1add9517826c48f317a26ba9f05e1c29729cabfbcf85bd9ccb5b7'
EDITS = (
    ("negative = (ctx['a'] < -2.5) & (~ctx['live'])",
     "negative = (ctx['a'] < -2.75) & (~ctx['live'])"),
    ("score = fnp.where(negative, ctx['a'] + 2.5, 0.0)",
     "score = fnp.where(negative, ctx['a'] + 2.75, 0.0)"),
)


def pin(path):
    p = Path(path).resolve()
    b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def once(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    parent, original_fragment, template = pin(P/'candidate.py'), pin(Y/'fragment.py'), pin(T/'native_screen.json')
    assert parent['sha256'] == PARENT_SHA
    assert original_fragment['sha256'] == FRAGMENT_SHA
    assert template['sha256'] == TEMPLATE_SHA
    base = (P/'candidate.py').read_text()
    old = (Y/'fragment.py').read_text()
    assert base.count(old) == 1
    fragment = old
    for before, after in EDITS:
        assert fragment.count(before) == 1 and after not in fragment
        fragment = fragment.replace(before, after, 1)
    child = base.replace(old, fragment, 1)
    assert child.count(fragment) == 1 and child.replace(fragment, old, 1) == base
    inverse = child
    for before, after in EDITS:
        assert inverse.count(after) == 1
        inverse = inverse.replace(after, before, 1)
    assert inverse == base
    ast.parse(child)
    delta = [(i, a, b) for i, (a, b) in enumerate(zip(base.splitlines(), child.splitlines()), 1) if a != b]
    assert len(base.splitlines()) == len(child.splitlines()) and len(delta) == 2
    assert [(a.strip(), b.strip()) for _, a, b in delta] == list(EDITS)
    for name, content in [('candidate.py', child), ('fragment.py', fragment),
                          ('source_diff.patch', ''.join(difflib.unified_diff(base.splitlines(True), child.splitlines(True), fromfile='D5/candidate.py', tofile=D.name+'/candidate.py')))]:
        with (D/name).open('x') as f:
            f.write(content)
    candidate = pin(D/'candidate.py')
    prior = json.loads((P/'native8_official_limit_extension.json').read_text())
    assert prior['passed_integrity'] and prior['admitted'] and prior['candidate'] == parent
    broad = json.loads((P/'broad_assessment.json').read_text())
    assert broad['passed'] and broad['candidate'] == parent
    protocol = dict(root_selected=True, prospective_scope='native2_only',
        proposal='One fixed YNT cutoff 2.75 selected before candidate outcomes, replacing the two 2.5 literals inside the exact original fragment.',
        parent=parent, candidate=candidate, original_fragment=original_fragment,
        parent_status='Fully locally qualified submission 333050; official grade pending at proposal selection.',
        basis=pin(P/'broad_assessment.json'),
        observed_parent_valid40_vs_s33={k: broad['metrics'][k] for k in ('raw_pct', 'bill_pct', 'adjusted_pct')},
        hypothesis='A stricter negative-tail cutoff may recover raw accuracy while giving back some bill savings. Block rounding and changed trajectories make the net adjusted effect uncertain.',
        fixed_threshold=2.75, fixed_train2=[35, 100],
        stage2_gates=dict(adjusted_pct_at_most=-.1, wins_min=1, max_per_network_regression_pct=1),
        parity_layers=list(range(7)), no_neighbor_sweep=True, no_result_based_retuning=True,
        wrapper_and_capture_changes=False, source_inverse_exact=True,
        native_driver=pin(P/'native_probe.py'), limits=dict(seconds=150, rss_bytes=7500000000, swap_growth_bytes=0),
        execution='Root reviews source/config pins and launches the existing guarded native driver. This builder executes no model.',
        future_only=True, reused_training_not_pristine=True,
        later_authority=False, bank_authority=False,
        later_work='A pilot gain is complementary research evidence only. Additional networks, its own boundary and broad/serving validation require separate admission; no inherited D5 resource pass.')
    once(D/'protocol.json', protocol)
    once(D/'build.json', dict(parent=parent, candidate=candidate, builder=pin(__file__),
        protocol=pin(D/'protocol.json'), original_fragment=original_fragment, fragment=pin(D/'fragment.py'),
        diff=pin(D/'source_diff.patch'), source_inverse_exact=True, edits=delta,
        only_two_numeric_literals_changed=True, wrappers_and_captures_unchanged=True,
        numerical_execution=False, template_schema_only=template))
    nc = copy.deepcopy(json.loads((T/'native_screen.json').read_text()))
    control = copy.deepcopy(json.loads((P/'native_screen.json').read_text())['arms']['candidate'])
    for b in prior['blocks']:
        control['blocks'][str(b['size'])]['artifacts'] = b['arms']['candidate']
    nc['arms'] = dict(candidate=copy.deepcopy(control), base=copy.deepcopy(control), parity_parent=copy.deepcopy(control))
    nc['arms']['candidate']['source'] = candidate
    for key, block in nc['arms']['candidate']['blocks'].items():
        block.pop('artifacts', None)
        block['tag'] = D.name+'_train'+key+'_v1'
        block['guard'] = str(D/('native'+key+'_guard.json'))
    nc['parity_layers'] = list(range(7))
    for key, stage in nc['stages'].items():
        stage['result'] = str(D/('native'+key+'_assessment.json'))
        stage['assessment_guard'] = str(D/('native'+key+'_assess_guard.json'))
    assert nc['stages']['2']['gates'] == dict(adjusted_pct_at_most=-.1, strict_adjusted=False, wins_min=1,
        ci_upper_below=None, max_per_network_regression_pct=1, additional6_gain_required=False)
    req = nc['boundary_before8']
    req['receipt'] = str(D/'boundary_assessment_not_yet_admitted.json')
    req['assessment_guard'] = str(D/'boundary_assess_guard_not_yet_admitted.json')
    req['assessment_command'][3] = str(D/'boundary_config_not_yet_admitted.json')
    req['assessment_command'][4] = '0'*64
    nc['provenance'] = [pin(__file__), pin(D/'protocol.json'), pin(D/'build.json'), pin(D/'fragment.py'),
        pin(D/'source_diff.patch'), template, parent, original_fragment, pin(Y/'PROTOCOL.md'),
        pin(P/'native_screen.json'), pin(P/'native8_official_limit_extension.json'),
        pin(P/'native8_official_limit_extension_guard.json'), pin(P/'broad_assessment.json'),
        pin(P/'FIRE_CHECKS_v1.json'), pin(P/'submission_receipt_v1.json')]
    nc['interpretation'] = 'Prospective one-candidate N2 YNT2.75 pilot against locally qualified D5. Exact L0-6 parity; original fixed cheap gates. No additional-network, boundary, broad or bank authority. D5 controls are pinned from its genuine native8 extension; the failed late-common study supplies schema only.'
    nc['prospective_scope'] = 'native2_only'
    once(D/'native_screen.json', nc)
    print(json.dumps(dict(candidate=candidate, config=pin(D/'native_screen.json'), protocol=pin(D/'protocol.json'),
        builder=pin(__file__), build=pin(D/'build.json'), diff=pin(D/'source_diff.patch'), edits=delta)))


if __name__ == '__main__':
    main()
