"""Repair public evaluation status schema; preserve the failed v1 execution."""
from pathlib import Path
import copy
import difflib
import hashlib
import json
import sys

D = Path(__file__).resolve().parent
H = D.parent


def pin(p):
    p = Path(p).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def once(p, obj):
    with p.open('x') as f: json.dump(obj, f, indent=2, allow_nan=False); f.write('\n')


def main():
    assert sys.argv[1:] in ([], ['--enable'])
    enabled = sys.argv[1:] == ['--enable']
    previous_engine = D/'boundary_assess_s34_reference.py'
    assert pin(previous_engine)['sha256'] == 'c4ec222c7426c35f23b04c11103e0bc8dbbdf3cc517c35a4fc95945edef8578d'
    old = previous_engine.read_text()
    before = "assert evaluation['submission_id'] == 333050 and evaluation['grading_status_cd'] == 'graded'"
    after = "assert evaluation['submission_id'] == 333050 and evaluation['grading_status_cd'] is None"
    assert old.count(before) == 1
    new = old.replace(before, after, 1)
    before = "    ev = evaluation['evaluation']\n"
    after = before + "    assert ev['state'] == 'completed' and ev['error'] is None\n"
    assert new.count(before) == 1
    new = new.replace(before, after, 1)
    new = new.replace('boundary_s34_reference_admission.json', 'boundary_s34_reference_v2_admission.json')
    new = new.replace('boundary_s34_reference_assessment.json', 'boundary_s34_reference_v2_assessment.json')
    new = new.replace('boundary_s34_reference_assess_guard.json', 'boundary_s34_reference_v2_assess_guard.json')
    engine = D/'boundary_assess_s34_reference_v2.py'
    repair_path = D/'s34_reference_v2_technical_repair.json'
    if enabled:
        assert engine.read_text() == new
    else:
        with engine.open('x') as f: f.write(new)
        with (D/'s34_reference_v2.diff').open('x') as f:
            f.write(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=previous_engine.name, tofile=engine.name)))
        once(repair_path, dict(reason='Pinned public evaluation has no grading_status_cd; completed/error-free state is nested in evaluation. Authenticated bank separately proves graded.',
            old_engine=pin(previous_engine), new_engine=pin(engine), failed_guard=pin(D/'boundary_s34_reference_assess_guard.json'),
            old_boundary_config=pin(D/'boundary_s34_reference_config.json'), old_native_config=pin(D/'native_screen_s34_reference.json'),
            old_native2=pin(D/'native2_s34_reference_assessment.json'), old_native2_guard=pin(D/'native2_s34_reference_assess_guard.json'),
            source_changes=False, score_or_resource_gate_changes=False, timing_or_model_rerun=False,
            original_failure_and_v1_guard_preserved=True, builder=pin(__file__), diff=pin(D/'s34_reference_v2.diff')))
    c = json.loads((D/'boundary_s34_reference_config.json').read_text())
    c['prospective_authorization'] = enabled
    c['engine'] = pin(engine)
    c['native_config_path'] = str(D/'native_screen_s34_reference_v2.json')
    c['technical_repair'] = pin(repair_path)
    c['provenance'] += [pin(__file__), pin(engine), pin(repair_path), pin(D/'s34_reference_v2.diff'),
        pin(D/'boundary_s34_reference_config.json'), pin(D/'boundary_s34_reference_assess_guard.json')]
    boundary = D/('boundary_s34_reference_v2_config.json' if enabled else 'boundary_s34_reference_v2_config_draft.json')
    once(boundary, c)
    original = json.loads((D/'native_screen_s34_reference.json').read_text())
    nc = copy.deepcopy(original)
    nc['prospective_authorization'] = enabled
    nc['stages']['2']['result'] = str(D/'native2_s34_reference_v2_assessment.json')
    nc['stages']['2']['assessment_guard'] = str(D/'native2_s34_reference_v2_assess_guard.json')
    req = nc['boundary_before8']
    req['assessor'] = pin(engine)
    req['receipt'] = str(D/'boundary_s34_reference_v2_assessment.json')
    req['assessment_guard'] = str(D/'boundary_s34_reference_v2_assess_guard.json')
    req['assessment_command'] = [str(H.parents[4]/'.venv/bin/python'), '-B', str(engine), str(D/'boundary_s34_reference_v2_config.json'), pin(boundary)['sha256']]
    nc['provenance'] += [pin(__file__), pin(engine), pin(boundary), pin(repair_path), pin(D/'native_screen_s34_reference.json'),
        pin(D/'native2_s34_reference_assessment.json'), pin(D/'native2_s34_reference_assess_guard.json')]
    assert all(nc[k] == original[k] for k in ('arms', 'engine', 'parity_layers', 'future_only', 'raw_error_veto'))
    assert all(nc['stages'][s]['gates'] == original['stages'][s]['gates'] for s in ('2', '8'))
    output = D/('native_screen_s34_reference_v2.json' if enabled else 'native_screen_s34_reference_v2_draft.json')
    once(output, nc)
    print(json.dumps(dict(enabled=enabled, engine=pin(engine), boundary_config=pin(boundary), native_config=pin(output), builder=pin(__file__), repair=pin(repair_path))))


if __name__ == '__main__': main()
