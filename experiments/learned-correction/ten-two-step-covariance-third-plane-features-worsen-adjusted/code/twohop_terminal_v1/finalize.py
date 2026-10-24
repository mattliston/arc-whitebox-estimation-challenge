"""Authenticate the fixed experiment and preserve its completed disposition."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
ROOT = Path('/Users/matt/arc-white-box-estimation')
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
NETS = [35, 36, 38, 40, 41, 43, 44, 45, 46, 47,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58]


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    now = datetime.now(timezone.utc).isoformat()
    source_seal = read(HERE / 'source_sealed.json')
    for name, expected in source_seal['files_sha256'].items():
        assert sha(HERE / name) == expected, name
    for record in read(HERE / 'source_review.json')['sources']:
        assert sha(P2 / record['path']) == record['sha256'], record['path']
    certificate = read(HERE / 'certificate.json')
    assert certificate['passed']
    assert certificate['feature_sha256'] == sha(HERE / 'feature.py')
    qualification = read(HERE / 'qualification/completed.json')
    assert qualification['completed'] and qualification['repeat_byte_exact']
    assert qualification['parent_all16_and_flops_exact']
    assert qualification['source_proof']['inverse_byte_exact']
    assert qualification['source_proof']['extra_numeric_operations_inside_parent'] == 0
    manifest = read(HERE / 'train_capture/manifest.json')
    assert manifest['completed'] and not manifest['truth_accessed']
    assert manifest['train'] == NETS and len(manifest['rows']) == 20
    assert manifest['source_sha256'] == sha(HERE / 'capture_all.py')
    assert manifest['feature_source_sha256'] == sha(HERE / 'feature.py')
    completions = {HERE / 'qualification/completed.json'}
    watchers = {HERE / 'certificate_watch.json', HERE / 'qualification_watch.json',
                HERE / 'information_watch.json'}
    for row in manifest['rows']:
        completion = Path(row['completion'])
        assert sha(completion) == row['completion_sha256']
        completions.add(completion)
        watchers.add(Path(row['watch']))
        rec = row['record']
        assert rec['rank'] == row['rank'] and rec['includes_initialization']
        assert sha(rec['capture']) == rec['capture_sha256']
    calls = []
    for path in sorted(completions):
        completed = read(path)
        assert completed['completed'] and not completed['truth_accessed']
        assert completed['source_sha256'] == sha(HERE / 'capture.py')
        assert completed['feature_source_sha256'] == sha(HERE / 'feature.py')
        assert completed['protocol_sha256'] == sha(HERE / 'PROTOCOL.md')
        assert completed['source_proof'] == qualification['source_proof']
        for rec in completed['records']:
            assert rec['candidate_sha256'] == sha(D / 'lazy_rotation_objective_v1/candidate.py')
            assert sha(rec['capture']) == rec['capture_sha256']
            assert rec['feature_flops'] == 6_489_753_600
            assert not rec['truth_accessed']
            calls.append(rec)
    assert len(calls) == 22 and len(watchers) == 22
    watch_records = []
    for path in sorted(watchers):
        watch = read(path)
        assert watch['completed'] and watch['exit_code'] == 0
        assert watch['whole_group_retired'] and watch['error'] is None
        assert watch['max_swap_growth_bytes'] == 0
        assert watch['max_rss_bytes'] < 8_000_000_000
        watch_records.append(watch)
    sealed = read(HERE / 'information/inputs_sealed.json')
    assert not sealed['targets_opened']
    assert sealed['capture_manifest_sha256'] == sha(HERE / 'train_capture/manifest.json')
    assert sealed['source_sha256'] == sha(HERE / 'information.py')
    assert sealed['ranks'] == NETS
    oof = read(HERE / 'information/oof_sealed.json')
    assert not oof['selection_used']
    assert oof['predictions_sha256'] == sha(HERE / 'information/oof_predictions.npz')
    for fit in oof['fits']:
        assert sha(HERE / f"information/fold{fit['fold']}_fit.npz") == fit['fit_sha256']
        assert set(fit['train_ranks']).isdisjoint(fit['check_ranks'])
        assert set(fit['train_ranks'] + fit['check_ranks']) == set(NETS)
    for rank, expected in oof['target_sha256'].items():
        path = P2 / f'public_g22_operator_features_v1/truthpost/truthpost_rank{int(rank):04d}.npz'
        assert sha(path) == expected, rank
    result = read(HERE / 'information/result.json')
    assert result['schema'] == 'twohop_terminal_information_v1'
    assert result['source_sha256'] == sha(HERE / 'information.py')
    assert result['capture_manifest_sha256'] == sha(HERE / 'train_capture/manifest.json')
    assert result['coefficient_projection_train_only']
    assert not result['deployment_bill_qualified']
    expected_pass = (result['extended_to_unchanged'] <= .95 and
                     result['extended_to_control'] <= .97 and result['fold_wins'] >= 4)
    assert result['gate_passed'] == expected_pass
    peak = max([w['max_rss_bytes'] for w in watch_records] +
               [r['max_rss_bytes'] for r in calls] + [result['max_rss_bytes']])
    assert peak < 8_000_000_000
    official = read(D / 'official_best_331599.json')
    assert official['grading_status'] == 'graded' and official['below_4e9']
    assert official['archive_sha256'] == sha(D / 'lazy_rotation_objective_v1/submission-lazy-objective-20260919.tar.gz')
    status = 'information gate passed; deployment unqualified' if expected_pass else 'closed negative at fixed information gate'
    lines = [
        '# Fixed terminal two-step operator result', '',
        status + '.', '',
        'The ten paid full-matrix covariance/third-plane features were evaluated',
        'on the predeclared TRAIN20 networks with five network folds. All normalizers,',
        'weighted projection and ridge coefficients used each fold\'s training networks only.',
        'These are reused research networks; this is exploratory cross-validation.', '',
        '| Arm | Mean raw final MSE | Mean adjusted score |',
        '| --- | ---: | ---: |',
    ]
    for arm in ['unchanged', 'control', 'extended']:
        raw = sum(r[arm]['raw'] for r in result['rows']) / len(result['rows'])
        lines.append(f"| {arm} | {raw:.16g} | {result['pooled_adjusted'][arm]:.16g} |")
    lines += ['', f"Extended/unchanged: {result['extended_to_unchanged']:.16g}.",
              f"Extended/affine control: {result['extended_to_control']:.16g}.",
              f"Fold wins against both: {result['fold_wins']}/5. Fixed gate passed: {expected_pass}.", '',
              'Each extended score charges its own measured cold parent FLOPs plus',
              '6,489,753,600 measured feature FLOPs and the fixed 2MF readout reserve.',
              'This is an information-test bill; no production candidate or client bill is claimed.', '',
              'The independent explicit-matrix and duplicate-span/augmented-solve',
              'certificates passed. Full35/100/35 observation preserved all16 parent outputs',
              'and parent FLOP bills exactly; repeat features were byte-identical.',
              'The parent physical state and submitted archive were never modified.', '',
              f'22 full predictions ran serially. Peak RSS including self-reports: {peak} bytes.',
              'All22 guarded stages exited0, added no swap, and retired their groups.',
              'Only compact terminal arrays were stored; no full covariance tapes.', '']
    if expected_pass:
        lines += ['A separate fixed full fit, CHECK evaluation, native/client qualification and',
                  'panel validation are required before any deployable-candidate claim.', '']
    else:
        lines += ['The fixed arm is closed. No CHECK targets, full fit, operator/layer/penalty',
                  'sweep, production port, archive or submission follows this failed gate.',
                  'This result does not prove that every joint operator or cumulant closure fails.', '']
    lines += [f"Official bank remains Lazy #331599 at {official['adjusted']:.16g} adjusted.",
              'The4e-9 milestone is confirmed; the approximately1e-9 goal remains active and unmet.', '',
              f'Recorded UTC: {now}.', '']
    report_name = 'INFORMATION_RESULT.md' if expected_pass else 'NEGATIVE_RESULT.md'
    with (HERE / report_name).open('x') as stream:
        stream.write('\n'.join(lines))
    resource = dict(full_prediction_calls=len(calls), guarded_stages=len(watch_records),
                    parent_flops=sum(r['parent_flops'] for r in calls),
                    feature_flops=sum(r['feature_flops'] for r in calls), peak_rss_bytes=peak,
                    additional_swap_bytes=0, all_owned_groups_retired=True,
                    total_guarded_elapsed_s=sum(w['elapsed_s'] for w in watch_records))
    write_json(HERE / 'resources.json', resource)
    artifacts = {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*'))
                 if p.is_file() and '__pycache__' not in p.parts}
    artifact_bytes = sum((HERE / name).stat().st_size for name in artifacts)
    assert artifact_bytes < 20_000_000
    closeout = dict(created_at_utc=now, status=status, goal_status='active', goal_achieved=False,
                    gate_passed=expected_pass, exploratory_train_network_cv=True,
                    extended_to_unchanged=result['extended_to_unchanged'],
                    extended_to_control=result['extended_to_control'], fold_wins=result['fold_wins'],
                    pooled_adjusted=result['pooled_adjusted'], production_candidate_qualified=False,
                    official_bank=official, resources=resource, artifact_bytes_before_closeout=artifact_bytes,
                    artifacts_sha256=artifacts, remaining_disk_GiB=shutil.disk_usage(ROOT).free / 2**30)
    closeout['external_artifacts_sha256'] = {name: sha(D / name) for name in [
        'official_best_331599.json', 'sub331599_poll_09.json',
        'official_lazy_grade_integrity_v1.json', 'official_lazy_objective_gap_v1.json',
        'official_lazy_objective_gap_v1.md', 'lazy_rotation_objective_v1/OFFICIAL_RESULT.md']}
    write_json(HERE / 'closeout.json', closeout)
    print(json.dumps({k: v for k, v in closeout.items() if k not in ['artifacts_sha256', 'official_bank']}))


if __name__ == '__main__':
    main()
