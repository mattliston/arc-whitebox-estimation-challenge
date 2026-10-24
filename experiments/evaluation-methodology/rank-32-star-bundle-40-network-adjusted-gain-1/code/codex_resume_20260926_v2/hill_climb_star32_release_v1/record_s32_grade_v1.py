"""Record an improved, completely graded S32 from two explicit saved snapshots only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tarfile

from prepare_submission import HERE, H2, SHA, check_tree, once, pin, read, verify_gate

SID = 332738
ARCHIVE_SHA = '92d17642a1b4728a45474c128e4a625937225f158fa5b270738f5fc4f3a0d8f0'
BASE_SCORE = 2.1736465056908115e-9
BASE_RAW = 1.6931451014556843e-8


def positive(value):
    assert not isinstance(value, bool)
    value = float(value)
    assert math.isfinite(value) and value > 0
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('authenticated_snapshot', type=Path)
    parser.add_argument('public_snapshot', type=Path)
    args = parser.parse_args()
    auth_path, public_path = args.authenticated_snapshot.resolve(), args.public_snapshot.resolve()
    assert auth_path != public_path
    gate, config = verify_gate()
    receipt_path = HERE / 'submission_receipt_v1.json'
    receipt = read(receipt_path); check_tree(receipt)
    assert receipt['submission_id'] == SID and receipt['baseline_submission'] == 332517
    assert receipt['candidate'] == gate['candidate'] and receipt['candidate']['sha256'] == SHA
    assert receipt['policy'] == gate['policy'] and receipt['qualification'] == pin(HERE / 'FIRE_CHECKS_v1.json')
    archive = read(HERE / 'archive_receipt.json'); check_tree(archive)
    assert archive['passed'] is True and archive['validation'] == 'passed'
    assert archive['candidate'] == receipt['candidate'] and archive['archive'] == receipt['archive']
    assert archive['qualification'] == receipt['qualification'] and archive['policy'] == receipt['policy']
    assert receipt['archive']['sha256'] == ARCHIVE_SHA
    with tarfile.open(receipt['archive']['path'], 'r:gz') as package:
        entries = package.getmembers()
        assert sorted(e.name for e in entries) == ['estimator.py', 'manifest.json']
        assert all(e.isfile() and e.size <= 2_000_000 for e in entries)
        source = package.extractfile('estimator.py').read()
        assert hashlib.sha256(source).hexdigest() == SHA and len(source) == receipt['candidate']['bytes']
        manifest = json.loads(package.extractfile('manifest.json').read())
        assert manifest == archive['manifest']
    prior = read(H2 / 'official_bank_S31_v1.json')
    assert prior['submission_id'] == 332517
    assert prior['authenticated_poll']['score'] == BASE_SCORE
    assert prior['authenticated_poll']['score_secondary'] == BASE_RAW

    auth = read(auth_path); check_tree(auth)
    assert auth['submission_id'] == SID and auth['url'] == receipt['url']
    assert auth['receipt'] == pin(receipt_path)
    assert auth['script'] == pin(HERE / 'poll_submission_v1.py')
    assert auth['response']['returncode'] == 0 and auth['response']['timed_out'] is False
    assert auth['payload'] == json.loads(auth['response']['stdout'])
    api = auth['payload'].get('data', auth['payload'])
    assert api['id'] == SID and api['challenge_id'] == 1174 and api['challenge_round_id'] == 1429
    assert api['grading_status_cd'] == 'graded'
    score, raw = positive(api['score']), positive(api['score_secondary'])
    assert score < BASE_SCORE, 'Official result is not an improvement; no bank files written'

    public_bytes = public_path.read_bytes(); public = json.loads(public_bytes)
    assert public['submission_id'] == SID and public['url'] == receipt['url']
    evaluation = public['evaluation']
    assert int(evaluation['submission_id']) == SID and evaluation['state'] == 'completed'
    assert evaluation['completed_at'] and evaluation['error'] is None and evaluation['truncation'] is None
    progress = evaluation['progress']
    assert progress['total_mlps'] == progress['mlps_completed'] == 100
    assert progress['mlps_failed'] == progress['mlps_in_flight'] == 0
    results = evaluation['results']; rows = results['per_mlp']
    assert len(rows) == 100 and sorted(r['mlp_index'] for r in rows) == list(range(100))
    failure_flags = ('budget_exhausted', 'time_exhausted', 'residual_wall_time_exhausted', 'combined_budget_exhausted')
    for row in rows:
        assert row['state'] == 'completed' and row['error_code'] is None
        assert all(row['telemetry'][key] is False for key in failure_flags)
    public_scores = results['aggregates']['public']
    assert public_scores['n_failed_mlps'] == 0
    assert all(value == 0 for value in public_scores['failure_breakdown'].values())
    times = [positive(r['telemetry']['participant_predict_wall_s']) for r in rows]
    residuals = [positive(r['telemetry']['residual_wall_time_s']) for r in rows]
    setups = results['setup_executions']
    assert setups and all(row['error_code'] == 'SUCCESS' for row in setups)
    setup_times = [positive(row['wall_s']) for row in setups]
    bank_path = H2 / 'official_bank_S32_v1.json'; grade_path = HERE / 'grading_receipt_v1.json'
    full_path = H2 / f'eval_{SID}.json'
    assert all(not path.exists() for path in (bank_path, grade_path, full_path))
    record = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), submission_id=SID,
        url=receipt['url'], source_sha256=SHA, candidate=receipt['candidate'], archive=receipt['archive'],
        authenticated_poll={key: api.get(key) for key in ('grading_status_cd', 'score', 'score_secondary', 'grading_message')},
        previous_bank=dict(submission_id=332517, score=BASE_SCORE, score_secondary=BASE_RAW,
            receipt=pin(H2 / 'official_bank_S31_v1.json')),
        adjusted_improvement_pct=100 * (1 - score / BASE_SCORE), raw_change_pct=100 * (raw / BASE_RAW - 1),
        telemetry=dict(completed=100, failed=0, max_predict_wall_s=max(times), mean_predict_wall_s=sum(times)/100,
            max_residual_wall_s=max(residuals), mean_residual_wall_s=sum(residuals)/100,
            max_setup_wall_s=max(setup_times), mean_setup_wall_s=sum(setup_times)/len(setup_times),
            setup_executions=len(setups), mean_compute_utilization=public_scores['mean_compute_utilization']),
        completed_at_utc=evaluation['completed_at'], authenticated_snapshot=pin(auth_path), public_snapshot=pin(public_path),
        submission_receipt=pin(receipt_path), archive_receipt=pin(HERE / 'archive_receipt.json'),
        qualification=receipt['qualification'], policy=receipt['policy'], local_validation=gate['validation'],
        local_validation_reference=pin(config['validation_comparison']), minimum_validated_gain_pct=1,
        policy_scope='At least 1% fully validated local gain; final official adjusted score must improve the bank.',
        recorder=pin(__file__), status='Official current Phase 2 bank S32; graded 100/100 with zero failures')
    with full_path.open('xb') as stream:
        stream.write(public_bytes)
    record['evaluation_receipt'] = pin(full_path)
    once(bank_path, record); once(grade_path, record)
    print(json.dumps(dict(recorded=True, submission_id=SID, score=score, raw=raw,
        adjusted_improvement_pct=record['adjusted_improvement_pct'], raw_change_pct=record['raw_change_pct'],
        telemetry=record['telemetry'], bank=pin(bank_path), grading_receipt=pin(grade_path))), flush=True)


if __name__ == '__main__':
    main()
