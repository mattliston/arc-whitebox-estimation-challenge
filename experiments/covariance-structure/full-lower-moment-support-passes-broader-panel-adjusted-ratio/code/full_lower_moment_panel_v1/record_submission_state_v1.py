"""Version the accepted upload and pending grade without changing experiment history."""
import datetime
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent
ROOT = Path('/Users/matt/arc-white-box-estimation')
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def write(path, value):
    with path.open('x') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')


def verify(record):
    actual = pin(Path(record['path']))
    assert actual['sha256'] == record['sha256'], record['path']
    assert actual['bytes'] == record['bytes'], record['path']


prior_verified = read(D / 'continuation_checkpoint_v60_verified.json')
for name in ['checkpoint', 'handoff', 'index', 'closeout']:
    verify(prior_verified[name])
prior_closeout = read(HERE / 'closeout.json')
mismatches = []
for record in prior_closeout['artifacts'] + prior_closeout['external_pins']:
    actual = pin(Path(record['path']))
    if actual != record:
        mismatches.append(dict(recorded=record, actual=actual))
assert len(mismatches) == 1
assert mismatches[0]['recorded']['path'] == str(HERE / 'finalize.log')
assert mismatches[0]['recorded']['bytes'] == 0
write(HERE / 'prior_closeout_log_pin_note.json', dict(
    recorded_at_utc=NOW,
    prior_closeout=pin(HERE / 'closeout.json'),
    pins_checked=len(prior_closeout['artifacts']) + len(prior_closeout['external_pins']),
    mismatch=mismatches[0],
    cause='The original finalizer hashed its redirected stdout log while empty, before printing to it.',
    effect='Only the finalizer log pin is stale; all numerical, source, archive and other pins verify.',
    resolution='Original closeout and log are preserved. New closeout hashes the completed stable log.',
))

package = read(HERE / 'package.json')
verify(package['archive'])
verify(package['candidate'])
assert package['archive']['bytes'] == 779737
assert package['archive']['sha256'] == '01e05883c1767534e23b2f4ee0d0e4b887f777c172080f14b3a91d70ea3402fd'
with tarfile.open(package['archive']['path'], 'r:gz') as archive:
    assert archive.getnames() == ['estimator.py', 'manifest.json']
    assert hashlib.sha256(archive.extractfile('estimator.py').read()).hexdigest() == package['candidate']['sha256']
receipt = read(HERE / 'submission.json')
attempt = read(HERE / 'submission_attempt.json')
assert receipt['ok'] and receipt['submission_id'] == 331672 and attempt['exit_code'] == 0
verify(attempt['receipt'])
poll_path = sorted(HERE.glob('sub331672_poll_*.json'))[-1]
poll = read(poll_path)
assert poll['grading_status_cd'] == 'submitted' and poll['score'] is None
preauthorization = D / 'board_submission_preauthorization_20260920.json'
assert read(preauthorization)['explicit_preauthorization_received']
assert read(preauthorization)['user_statement'] == 'you are approved for all uploads'
panel = read(HERE / 'panel_score.json')
confirmation = read(HERE / 'confirmation_score.json')
runtime = read(HERE / 'matched_runtime_comparison.json')
resources = read(HERE / 'resources.json')
assert panel['panel_advancement_passed'] and confirmation['confirmation_advancement_passed']
assert runtime['calibrated_runtime_gate_passed'] and package['validated']
assert not runtime['original_absolute_runtime_gate_passed']
assert resources['all_owned_groups_retired'] and resources['max_swap_growth_bytes'] == 0
bank = read(D / 'official_best_331599.json')
url = 'https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/331672'
state = dict(
    observed_at_utc=NOW, submission_id=331672, submission_url=url,
    accepted_at_utc=receipt['submission']['data']['created_at'],
    grading_status=poll['grading_status_cd'], official_score=None,
    receipt=pin(HERE / 'submission.json'), latest_poll=pin(poll_path),
    archive=package['archive'], candidate=package['candidate'],
    approval_pending=False, upload_worker_active=False,
    exact_archive_user_approval=pin(HERE / 'upload_approval_received.json'),
    standing_upload_authorization=pin(preauthorization),
    official_bank=dict(id=bank['submission_id'], adjusted=bank['adjusted'], raw=bank['raw']),
    next_action='Poll submission 331672, sequence 3 or next unused sequence; do not resubmit.',
)
write(HERE / 'accepted_submission_state_v1.json', state)
write(HERE / 'RESULT_v2.md', f'''# Full lower-moment panel: uploaded; official grading pending

Updated {NOW}. This record supersedes the approval-pending status in RESULT.md and closeout.json; those historical records remain unchanged.

The exact approved archive was accepted as [AIcrowd submission #331672]({url}) at {state['accepted_at_utc']}. The latest authenticated check ({poll['observed_at_utc']}) reports submitted/enqueued, with no official score. The official bank remains Lazy #331599, adjusted {bank['adjusted']:.16g}, raw {bank['raw']:.16g}. The 4e-9 milestone is achieved; 3e-9, 2e-9 and approximately 1e-9 remain unmet. The broad goal stays active.

The user explicitly approved this archive, then authorized all uploads. board_submission_preauthorization_20260920.json preserves that standing authorization. No approval or upload worker is pending. Continue monitoring the existing submission, without a duplicate upload.

The archive is 779,737 bytes, SHA256 {package['archive']['sha256']}. It contains only estimator.py and manifest.json. The embedded estimator matches the unchanged preserved full_lower_moment_support_v1 source, SHA256 {package['candidate']['sha256']}.

The 14-network local panel gives noise-corrected adjusted {panel['groups']['all14']['candidate']['adjusted_nc']:.16g}, ratio {panel['groups']['all14']['adjusted_nc_ratio']:.12g} to Lazy, with 9/14 wins. The eight additional confirmation networks all improve; ordinary adjusted {confirmation['aggregate']['candidate']['adjusted']:.16g}, ratio {confirmation['pooled_adjusted_ratio']:.12g}. These are reused research networks and local scores, not pristine holdouts or an official grade. No new fit or samples were used.

The original strict development gate and original absolute/historical runtime gates failed and remain recorded as failures. The broader panel and confirmation protocols passed. A separately declared matched-runtime comparison found the unchanged Lazy parent 16.94% slower than its old saved run in the current environment; candidate/current-parent wall ratio is {runtime['pooled_wall_ratio']:.12g}. Its calibrated runtime gate passed. This justified one platform trial, without guaranteeing platform completion. The NumPy-free client/server outputs and FLOP bills exactly match the native records; contract and package checks pass.

All {resources['numerical_guards']} guard groups retired, with peak observed RSS {resources['peak_rss_bytes']} bytes and zero added swap. There were 20 new native and 6 client full calls, plus two reused development cold calls. No new samples, fits or GPU jobs. One scorer assertion failure was preserved and repaired before reference decoding; the candidate and scoring formulas were unchanged.

Bookkeeping correction: original closeout.json pinned finalize.log while that redirected log was empty; the finalizer then wrote to it. All other 205 original pins verify. The original closeout/log are preserved, and prior_closeout_log_pin_note.json records the mismatch. closeout_after_submission_v1.json hashes the completed stable log and all current study artifacts. This does not require numerical reruns.
''')

closeout_path = HERE / 'closeout_after_submission_v1.json'
artifacts = [pin(p) for p in sorted(HERE.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p != closeout_path]
external = prior_closeout['external_pins'] + [pin(preauthorization), pin(D / 'official_best_331599.json')]
closeout = dict(
    created_utc=NOW, numerical_study_complete=True, official_grading_pending=True,
    status='Exact archive accepted as 331672; grade pending, no approval pending',
    artifacts=artifacts, external_pins=external, resources=resources,
    submission=state, panel_advancement_passed=True, confirmation_advancement_passed=True,
    old_development_gate_passed=False, original_runtime_gate_passed=False,
    calibrated_runtime_gate_passed=True, package_validated=True,
    prior_closeout_log_pin_exception=pin(HERE / 'prior_closeout_log_pin_note.json'),
    official_bank=state['official_bank'], goal_achieved=False,
)
write(closeout_path, closeout)
for record in artifacts + external:
    verify(record)

handoff = ROOT / 'HANDOFF_CURRENT.md'
index = D / 'REEXAMINATION_INDEX_20260919.md'
backups = []
for original, name in [(handoff, 'handoff_before_checkpoint61.md'), (index, 'index_before_checkpoint61.md')]:
    destination = D / name
    with destination.open('xb') as stream:
        stream.write(original.read_bytes())
    backups.append(pin(destination))
entry = f'''## 2026-09-20 UTC — full lower-moment archive accepted as #331672; grading pending

- Exact approved archive uploaded successfully at {state['accepted_at_utc']}. Latest authenticated poll {poll['observed_at_utc']}: submitted/enqueued, no official score. Monitor #331672 using full_lower_moment_panel_v1/poll_submission.py with the next unused sequence (currently 3); do not duplicate upload. CLI session77680 exited0 and is terminal. No approval, upload worker or numerical worker pending.
- User explicitly said “Approve this exact upload”, then “you are approved for all uploads”. Standing qualified Phase2 upload authorization is recorded in board_submission_preauthorization_20260920.json; do not request it again. Historical auto-review rejection is resolved.
- Official bank remains Lazy #331599 adjusted3.8164492657795035e-9/raw2.169264529783277e-8. 4e-9 achieved;3/2/~1e-9 unmet; goal active. New local score is not an official replacement until grading succeeds.
- Expanded unchanged full_lower_moment_support candidate: NC14 adjusted3.6889657488632113e-9, ratio.9666493395546764 (9/14 wins); fixed eight confirmation ordinary adjusted3.6899806195313174e-9, ratio.9562402203721203 (8/8 wins). Reused research networks; no new fit/samples. Original stricter development gate remains failed. Broader panel and confirmation gates passed.
- Exact NumPy-free boundary predictions/bills and contract/package pass. Original runtime gates remain failed; contemporaneous accepted Lazy is16.94% slower than its saved run, candidate/current-parent wall ratio1.0083372236336914. Separately declared calibrated runtime gate passed; platform completion remains unproven while queued.
- Archive full_lower_moment_panel_v1/submission-full-lower-moments-20260920.tar.gz:779737B, SHA256 {package['archive']['sha256']}; source {package['candidate']['sha256']}. 29 numerical guards retired, peak{resources['peak_rss_bytes']}B, zero added swap;20new native+6client full calls, no new samples/fits/GPU. About{shutil.disk_usage(ROOT).free / 2**30:.1f}GiB disk free.
- Checkpoint61 supersedes60. RESULT_v2.md and closeout_after_submission_v1.json record acceptance, all current pins and the preserved original finalize.log pin timing error (only one of206 original pins stale; all actual experiment/package pins intact). No next numerical study selected; first resolve the existing official grade.

'''
handoff.write_text(entry + handoff.read_text())
index.write_text(entry + index.read_text())
checkpoint = dict(
    checkpoint_version=61, updated_at_utc=NOW, supersedes=60,
    prior_checkpoint=pin(D / 'continuation_checkpoint_v60.json'),
    historical_experiments='All earlier experiment records remain in the prior checkpoint and versioned source directories.',
    goal_status='active', goal_achieved=False,
    official_submission=331672, submission_status=poll['grading_status_cd'],
    pending_submission=state, official_bank=state['official_bank'],
    official_bank_record=pin(D / 'official_best_331599.json'),
    approval_pending=False, upload_worker_active=False, numerical_workers_active=False,
    active_numerical_session=None, board_submission_preauthorized=True,
    board_preauthorization_record=pin(preauthorization),
    standing_board_submission_authorization='User explicitly approved all uploads; continue qualified Phase2 submissions without renewed permission.',
    full_lower_moment_panel=dict(closeout=pin(closeout_path), result=pin(HERE / 'RESULT_v2.md'),
        local_nc14_adjusted=panel['groups']['all14']['candidate']['adjusted_nc'],
        confirmation_adjusted=confirmation['aggregate']['candidate']['adjusted'], confirmation_wins=8),
    milestones=read(D / 'continuation_checkpoint_v60.json')['milestones'],
    previous_goal_turn_classification='Completed numerical progress',
    current_goal_turn_classification='Expanded validation and accepted authorized upload; verified external grading wait',
    next_numerical_study_selected=False, next_action=state['next_action'],
    this_turn_resources=resources, preserved_previous_handoff_and_index=backups,
    handoff=pin(handoff), index=pin(index),
)
checkpoint_path = D / 'continuation_checkpoint_v61.json'
write(checkpoint_path, checkpoint)
verified = dict(
    recorded_at_utc=NOW, checkpoint=pin(checkpoint_path), handoff=pin(handoff), index=pin(index),
    closeout=pin(closeout_path), submission_receipt=state['receipt'], poll=state['latest_poll'],
    archive=package['archive'], candidate=package['candidate'],
    current_study_artifact_pins_verified=len(artifacts), external_pins_verified=len(external),
    original_closeout_exception='Only finalize.log was pinned before its redirected output was written; explicitly recorded and re-pinned.',
    approval_pending=False, numerical_workers_active=False, upload_worker_active=False,
    official_grading_pending=True,
)
write(D / 'continuation_checkpoint_v61_verified.json', verified)
for key in ['checkpoint', 'handoff', 'index', 'closeout', 'submission_receipt', 'poll', 'archive', 'candidate']:
    verify(verified[key])
print(json.dumps(dict(checkpoint=61, submission=331672, status=poll['grading_status_cd'],
    approval_pending=False, artifact_pins_verified=len(artifacts), external_pins_verified=len(external),
    closeout=pin(closeout_path), goal_status='active'), indent=2))
