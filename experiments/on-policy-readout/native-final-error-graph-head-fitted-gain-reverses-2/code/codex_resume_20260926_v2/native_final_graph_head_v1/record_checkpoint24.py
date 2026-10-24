"""Close the completed fixed pilot using receipts and byte hashes only."""
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import shutil
import subprocess

G = Path(__file__).resolve().parent
H = G.parent
ROOT = next(p for p in G.parents if (p / 'HANDOFF_CURRENT.md').is_file())


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, value):
    with path.open('x') as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, indent=2)
            stream.write('\n')


checkpoint_path = H / 'continuation_checkpoint_v24.json'
backups = {
    H / 'RESULTS.md': H / 'RESULTS_before_checkpoint24.md',
    H / 'NEXT_RESEARCH_v1.md': H / 'NEXT_RESEARCH_before_checkpoint24.md',
    ROOT / 'HANDOFF_CURRENT.md': ROOT / 'HANDOFF_CURRENT.md.before_codex_resume_v2_checkpoint24',
}
new_files = [checkpoint_path, *backups.values(), *[
    G / name for name in ['RESULT.md', 'progress.json', 'process_census.json',
                         'artifact_manifest.json', 'closeout.json']
]]
assert not any(p.exists() for p in new_files), 'Closeout is write-once'
assert not (G / 'native_build.json').exists()
assert not (H / 'candidate_native_final_graph_head_v1.py').exists()

assessment = read(G / 'assessment.json')
survivor = read(G / 'survivor.json')
capture = read(G / 'capture_complete.json')
pipeline = read(G / 'pipeline_complete.json')
qualification = read(G / 'qualification.json')
audit = read(G / 'additional_prediction_byte_audit_v1.json')
fits = {arm: read(G / f'fit_{arm}/result.json') for arm in ['graph', 'control']}
precheck = read(G / 'precheck_seal.json')
guards = {p.name: read(p) for p in sorted(G.glob('*_guard.json'))}

assert len(guards) == 21
assert all(v['completed'] and v['exit_code'] == 0 and v['whole_group_retired']
           and not v['remaining_rows'] and not v['max_swap_growth_bytes']
           for v in guards.values())
assert pipeline['all_groups_retired'] and capture['all_groups_retired']
assert not assessment['passes_information_gate'] and not any(assessment['gates'].values())
assert survivor['survivor'] is None and not survivor['eligible']
assert not any(flag for arm in survivor['paid_screen_gates'].values() for flag in arm.values())
assert not assessment['valid40_accessed'] and not assessment['submission_qualified']
assert not survivor['new_label_reads'] and not survivor['valid40_accessed']
assert qualification['passed'] and not qualification['targets_accessed']
assert len(capture['rows']) == 160 and len({v['rank'] for v in capture['rows']}) == 160
assert not capture['targets_accessed']
assert audit['checked'] == 20 and all(v['all16_file_bytes_exact'] for v in audit['records'])
assert len(fits['graph']['fit_ranks']) == 128 and len(fits['graph']['check_ranks']) == 32
assert set(fits['graph']['fit_ranks']).isdisjoint(fits['graph']['check_ranks'])
for arm, fit in fits.items():
    assert fit == precheck['models'][arm]
    assert fit['fit_ranks'] == fits['graph']['fit_ranks']
    assert fit['check_ranks'] == assessment['check_ranks']
    assert fit['terminal_epoch'] == 40 and fit['steps'] == 1280
    assert not fit['check_labels_accessed'] and not fit['valid_labels_accessed']
    assert sha(G / f'fit_{arm}/weights.npz') == fit['weights_sha256']
    assert sha(G / f'fit_{arm}/check_predictions.npy') == fit['predictions_sha256']
    assert sha(G / 'capture_complete.json') == fit['capture_sha256']
assert sha(G / 'assessment.json') == survivor['assessment_sha256'] == pipeline['assessment_sha256']
assert sha(G / 'precheck_seal.json') == assessment['precheck_seal_sha256']
assert sha(G / 'assess.py') == assessment['assessment_sha256']
assert sha(G / 'banking_retention_addendum_v2.json') == survivor['addendum_sha256']
for rel, expected in read(G / 'pre_fit_source_seal.json')['files'].items():
    assert sha(G / rel) == expected, rel
for row in capture['rows']:
    assert sha(G / row['path']) == row['sha256'], row['rank']
    assert not row['targets_accessed']
assert sum(v['bytes'] for v in capture['rows']) == capture['bytes'] == 121402240

preserved = {
    H / 'candidate_old_both_tail_trace_bounded_v6.py':
        '5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26',
    ROOT / 'submissions/submission-20260926-213059.tar.gz':
        '8d3ae68e1550ff801ff8688c64e93787fdab6d0fd1064fbf72e71c02b58db7ff',
    H / 'candidate_thin_star_route_v1.py':
        '25c22c219420c13db8eae758c16b96a322f24b4de5d40efb43890605b4d3b3ee',
}
for path, expected in preserved.items():
    assert sha(path) == expected, path

now = datetime.now(timezone.utc)
asof = now.isoformat()
local = now.astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M %Z')
processes = subprocess.run(['ps', '-axo', 'pid=,pgid=,command='],
                           check=True, text=True, capture_output=True).stdout
pattern = re.compile(r'native_final_graph_head_v1/(capture|queue_capture|fit|finish|assess)\.py')
matches = [line.strip() for line in processes.splitlines() if pattern.search(line)]
assert not matches, matches
write_once(G / 'process_census.json', {
    'asof_utc': asof, 'scope': pattern.pattern, 'matching_workers': matches,
    'command': ['ps', '-axo', 'pid=,pgid=,command='],
    'all_21_guarded_groups_retired': True,
})

peak = max(v['max_rss_bytes'] for v in guards.values())
free = shutil.disk_usage(ROOT).free
fit_pct = {arm: 100 * (v['frozen_fit_mse'] / v['baseline_fit_mse'] - 1)
           for arm, v in fits.items()}
body = f'''# Native final-error graph head: fixed pilot closed

Recorded {local}. **No new banked improvement.** Official S31 submission
332517 remains **2.1736465056908115e-9 adjusted**, with 100/100 evaluations
complete and zero failures in its preserved official receipt. Its source,
submission archive and retained thin-star source hashes are unchanged.

The graph head improves frozen FIT128 raw MSE by 4.84726%, but worsens
excluded CHECK32 raw MSE by **2.97073%** (paired network-bootstrap 95%
interval **[+2.31249%, +3.67570%]**, **2/32 wins**). The local-feature
control improves FIT by 0.50001% and is **0.07575% worse** on CHECK32
(95% interval [-0.16869%, +0.33286%], 16/32 wins). Graph versus control
is +2.89280% raw MSE. The FIT gain does not generalize in this fixed test.

All original graph information gates and all prospective smaller-gain
paid-screen gates fail. `survivor.json` selects neither head. Both heads
are closed with no CHECK-guided sign, amplitude, epoch, feature or
architecture rescue. No changed-head native candidate, metered inference
qualification, VALID40 evaluation, package or submission follows.

## Scope and controls

The fixed split contains 128 FIT and 32 excluded CHECK networks, all from
the existing TRAIN160 research set. These are not pristine independent
holdouts: the underlying estimator and earlier research have used TRAIN.
VALID40 was unopened for this pilot. The split and all model choices were
fixed before this pilot read labels.

Passive native S31 capture supplies final scalar/state features plus 57
transported fields from layers 12, 13 and 14. Those fields use signed and
absolute correlation-neighbor averages, soft neighbors and a covariance
power-subspace projection. Both heads have a fixed 151-512-256-1 SiLU
architecture, zero initial output, identical data order/seed, and 40 epochs
(1,280 updates). The control zeros the 57 graph columns. Normalization uses
FIT only. Terminal predictions are rounded to float32 before scoring.
Both fitted weights and CHECK predictions were sealed before assessment
read CHECK labels; no approximate native trajectory derivative is used.

This is an offline final-error readout on native captures, not a deployed
or cost-qualified changed native head. It closes this feature/architecture/
training instance, not every possible graph estimator.

Tiny target-free permutation, neighbor and transport checks pass. All 160
captures finish without reading targets. Twenty independent historical
all-16-layer prediction files match byte for byte. The original matched
35/100 smoke also matches complete bills; the other historical comparisons
do not certify bills because process orders differ. Capture bytes and both
fitted artifacts match their existing seals at closeout.

## Prospective retention and provenance

`control_retention_addendum_v1.json` and the superseding retention-only
`banking_retention_addendum_v2.json` were written before either fit began
or any new FIT/CHECK score existed. The second permits an already fixed
head with at least 3% raw gain, a 95% upper ratio below 0.99 and at least
24/32 wins to enter paid native screening. It preserves the original 5%
graph information gate and requires at least 2% actual adjusted gain plus
further qualification before release. Neither arm passes either rule.
No model, prediction, primary assessment or threshold was changed after
the CHECK result. The pre-fit source seal remains exact.

`assessment.json`, `survivor.json`, `precheck_seal.json`, and both
`fit_*/result.json` files are the scientific receipts. `native_helper_draft.py`
and `build_native.py` are unexecuted drafts; they establish no inference
parity, bill or performance result. Historical launch/in-progress/session
records are superseded by `closeout.json` and checkpoint 24.

## Resources and continuation

All 21 guarded groups completed successfully and retired. Peak sampled
RSS was {peak:,} bytes ({peak / 1e9:.3f} GB); added swap was zero.
The serial capture contains 121,402,240 bytes, with no full covariance
dump. About {free / 2**30:.1f} GiB is free at closeout. The process census
finds no remaining capture, fit, assessment or supervisor workers.

Board submissions remain authorized. No permission, upload or grading is
pending. Thin-star's earlier 0.57346% TRAIN8 adjusted gain remains below
the submission bar and lacks held-out/actual-worker qualification. No new
candidate is selected. The 2e-9 and approximately 1e-9 goals remain unmet;
the goal is active. Restart from `../continuation_checkpoint_v24.json`.
'''
write_once(G / 'RESULT.md', body)
write_once(G / 'progress.json', {
    'asof_utc': asof, 'stage': 'closed_negative', 'survivor': None,
    'all_workers_retired': True, 'valid40_accessed': False,
    'candidate_built': False, 'board_submitted': False,
    'supersedes': ['launch_state.json', 'running_sessions_v1.json',
                   'in_progress_v1.json', 'in_progress_v2.json'],
    'checkpoint': str(checkpoint_path.relative_to(ROOT)),
})
manifest = {
    str(p.relative_to(G)): {'sha256': sha(p), 'bytes': p.stat().st_size}
    for p in sorted(G.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
}
write_once(G / 'artifact_manifest.json', {
    'asof_utc': asof, 'scope': 'Pilot files before manifest and closeout receipts',
    'files': manifest, 'total_bytes': sum(v['bytes'] for v in manifest.values()),
})
previous = read(H / 'continuation_checkpoint_v23.json')
checkpoint = {
    'asof_utc': asof, 'goal': 'active; 2e-9 and approximately 1e-9 unmet',
    'classification': 'Completed native-capture final-error graph pilot; no new banked gain',
    'official_bank': previous['official_bank'],
    'official_bank_receipt_reused_not_repolled': True,
    'user_steering': 'Board submissions remain authorized for meaningful couple-percent adjusted gains',
    'experiment': str(G.relative_to(ROOT)),
    'protocol_sha256': sha(G / 'protocol.json'),
    'fit_raw_pct': fit_pct, 'assessment': assessment, 'survivor': survivor,
    'fixed_model_fit_counts': {'fit': 128, 'check': 32, 'epochs': 40, 'updates': 1280},
    'new_labels_read': 'FIT128 for fitting; CHECK32 only after both fixed terminal predictions sealed',
    'split_scope': 'Reused TRAIN160 research networks; not pristine held-out evidence',
    'capture': {'count': 160, 'bytes': capture['bytes'], 'all_hashes_reverified': True,
                'targets_accessed': False, 'independent_prediction_byte_controls': 20,
                'independent_full_bill_controls': [35, 100]},
    'pre_fit_source_seal_verified': True, 'fitted_weight_and_prediction_seals_verified': True,
    'changed_head_native_candidate_built': False, 'native_cost_qualified': False,
    'valid40_accessed': False, 'board_submitted': False, 'pending_approval': False,
    'pending_grading': False, 'numerical_workers_running': False,
    'resources': {'guard_count': len(guards), 'all_completed': True,
                  'all_retired': True, 'max_guard_rss_bytes': peak,
                  'max_added_swap_bytes': 0, 'free_disk_bytes': free,
                  'guard_elapsed_sum_s': sum(v['elapsed_s'] for v in guards.values()),
                  'pipeline_elapsed_s': pipeline['elapsed_s'],
                  'artifact_manifest_bytes': sum(v['bytes'] for v in manifest.values())},
    'preserved_files': {str(p.relative_to(ROOT)): s for p, s in preserved.items()},
    'artifact_manifest_sha256': sha(G / 'artifact_manifest.json'),
    'decision': 'Close both fixed heads; no CHECK-guided sign, amplitude, epoch, architecture or feature rescue',
    'next_candidate': None,
    'retained_component': 'thin_star_route_v1: TRAIN8 adjusted -0.57346%; below submission bar and not held-out/actual-worker qualified',
}
write_once(checkpoint_path, checkpoint)

for original, backup in backups.items():
    write_once(backup, original.read_text())
entry = f'''

## Checkpoint24 — {local}: native final-error graph pilot closed

S31 #332517 remains **2.1736465056908115e-9 adjusted**, 100/100 complete,
zero failures; bank source/archive and thin-star source are unchanged.
There is no new banked gain. Fixed native-capture FIT128/CHECK32 graph
head: FIT raw -4.84726%, CHECK raw **+2.97073%**, 95% interval
[+2.31249%, +3.67570%], **2/32 wins**. Local-feature control: FIT
-0.50001%, CHECK +0.07575%, interval [-0.16869%, +0.33286%], 16/32 wins.
Graph versus control is +2.89280%. Both original information and prospective
smaller-gain paid-screen gates fail; neither head survives.

All 160 passive native captures complete; 20 historical all16 prediction
files are byte-exact, and the matched 35/100 smoke also preserves full bills.
Both 40-epoch heads and CHECK predictions were sealed before CHECK labels
were read. These are reused TRAIN networks, not pristine validation.
No changed-head native candidate, metered cost claim, VALID40, rescue,
package or upload follows. This closes the fixed nonlinear neighbor/subspace
final-error readout instance; earlier one-step head results were a different
consumer. Do not present this fixed native-error pilot as untried again.

All 21 guarded groups completed and retired; peak RSS {peak / 1e9:.3f} GB,
zero added swap, captures 121.402 MB, about {free / 2**30:.1f} GiB free.
No approval, upload, grading or numerical job is pending. Thin-star remains
below the submission bar; no new candidate is selected. The goal is active
and 2e-9/~1e-9 remain unmet. Details: `native_final_graph_head_v1/RESULT.md`,
`assessment.json`, `survivor.json`, and `continuation_checkpoint_v24.json`.
'''
for name in ['RESULTS.md', 'NEXT_RESEARCH_v1.md']:
    with (H / name).open('a') as stream:
        stream.write(entry)
banner = f'''**{local}: CHECKPOINT24; native final-error graph pilot CLOSED, no new banked gain.**
S31 #332517 remains 2.1736465056908115e-9 adjusted, 100/100 complete, zero
failures. Source 5d439308 and archive 8d3ae68e are unchanged. Submission
authorization persists; no approval, upload or grading is pending.

Fixed FIT128/CHECK32 graph head: FIT raw -4.84726%, CHECK raw +2.97073%,
2/32 wins. Local control CHECK +0.07575%, 16/32 wins. Both gates fail;
no changed-head native candidate, VALID40, rescue or submission. All 21
guarded groups and supervisors retired, peak {peak / 1e9:.3f} GB, zero added
swap. Historical RUNNING/session records below are superseded; do not
restart them. Twenty historical all16 predictions are byte-exact.

Restart: codex_resume_20260926_v2/continuation_checkpoint_v24.json and
native_final_graph_head_v1/RESULT.md, assessment.json, survivor.json,
closeout.json. Thin-star TRAIN8 -0.57346% remains below upload bar and
unqualified on held-out/actual-worker gates. No new candidate selected.
Goal active; 2e-9 and approximately 1e-9 unmet.

'''
handoff = ROOT / 'HANDOFF_CURRENT.md'
handoff.write_text(banner + backups[handoff].read_text())
write_once(G / 'closeout.json', {
    'asof_utc': asof, 'status': 'closed_negative', 'survivor': None,
    'all_workers_retired': True, 'pending_approval': False,
    'checkpoint': str(checkpoint_path.relative_to(ROOT)),
    'checkpoint_sha256': sha(checkpoint_path),
    'artifact_manifest_sha256': sha(G / 'artifact_manifest.json'),
    'handoff_sha256': sha(handoff),
    'documentation_sha256': {name: sha(H / name) for name in ['RESULTS.md', 'NEXT_RESEARCH_v1.md']},
    'prior_documentation_backups': {str(p.relative_to(ROOT)): sha(p) for p in backups.values()},
    'historical_running_records_superseded': True,
})
for rel, record in manifest.items():
    assert sha(G / rel) == record['sha256'], rel
for path, expected in preserved.items():
    assert sha(path) == expected, path
assert handoff.read_text().endswith(backups[handoff].read_text())
print(json.dumps({'checkpoint': str(checkpoint_path), 'status': 'closed_negative',
                  'guard_count': len(guards), 'max_rss_bytes': peak,
                  'swap_growth_bytes': 0, 'manifest_files_verified': len(manifest),
                  'fit_raw_pct': fit_pct, 'survivor': None,
                  'bank_adjusted': previous['official_bank']['authenticated_poll']['score']}))
