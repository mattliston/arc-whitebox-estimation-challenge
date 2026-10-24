"""Preserve the completed two-network cost ceiling and continuation state."""
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil

D = Path(__file__).resolve().parent
H = D.parent
ROOT = next(p for p in D.parents if (p / 'HANDOFF_CURRENT.md').is_file())
read = lambda p: json.loads(p.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def once(path, value):
    with path.open('x') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')


checkpoint = H / 'continuation_checkpoint_v25.json'
backups = {
    ROOT / 'HANDOFF_CURRENT.md': ROOT / 'HANDOFF_CURRENT.md.before_codex_resume_v2_checkpoint25',
    H / 'RESULTS.md': H / 'RESULTS_before_checkpoint25.md',
    H / 'NEXT_RESEARCH_v1.md': H / 'NEXT_RESEARCH_before_checkpoint25.md',
}
assert all(not p.exists() for p in [checkpoint, D / 'RESULT.md', D / 'closeout.json', *backups.values()])
result = read(D / 'result.json')
guard = read(D / 'census_guard.json')
audit = read(D / 'independent_audit.json')
prior = read(H / 'continuation_checkpoint_v24.json')
assert not result['passes_ceiling_gate']
assert guard['completed'] and guard['whole_group_retired'] and not guard['remaining_rows']
assert guard['exit_code'] == 0 and guard['max_swap_growth_bytes'] == 0
assert audit['source_seal_verified']
assert all(r['full_prediction_file_bytes_exact'] and r['exhaustive_pairwise_replay_exact'] for r in audit['records'])
for rel, expected in prior['preserved_files'].items():
    assert sha(ROOT / rel) == expected, rel
for rel, expected in read(D / 'launch_seal.json')['files'].items():
    assert sha(D / rel) == expected, rel
now = datetime.now(timezone.utc)
asof = now.isoformat()
local = now.astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M %Z')
free = shutil.disk_usage(ROOT).free
peak = max(guard['max_rss_bytes'], result['peak_rss_bytes'])
summary = f'''## Checkpoint25 — {local}: exact transform reuse ceiling is too small

Official S31 #332517 remains **2.1736465056908115e-9 adjusted**, 100/100
complete, zero failures. Its source/archive and thin-star source are
unchanged. This turn supplies new cost evidence, not a banked score gain.

Unchanged native S31 TRAIN35/100 captures 409/407 A/B input transforms.
There are 60/63 exact input-key repeats, of which 27/24 are already free
resident reuses. All repeated transform results match byte for byte.
Removing every remaining repeated transform at zero lookup, copy or
storage cost saves at most **0.345746% / 0.381218%** of the complete bills.
This is an optimistic unlimited-cache ceiling, not a deployed saving.
It fails the prospectively fixed 1.5% implementation gate and cannot bring
the retained ~0.57% thin-star component near the couple-percent release bar.
No numerical cache candidate is built.

All 16-layer prediction files and complete bills equal independent S31
controls exactly. Streaming-hash fixtures and an independent exhaustive
pairwise regrouping pass. This covers identical same-side/depth/dimension
input transforms at the two observed current global bindings; it does not
bound new algebra, A/B transpose conversion, partial transforms or other
inputs. The older whole-product census explicitly excluded this question;
the existing E-73 resident A reuse was not counted as an added saving.

One guarded process retired successfully after {guard['elapsed_s']:.2f}s,
peak RSS {peak / 1e9:.3f} GB, zero added swap; about {free / 2**30:.1f} GiB
free. No truth arrays, training, VALID40, new samples, changed estimator,
package or upload. No numerical job, approval or grading is pending.
Submission authorization persists. No new candidate is selected; the 2e-9
and approximately 1e-9 goals remain unmet and active. Scientific receipts:
`factor_transform_reuse_v1/result.json`, `independent_audit.json`,
`census_guard.json`; restart from `continuation_checkpoint_v25.json`.
'''
once(D / 'RESULT.md', '# Current S31 transform-reuse census: closed\n\n' + summary)
sources = [
    H.parent / 'codex_resume_20260919_v1/current_product_identity_census_v1/RESULT.md',
    H.parent / 'fable_engineer_v1/RESULT.md',
    H.parent / 'codex_resume_20260926_v1/s30_profile_summary_v1.json',
]
once(D / 'source_review.json', {
    'asof_utc': asof,
    'scope': 'Read-only prior evidence; no new endpoint or general impossibility claim',
    'sources': {str(p.relative_to(ROOT)): sha(p) for p in sources},
    'distinctions': [
        'Older 295-product identity census found zero full-product duplicates and explicitly excluded partial transforms.',
        'E-73 already reuses consecutive resident A transforms, is included in S31, and contributes zero to the new charged-repeat ceiling.',
        'The S30 profile locates large transport/read costs but is an older profile, not a new S31 attribution.',
        'Current S31 factor tensors persist between joins, but repeated transform charges now measured are too small for the proposed material bundle.',
    ],
})
files = {str(p.relative_to(D)): {'sha256': sha(p), 'bytes': p.stat().st_size}
         for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
once(D / 'artifact_manifest.json', {'asof_utc': asof, 'files': files,
     'total_bytes': sum(v['bytes'] for v in files.values()),
     'scope': 'All pilot files before the manifest and closeout receipts'})
record = {
    'asof_utc': asof, 'goal': 'active; 2e-9 and approximately 1e-9 unmet',
    'previous_turn_classification': 'progress: fixed graph pilot completed and rejected; no banked gain',
    'classification': 'progress: current-S31 exact transform reuse ceiling measured and rejected',
    'official_bank': prior['official_bank'], 'official_bank_repolled': False,
    'preserved_files': prior['preserved_files'], 'all_preserved_hashes_reverified': True,
    'experiment': str(D.relative_to(ROOT)),
    'result_sha256': sha(D / 'result.json'), 'result': result,
    'independent_audit_sha256': sha(D / 'independent_audit.json'),
    'artifact_manifest_sha256': sha(D / 'artifact_manifest.json'),
    'resources': {'guards': 1, 'all_completed': True, 'all_retired': True,
                  'max_rss_bytes': peak, 'added_swap_bytes': 0,
                  'guard_elapsed_s': guard['elapsed_s'], 'free_disk_bytes': free,
                  'artifact_bytes_before_manifest': sum(v['bytes'] for v in files.values())},
    'numerical_jobs_running': False, 'exec_session_4467': 'reaped, exit 0',
    'targets_accessed': False, 'valid40_accessed': False, 'candidate_built': False,
    'submitted': False, 'pending_approval': False, 'pending_grading': False,
    'submission_authorization': 'Standing user authorization persists for meaningful couple-percent adjusted gains',
    'decision': 'No cache implementation: even free unlimited exact-input reuse saves less than 0.4% on either trace',
    'next_candidate': None,
    'retained_component': 'thin_star_route_v1: TRAIN8 adjusted -0.57346%; below submission bar and not held-out/actual-worker qualified',
}
once(checkpoint, record)
for original, backup in backups.items():
    once(backup, original.read_text())
for name in ['RESULTS.md', 'NEXT_RESEARCH_v1.md']:
    p = H / name
    old = backups[p].read_text()
    p.write_text(f'**Current: checkpoint25, {local}; S31 unchanged, no new banked gain.**\n'
                 'Exact transform reuse closes below 0.4% even with free unlimited storage. '
                 'All jobs retired. Earlier entries below are preserved chronology.\n\n' + old + '\n\n' + summary)
banner = f'''**{local}: CHECKPOINT25; no new banked gain.**
S31 #332517 remains 2.1736465056908115e-9 adjusted, 100/100 complete,
zero failures; source 5d439308 and archive 8d3ae68e unchanged.

Exact same-side input-transform reuse on unchanged native TRAIN35/100
saves at most 0.345746% / 0.381218% of complete bills even with free
unlimited storage. Fixed 1.5% gate fails; no numerical cache candidate.
409/407 transforms inspected, all prediction files and bills byte/exact
to independent controls. No truth, fit, VALID, new samples or upload.
One process retired, peak {peak / 1e9:.3f} GB, zero added swap. No numerical
worker, approval, upload or grading pending. Submission authorization stands.

Restart: codex_resume_20260926_v2/continuation_checkpoint_v25.json and
factor_transform_reuse_v1/RESULT.md. The prior fixed graph head remains
closed (+2.97073% CHECK raw). Thin-star TRAIN8 -0.57346% remains below
the upload bar and not held-out/actual-worker qualified. No new candidate
selected. Goal active; 2e-9 and approximately 1e-9 unmet.

'''
handoff = ROOT / 'HANDOFF_CURRENT.md'
handoff.write_text(banner + backups[handoff].read_text())
once(D / 'closeout.json', {
    'asof_utc': asof, 'status': 'closed_insufficient_cost_ceiling',
    'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': sha(checkpoint),
    'artifact_manifest_sha256': sha(D / 'artifact_manifest.json'),
    'handoff_sha256': sha(handoff), 'all_jobs_retired': True,
    'documentation_sha256': {n: sha(H / n) for n in ['RESULTS.md', 'NEXT_RESEARCH_v1.md']},
    'prior_documentation_backups': {str(p.relative_to(ROOT)): sha(p) for p in backups.values()},
})
for rel, value in files.items():
    assert sha(D / rel) == value['sha256'], rel
for rel, value in prior['preserved_files'].items():
    assert sha(ROOT / rel) == value, rel
assert handoff.read_text().endswith(backups[handoff].read_text())
print(json.dumps({'checkpoint': str(checkpoint), 'status': 'closed_insufficient_cost_ceiling',
                  'manifest_files_verified': len(files), 'peak_rss_bytes': peak,
                  'added_swap_bytes': 0, 'official_adjusted': 2.1736465056908115e-9,
                  'artifact_bytes_before_manifest': sum(v['bytes'] for v in files.values())}))
