"""Preserve the failed native candidate and the complete continuation state."""
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil

D = Path(__file__).resolve().parent
H = D.parent
ROOT = next(p for p in D.parents if (p / 'HANDOFF_CURRENT.md').is_file())
OUT = H / 'old_row_importance_train2_v1'
read = lambda p: json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def once(p, value):
    with p.open('x') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False) + '\n')


checkpoint = H / 'continuation_checkpoint_v27.json'
backups = {
    ROOT / 'HANDOFF_CURRENT.md': ROOT / 'HANDOFF_CURRENT.md.before_codex_resume_v2_checkpoint27',
    H / 'RESULTS.md': H / 'RESULTS_before_checkpoint27.md',
    H / 'NEXT_RESEARCH_v1.md': H / 'NEXT_RESEARCH_before_checkpoint27.md',
}
assert all(not p.exists() for p in [checkpoint, D / 'RESULT.md', D / 'closeout.json', *backups.values()])
prior = read(H / 'continuation_checkpoint_v26.json')
for rel, expected in prior['preserved_files'].items():
    assert sha(ROOT / rel) == expected, rel
for p, expected in read(D / 'pre_endpoint_seal.json')['files'].items():
    assert sha(Path(p)) == expected, p
assessment = read(D / 'assessment.json')
assert not assessment['passes_train2_gate']
assert assessment['source_seal_verified'] and assessment['scores_independently_reconciled']
assert all(r['first7_output_bytes_exact'] and not r['adjusted_win'] for r in assessment['networks'])
guards = {name: read(D / name) for name in ['qualification_guard.json', 'train2_guard.json', 'assessment_guard.json']}
assert all(g['completed'] and g['whole_group_retired'] and not g['remaining_rows'] and
           g['exit_code'] == 0 and g['max_swap_growth_bytes'] == 0 for g in guards.values())
now = datetime.now(timezone.utc)
asof = now.isoformat()
local = now.astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M %Z')
peak = max(g['max_rss_bytes'] for g in guards.values())
free = shutil.disk_usage(ROOT).free
change = assessment['pooled_change_pct']
summary = f'''## Checkpoint27 — {local}: magnitude-aware old-row allocation fails natively

Official S31 #332517 remains **2.1736465056908115e-9 adjusted**,100/100
complete,zero failures. Bank source/archive and retained thin-star source
are unchanged. No new banked gain, approval, upload or grading is pending.
Standing authorization for meaningful adjusted-score improvements persists.

The new fixed candidate SHA{assessment['candidate_sha256']} replaces
S31's local old-row allocation with Gaussian-curvature/normalized-row-magnitude
ranking, retains128 fewer rows per call, and protects every actual live
negative row. It preserves the full carried tensor, all young columns,
existing diagonal/radial repairs and learned parameters. This is a complete
allocation rule, not an isolated matched-budget test of the magnitude term.

Native TRAIN35/100: pooled raw **{change['mse']:+.6f}%**, complete bill
**{change['flops']:+.6f}%**, adjusted **{change['adjusted']:+.6f}%**,0/2 wins.
Per-network adjusted changes are+44.123438%/+39.436417%. Both prospective
score gates fail. Close this fixed rule; no count, score-feature, threshold
or coefficient rescue, six-network expansion, refit, VALID40 or upload.
The cost saving alone is also below its1.5% adjusted advance threshold.

Nine independent scalar selection fixtures pass, including mandatory
membership, stable ties, full row partitions, zero magnitude and no input
mutation. Mathematical rotation/row-scale invariance holds on the checked
well-conditioned, unfloored inputs, not necessarily across numerical floors
or bitwise arithmetic. Native first7 output layers are byte-exact to S31;
both final MSEs and own-bill adjusted scores independently reconcile via
scalar sums. Exactly8 calls per network omit1024 additional local row
reconstructions in total. These are reused TRAIN networks, not validation.

All three guards completed and retired, sampled peak RSS{peak / 1e9:.3f}GB,
zero added swap. Sessions5137 and26447 reapedexit0; assessment completed
directly with exit0. No numerical worker is running. No new samples or bulk
data. Thin-star TRAIN8-0.57346% remains below the release bar and not
held-out/actual-worker qualified. No new candidate is selected;2e-9 and
approximately1e-9 remain unmet and active.

Read `old_row_importance_v1/RESULT.md`, `assessment.json`,
`pre_endpoint_seal.json`, `SCOPE.md`, and the three guard receipts.
Restart from `continuation_checkpoint_v27.json`.
'''
once(D / 'RESULT.md', '# Fixed old-row importance candidate: closed\n\n' + summary)
source_notes = {
    H / 'candidate_old_both_tail_trace_bounded_v6.py': 'Parent local old-row mask uses alpha and its negative-dead condition; complete carry remains separate from local reconstruction.',
    H.parent / 'codex_resume_20260919_v1/canonical_selector_response_v1/SOURCE_REVIEW.md': 'Older canonical-selector diagnostic changed order without changing top-count membership; this new test changes membership and count and is distinct.',
    H.parent / 'fable_physicist_v1/lean/RESULT_LEAN.md': 'Consumption-ranked birth pruning is a different carrier intervention; its completed negative is not relabeled as untried.',
    H.parent / 'fable_covhunter_v1/RESULT.md': 'The warm-start width experiment already had a matched control. R320 routing and rank compensation also have completed negative evidence.',
    H.parent / 'fable_engineer_v1/RESULT.md': 'Symmetric-output cost improvements are already implemented. The older census does not identify a free symmetric-input C-times-W discount.',
}
once(D / 'source_review.json', dict(asof_utc=asof,
    scope='Read-only prior evidence; no new training, routing fit or symmetric-input cost claim',
    sources=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p), finding=v) for p,v in source_notes.items()]))
files = {str(p.relative_to(H)): dict(sha256=sha(p), bytes=p.stat().st_size)
         for directory in [D, OUT] for p in sorted(directory.iterdir()) if p.is_file()}
total = sum(v['bytes'] for v in files.values())
assert total < 10_000_000
once(D / 'artifact_manifest.json', dict(asof_utc=asof, files=files, total_bytes=total,
    scope='Study and endpoint artifacts before manifest, checkpoint and closeout'))
record = dict(asof_utc=asof, goal='active;2e-9 and approximately1e-9 unmet',
    previous_turn_classification='Progress: two scoped negative diagnostics changed the next action; no score gain',
    classification='Progress: new fixed native allocation candidate built, qualified and rejected on actual own-cost score',
    official_bank=prior['official_bank'], official_bank_repolled=False,
    preserved_files=prior['preserved_files'], all_preserved_hashes_reverified=True,
    experiment=str(D.relative_to(ROOT)), candidate_sha256=sha(D / 'candidate.py'),
    assessment_sha256=sha(D / 'assessment.json'), assessment=assessment,
    manifest_sha256=sha(D / 'artifact_manifest.json'),
    resources=dict(groups=3, all_completed=True, all_retired=True, max_rss_bytes=peak,
                   added_swap_bytes=0, total_guard_seconds=sum(g['elapsed_s'] for g in guards.values()),
                   artifact_bytes_before_manifest=total, free_disk_bytes=free),
    numerical_jobs_running=False, exec_sessions={'5137':'reaped,exit0','26447':'reaped,exit0'},
    candidate_built=True, submitted=False, final_mean_training_truth_read=True,
    fits_run=False, valid40_accessed=False, new_samples=False, pending_approval=False,
    pending_grading=False, next_candidate=None,
    submission_authorization='Standing user authorization persists for meaningful couple-percent adjusted improvements',
    decision='Close the fixed allocation rule; no rescue, expansion or validation',
    retained_component=prior['retained_component'])
once(checkpoint, record)
for original, backup in backups.items():
    once(backup, original.read_text())
for name in ['RESULTS.md','NEXT_RESEARCH_v1.md']:
    p = H / name
    p.write_text(f'**Current: checkpoint27,{local};S31 unchanged,no new banked gain.**\n'
                'Fixed magnitude-aware old-row allocation fails native TRAIN2. All jobs retired. Earlier entries are preserved chronology.\n\n'
                + backups[p].read_text() + '\n\n' + summary)
banner = f'''**{local}: CHECKPOINT27; no new banked gain.**
S31 #332517 remains2.1736465056908115e-9 adjusted,100/100 complete,
zero failures; source5d439308 and archive8d3ae68e unchanged.

Fixed old-row magnitude/sensitivity allocation SHA{assessment['candidate_sha256'][:8]}
fails native TRAIN2: raw+43.50464%,bill-1.25655%,adjusted+41.70319%,0/2 wins.
First7 outputs byte-exact; full own scores independently reconciled.
No count/feature/threshold rescue, TRAIN6, refit, VALID or upload.

All three guards retired,peak{peak / 1e9:.3f}GB,zero added swap. No worker,
approval,upload or grading pending. Submission authorization stands.
Thin-star TRAIN8-0.57346% remains below bar and not held-out/actual-worker
qualified. No new candidate selected. Goal active;2e-9/~1e-9 unmet.

Restart: codex_resume_20260926_v2/continuation_checkpoint_v27.json and
old_row_importance_v1/RESULT.md, assessment.json, SCOPE.md.

'''
handoff = ROOT / 'HANDOFF_CURRENT.md'
handoff.write_text(banner + backups[handoff].read_text())
once(D / 'closeout.json', dict(asof_utc=asof, status='closed_native_train2_failed',
    checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
    artifact_manifest_sha256=sha(D / 'artifact_manifest.json'), handoff_sha256=sha(handoff),
    documentation_sha256={n:sha(H / n) for n in ['RESULTS.md','NEXT_RESEARCH_v1.md']},
    prior_documentation_backups={str(p.relative_to(ROOT)):sha(p) for p in backups.values()},
    all_groups_retired=True, source_archive_unchanged=True, board_submitted=False))
for name, meta in files.items():
    assert sha(H / name) == meta['sha256'], name
for rel, expected in prior['preserved_files'].items():
    assert sha(ROOT / rel) == expected, rel
assert handoff.read_text().endswith(backups[handoff].read_text())
print(json.dumps(dict(checkpoint=str(checkpoint), status='closed_native_train2_failed',
    artifact_bytes=total, verified_artifacts=len(files), peak_rss_bytes=peak,
    added_swap_bytes=0, official_adjusted=2.1736465056908115e-9, goal='active')))
