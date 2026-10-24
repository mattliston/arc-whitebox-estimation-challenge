"""Preserve the completed fixed-protocol negative without opening CHECK labels."""
import datetime
import shutil
from common import HERE, D, PARENT, TRAIN, read, write, pin, verify, seal_check

for name in ['capture_sources_sealed.json', 'fit_sources_sealed.json', 'fit_sources_sealed_v2.json']:
    seal_check(name)
for p in read(HERE/'analysis_sources_sealed.json')['pins']:
    verify(p)
cv = read(HERE/'cv_result.json')
outcome = read(HERE/'outcome.json')
assert not cv['passed'] and outcome['study_closed'] and not outcome['full_fit']
assert read(HERE/'analysis_completed.json')['completed']
assert read(HERE/'captures_sealed.json')['passed']
assert read(HERE/'unit.json')['passed'] and read(HERE/'unit_v2.json')['passed']
for name in ['terminal.npz', 'check_result.json', 'check_predictions_sealed.json', 'candidate.py', 'submission_attempt.json']:
    assert not (HERE/name).exists(), name
assert len(list(HERE.glob('fold*_*.npz'))) == 10
guards = {p.name: read(p) for p in HERE.glob('*_watch.json')}
assert len(guards) == 36
assert all(g['completed'] and g['whole_group_retired'] and g['exit_code'] == 0
           and g['max_swap_growth_bytes'] == 0 and g['max_rss_bytes'] < 8_000_000_000
           for g in guards.values())
native = []
for p in sorted(HERE.glob('*/result.json')):
    native.extend(read(p)['records'])
assert len(native) == 32 and all(r['max_rss_bytes'] < 8_000_000_000 for r in native)
leaves = [v for k, v in guards.items() if k not in ['capture_controller_watch.json', 'analysis_controller_watch.json']]
resources = dict(
    guarded_groups=len(guards), leaf_guarded_groups=len(leaves), all_groups_retired=True,
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards.values()), max_swap_growth_bytes=0,
    leaf_elapsed_s=sum(g['elapsed_s'] for g in leaves), native_full_calls=len(native),
    native_full_flops=sum(r['flops'] for r in native),
    peak_native_self_rss_bytes=max(r['max_rss_bytes'] for r in native),
    peak_fit_self_rss_bytes=cv['peak_rss_bytes'], cv_fits=10, full_fits=0,
    new_samples=0, new_dataset_downloads=0, GPU_calls=0, client_calls=0,
    disk_free_bytes=shutil.disk_usage(HERE).free,
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()))
assert resources['artifact_bytes_before_closeout'] < 500_000_000
write(HERE/'resources.json', resources)
lines = [
    '# Frozen-current-parent all-layer mean observer: closed negative', '',
    'Official parent #331684 remains adjusted 3.5844144212297072e-9, with all 100 networks completed.',
    'This separate 102-coefficient FP64 mean observer used the current frozen parent trajectory,',
    'existing 1B mean targets at all eight late layers, and five fixed network folds.',
    'All 20 training captures reproduce the parent output bytes and complete bills.',
    'Independent scalar, ridge, constant-column, and actual FNP tests passed before target decoding.',
    'The source-only normalization repair preceded all fits and retained the original unexecuted draft.', '',
    f"- Pooled adjusted recurrent/parent ratio: {cv['pooled_adjusted_ratio']:.15g} (required <= 0.95).",
    f"- Recurrent/local-control ratio: {cv['recurrent_over_local']:.15g} (required <= 0.99).",
    f"- Parent wins: {cv['wins']}/20 networks and {cv['fold_wins']}/5 folds (required >= 12 and >= 4).",
    f"- Worst network ratio: {cv['worst_adjusted_ratio']:.15g} (required <= 1.20).", '',
    'The declared CV gate failed. No all-training fit, CHECK label decoding, native candidate,',
    'panel, confirmation, client qualification, package, or upload followed.',
    'The local control is not substituted as a rescue. The ten fixed fits and all scores are preserved.',
    'Scores use the declared provisional 1GF increment; no native added-cost qualification is claimed.',
    'These are reused public research networks, not pristine holdouts or official scores.', '',
    f"Resources: {len(native)} full native calls; {len(guards)} retired guarded groups; zero added swap;",
    f"peak sampled group RSS {resources['peak_group_rss_bytes']} bytes;",
    f"{resources['artifact_bytes_before_closeout']} artifact bytes before closeout; no new samples, downloads, or GPU.",
    'The approximately 1e-9 goal remains active and unachieved.', '']
with (HERE/'RESULT.md').open('x') as f:
    f.write('\n'.join(lines))
files = sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
record = dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              outcome=outcome, cv={k:v for k,v in cv.items() if k not in ['rows', 'folds']},
              resources=resources, uploaded=False, goal_achieved=False,
              artifact_pins=[pin(p) for p in files],
              external_pins=[pin(PARENT), pin(D/'official_best_331684.json')])
for p in record['artifact_pins'] + record['external_pins']:
    verify(p)
write(HERE/'closeout.json', record)
print(dict(artifact_pins=len(files), cv=record['cv'], resources=resources), flush=True)
