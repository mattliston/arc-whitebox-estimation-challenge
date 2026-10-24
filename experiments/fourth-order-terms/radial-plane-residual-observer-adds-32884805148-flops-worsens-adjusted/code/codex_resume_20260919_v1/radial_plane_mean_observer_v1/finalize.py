"""Authenticate and preserve this fixed observer study without new numerical work."""
import ast
from datetime import datetime, timezone
import os
import shutil
import subprocess
from common import *

seal_check()
for p in read(HERE/'finalize_sources_sealed.json')['pins']:
    check(p)
complete, score, q, unit = [read(HERE/n) for n in
    ['completed.json', 'score.json', 'qualification.json', 'unit.json']]
assert complete['completed'] and complete['passed'] == score['passed']
assert q['passed'] and unit['passed'] and len(unit['cases']) == 6
assert score['independent_loss_reduction_agreed'] and not score['official_score']
assert score['passed'] == (all(r['adjusted_ratio'] < 1 for r in score['rows'])
                          and score['pooled_adjusted_ratio'] <= .99)
assert q['source_comparisons'] == q['ray_comparisons'] == q['recurrence_comparisons'] == 60
assert q['max_source_relative'] <= 2e-6
assert max(q['max_ray_relative'], q['max_recurrence_relative']) <= 3e-5
assert max(q['extra_fractions']) <= .10
build = read(HERE/'build.json')
for p in build.values():
    if isinstance(p, dict) and {'path', 'sha256', 'bytes'} <= p.keys():
        check(p)
assert (HERE/'candidate.py').read_bytes() == PARENT.read_bytes()+b'\n\n'+(HERE/'fragment.py').read_bytes()
old_source, new_source = PARENT.read_text(), (HERE/'candidate.py').read_text()
old_tree, new_tree = ast.parse(old_source), ast.parse(new_source)
assert len(build['inverses']) == 3
for item in build['inverses']:
    def extract(source, tree):
        node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == item['function']][-1]
        return ast.get_source_segment(source, node)
    old, new = extract(old_source, old_tree), extract(new_source, new_tree)
    assert hashlib.sha256(old.encode()).hexdigest() == item['original_sha256']
    assert hashlib.sha256(new.encode()).hexdigest() == item['appended_sha256']
    recovered = new
    for before, after, count in reversed(item['edits']):
        assert recovered.count(after) == count
        recovered = recovered.replace(after, before)
    assert recovered == old
names = ['unit', 'disabled', 'enabled', 'cold100', 'qualify', 'score']
guards = [done(n) for n in names]
groups = sorted(g['pid'] for g in guards)
remaining, workers = [], []
markers = [str(RUNTIME), str(D/'run_guarded.py'), str(HERE/'run.py')]
for line in subprocess.check_output(['/bin/ps', '-axo', 'pid=,pgid=,rss=,command='], text=True).splitlines():
    row = line.strip().split(None, 3)
    if len(row) != 4:
        continue
    if int(row[0]) in groups or int(row[1]) in groups:
        remaining.append(line)
    if int(row[0]) != os.getpid() and any(m in row[3] for m in markers):
        workers.append(line)
assert not remaining and not workers, (remaining, workers)
records = [r for n in ['disabled','enabled','cold100'] for r in read(HERE/n/'result.json')['records']]
for r in records:
    for key in ['prediction','shifts','weights']:
        check(r[key])
    if r['rays']:
        check(r['rays'])
    assert r['numeric_cache_retired'] and r['first_parent_row_exact']
    assert r['underlying_all16_parent_bytes_exact'] and r['underlying_context_bill_exact']
    assert r['flops']-r['extra_flops'] == r['baseline_context_flops']
sealed = read(HERE/'predictions_sealed.json')
check(sealed['candidate']); check(sealed['qualification'])
assert sealed['targets_not_yet_decoded']
for p in sealed['result_records']:
    check(p)
inputs = read(HERE/'inputs.json')
for row in inputs['rows']:
    for key in ['weights','truth','parent_result','prediction']:
        check(row[key])
for context in inputs['matching_contexts'].values():
    check(context['pin'])
bank = read(D/'official_best_331684.json')
bank_pins = [v for v in bank.values() if isinstance(v,dict) and {'path','sha256','bytes'} <= v.keys()]
assert len(bank_pins) == 5
for p in bank_pins:
    check(p)
previous = read(D/'continuation_checkpoint_v77.json')
check(previous['study']); check(previous['official_bank'])
artifact_bytes = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
assert artifact_bytes < 15_000_000
resources = dict(guarded_groups=len(guards), groups=groups, all_groups_retired=True,
    controller_terminal_session=77086, full_native_calls=len(records),
    full_native_flops=sum(r['flops'] for r in records),
    enabled_added_flops=sum(r['extra_flops'] for r in records),
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_self_reported_rss_bytes=max([r['peak_rss_bytes'] for r in records]
        +[unit['peak_rss_bytes'],q['peak_rss_bytes'],score['peak_rss_bytes']]),
    max_swap_growth_bytes=0, existing_reference_networks_decoded=2,
    new_samples=0, fits=0, new_downloads=0, GPU_calls=0,
    artifact_bytes_before_closeout=artifact_bytes, disk_free_bytes=shutil.disk_usage(HERE).free)
assert resources['full_native_calls'] == 5
assert resources['full_native_flops'] == q['native_full_flops']
assert resources['enabled_added_flops'] == q['enabled_added_flops']
assert max(resources['peak_group_rss_bytes'],resources['peak_self_reported_rss_bytes']) < 8_000_000_000
write(HERE/'resources.json',resources)
summary = dict(completed=True,negative=not score['passed'],score=score,resources=resources,
    qualification=q,unit=unit,whole_function_inverses=3,whole_parent_prefix_exact=True,
    underlying_parent_trajectory_and_bill_exact=True,fixed_natural_strength_mean_only_observer=True,
    full_native_tangent=False,official_changed=False,goal_achieved=False,next_numerical_study=None)
write(HERE/'summary.json',summary)
status = 'passes development; wider validation required' if score['passed'] else 'fails the complete adjusted objective'
report = f'''# Radial-plane residual mean observer: {status}

The bank's nonlinear maps form a full postactivation fourth repeated plane Q
and reduce it to a radial scalar. This fixed observer subtracts its own
symmetric radial projection from Q, casts that residual D to FP32, and
contracts the coordinate-pair tensor with the next actual weight matrix.
It then propagates a separate mean correction through the original Gaussian
gains. The natural coefficient is one at every layer; there is no fit or
selection. Parent means, covariances, cumulant banks, selectors, learned
heads and physical state remain unchanged. The original output receives
only the observer's mean shift before the original nonnegative FP32 cast.

| Existing 1B network | Raw error ratio | Complete cold cost ratio | Adjusted error ratio |
| --- | ---: | ---: | ---: |
'''
for r in score['rows']:
    report += f"| {r['rank']} | {r['raw_ratio']:.12f} | {r['cost_ratio']:.12f} | {r['adjusted_ratio']:.12f} |\n"
report += f'''
Pooled adjusted ratio: **{score['pooled_adjusted_ratio']:.12f}**. The fixed
gate requires both adjusted ratios <1 and pooled <=.99. Passed: {score['passed']}.
Cold parent bills are {score['rows'][0]['parent']['flops']} /
{score['rows'][1]['parent']['flops']} F; candidate bills are
{score['rows'][0]['candidate']['flops']} / {score['rows'][1]['candidate']['flops']} F.
Every formation, cast, copy, matrix product and read is charged inside
predict. Extra observer work is {records[1]['extra_flops']} F per call.
The cost multiplier is each full cold bill divided by 2^41, floored at .1.
Independent scalar-fsum and NumPy loss reductions agree. Predictions,
bills and qualification were sealed before ordinary reference target
decoding. These reused development networks are not held-out validation
and these ratios are not official scores.

Six tiny signed, zero and unequal-variance tensor cases pass, with explicit
dense fourth tensors, double-trace cancellation, coordinate rescaling,
permutation, independent mean recurrences, input preservation, and repeated
FNP bytes and bills. The zero double trace applies before FP32 rounding.
Three whole-function inverses and the complete parent byte prefix verify.
Disabled cold35 exactly reproduces all 16 parent rows and the complete bill.
Enabled35/100/35 and independent cold100 each preserve all 16 underlying
parent output rows, and subtracting measured helper work restores the
matching complete parent bill exactly. Own predictions, rays, sources and
shifts repeat exactly for repeated35 and warm/cold100. First output rows,
finite/nonnegative FP32 contract and numeric-cache retirement checks pass.

The independent diagnostic source, ray and full mean-recurrence comparisons
each cover 60 layer events. Their maximum relative errors are
{q['max_source_relative']:.12g}, {q['max_ray_relative']:.12g} and
{q['max_recurrence_relative']:.12g}, within frozen 2e-6 / 3e-5 / 3e-5 limits.
Research NumPy reference operations do not feed candidate results. Research
wall times are not production runtime qualification.

This tensor has only iiii and iijj entries. Its other collision sectors are
zero; the parent retains its complete existing baseline. The observer omits
covariance, learned-state and higher-cumulant feedback, and is not the full
native tangent or a complete fourth-state carrier. This is distinct from
the older fourth-plane-only chain, which replaced the fourth-state baseline.

Five full native calls total {resources['full_native_flops']} F. Six serial
guarded groups retired; controller 77086 is terminal. Peak sampled group
RSS is {resources['peak_group_rss_bytes']} B and self-report is
{resources['peak_self_reported_rss_bytes']} B; added swap is zero. Fast
stages can understate sampled peaks. Artifacts before closeout total about
{artifact_bytes/1e6:.3f} MB. No new samples, fits, downloads or GPU calls.
All five official source/archive/receipt/poll/evaluation pins are unchanged.
'''
if score['passed']:
    report += '\nA development pass requires a separately frozen wider panel and production runtime, client, contract and package checks before an authorized upload.\n'
else:
    report += '\nClose this fixed observer without amplitude, layer, source, precision, gain, fit or blend rescue. No wider panel, production qualification, package or upload follows. This result does not bound all fourth-state or joint-cumulant methods.\n'
report += '\nOfficial 331684 remains adjusted 3.5844144212297072e-9, with 100/100 complete and zero failures. The 4e-9 bank is retained; 3/2/~1e-9 remain unmet. The goal remains active.\n'
with (HERE/'RESULT.md').open('x') as f:
    f.write(report)
external = [p for p in read(HERE/'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]
for row in inputs['rows']:
    external += [row[k] for k in ['weights','truth','parent_result','prediction']]
external += bank_pins+[previous['study']]
external = list({p['path']:p for p in external}.values())
for p in external:
    check(p)
artifacts = [pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
write(HERE/'closeout.json',dict(completed=True,negative=not score['passed'],
    created_at_utc=datetime.now(timezone.utc).isoformat(),artifacts=artifacts,
    external_pins=external,resources=resources,official_changed=False,goal_achieved=False,
    candidate=pin(HERE/'candidate.py'),source_audit=pin(HERE/'SOURCE_REVIEW.md'),
    next_numerical_study=None))
print(dict(closeout=pin(HERE/'closeout.json'),artifacts=len(artifacts),external=len(external),resources=resources),flush=True)
