"""Recheck and preserve the completed native test; no numerical execution."""
import ast
from datetime import datetime, timezone
import os
import shutil
import subprocess
from common import *

seal_check()
for p in read(HERE / 'finalize_sources_sealed.json')['pins']:
    check(p)
complete = read(HERE / 'completed.json')
score = read(HERE / 'score.json')
qualification = read(HERE / 'qualification.json')
unit = read(HERE / 'unit.json')
assert complete['completed'] and complete['passed'] == score['passed']
assert qualification['passed'] and unit['passed']
assert score['independent_loss_reduction_agreed'] and not score['official_score']
assert score['passed'] == (all(r['adjusted_ratio'] < 1 for r in score['rows'])
                            and score['pooled_adjusted_ratio'] <= .99)
build = read(HERE / 'build.json')
for value in build.values():
    if isinstance(value, dict) and {'path', 'sha256', 'bytes'} <= value.keys():
        check(value)
assert (HERE / 'candidate.py').read_bytes() == PARENT.read_bytes() + b'\n\n' + (HERE / 'fragment.py').read_bytes()
def last_function(source, name):
    tree = ast.parse(source)
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name][-1]
    return ast.get_source_segment(source, node)
old = last_function(PARENT.read_text(), '_predict_binary')
new = last_function((HERE / 'candidate.py').read_text(), '_predict_binary')
assert new.replace('_cubic_support_select(', 'select(') == old
calls = [n.func.id for n in ast.walk(ast.parse(new)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
assert calls.count('_cubic_support_select') == 2 and calls.count('select') == 1
names = ['unit', 'disabled', 'enabled', 'cold100', 'qualify', 'score']
guards = [done(n) for n in names]
groups = sorted(g['pid'] for g in guards)
remaining, workers = [], []
markers = [str(RUNTIME), str(D / 'run_guarded.py'), str(HERE / 'run.py')]
for line in subprocess.check_output(['/bin/ps', '-axo', 'pid=,pgid=,rss=,command='], text=True).splitlines():
    row = line.strip().split(None, 3)
    if len(row) != 4:
        continue
    if int(row[0]) in groups or int(row[1]) in groups:
        remaining.append(line)
    if int(row[0]) != os.getpid() and any(m in row[3] for m in markers):
        workers.append(line)
assert not remaining and not workers, (remaining, workers)
records = [r for n in ['disabled', 'enabled', 'cold100'] for r in read(HERE / n / 'result.json')['records']]
for r in records:
    check(r['prediction']); check(r['weights'])
    assert r['numeric_cache_retired'] and r['first_parent_row_exact']
    if r['full_cubic_support']:
        assert all(a['count'] == 1024 for a in r['rosters'])
for r in read(HERE / 'inputs.json')['rows']:
    for k in ['weights', 'truth', 'parent_result', 'prediction']:
        check(r[k])
for p in read(HERE / 'predictions_sealed.json')['result_records']:
    check(p)
bank = read(D / 'official_best_331684.json')
bank_pins = [v for v in bank.values() if isinstance(v, dict) and {'path', 'sha256', 'bytes'} <= v.keys()]
for p in bank_pins:
    check(p)
previous = read(D / 'continuation_checkpoint_v75.json')
check(previous['study']); check(previous['official_bank'])
artifact_bytes = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
assert artifact_bytes < 20_000_000
resources = dict(guarded_groups=len(guards), all_groups_retired=True, controller_terminal_session=96275,
    full_native_calls=len(records), full_native_flops=sum(r['flops'] for r in records),
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_self_reported_rss_bytes=max(r['peak_rss_bytes'] for r in records), max_swap_growth_bytes=0,
    reference_networks_decoded=2, new_samples=0, fits=0, new_downloads=0, GPU_calls=0,
    artifact_bytes_before_closeout=artifact_bytes, disk_free_bytes=shutil.disk_usage(HERE).free)
write(HERE / 'resources.json', resources)
summary = dict(completed=True, negative=not score['passed'], score=score, resources=resources,
    selector_cases=unit['selector_cases'], dense_cases=unit['dense_cases'],
    strict_tensor_max_error=unit['strict_tensor_max_error'], whole_source_inverse=True,
    exactly_two_support_calls_changed=True, birth_selector_unchanged=True,
    native_disabled_exact=True, enabled_first_row_exact=True,
    repeat_output_roster_and_cohorts_exact=True, all_enabled_rosters_full=True,
    official_changed=False, goal_achieved=False, next_numerical_study=None)
write(HERE / 'summary.json', summary)
status = 'passes development; further validation required' if score['passed'] else 'fails the complete adjusted objective'
report = f'''# Full cubic row support: {status}

The fixed candidate restores every neuron row to the current bank's cubic
factor transport, row gates, shared basis, pair reads and residual-star
bookkeeping. Birth membership still uses the original .01 threshold and
padding. The all-row response set has no omitted-response radial correction;
the original computation remains present and paid. Covariance, star and
trace support were already full. This adds no new cumulant formula or fit.

| Existing 1B network | Raw error ratio | Complete cold cost ratio | Adjusted error ratio |
| --- | ---: | ---: | ---: |
'''
for r in score['rows']:
    report += f"| {r['rank']} | {r['raw_ratio']:.12f} | {r['cost_ratio']:.12f} | {r['adjusted_ratio']:.12f} |\n"
report += f'''
Pooled adjusted ratio: **{score['pooled_adjusted_ratio']:.12f}**. Fixed gate:
both adjusted ratios <1 and pooled <=.99. Passed: {score['passed']}.
Each arm pays its independently measured complete cold bill, including
initialization; no old warm bill is substituted. Parent bills are
413448696909 / 416264792412 F. Candidate bills are
{score['rows'][0]['candidate']['flops']} / {score['rows'][1]['candidate']['flops']} F.
Losses were independently accumulated with math.fsum and NumPy. Predictions,
costs and qualification were sealed before decoding the existing ordinary1B
references. These reused development networks are not held-out validation
or official scores.

Only two selector call names change in a copied last predictor; the appended
helper retains descending stable score order and returns full support.
Whole-function inverse and exact original-source byte prefix are verified.
The original birth selector, learned literals, formulas, rank384/age4/Lazy
schedule, matrix kernels and cache ownership are retained. Future state,
birth counts and kernel dispatches are endogenous; no unchanged-trajectory
claim is made. The disabled cold35 call reproduces all16 parent output bytes
and the full original bill. Enabled35/100/35 and independent cold100 preserve
the first parent output row and satisfy finite/nonnegative FP32 output and
numeric-cache retirement checks. All30 live/response rosters have1024 entries.
Repeated35 and warm/cold100 outputs, ordered rosters and cohort metadata
match exactly. Warm/repeat FLOP bills are retained separately.

All{unit['selector_cases']} actual-FNP selector fixtures pass, including both
enabled and disabled behavior, empty inputs, stable ties and padded lengths.
Repeated selector bytes and bills match and inputs remain unchanged.
Six independent dense signed cubic fixtures verify gate/affine transport,
complete repeated-plane reads and star restoration, maximum strict relative
error {unit['strict_tensor_max_error']:.12g} (absolute for zero references)
against2e-12. A generic de-repeated tensor witness has identical zero current
repeated planes but a nonzero future contraction after removing a small row.
It motivates the information question, not an accuracy forecast or a claim
about a small padded native selector.

Five full native calls total{resources['full_native_flops']} F. Six guarded
groups retired; controller96275 terminal. Peak sampled group RSS
{resources['peak_group_rss_bytes']} B, self-report
{resources['peak_self_reported_rss_bytes']} B; zero added swap. Artifacts before
closeout: about{artifact_bytes/1e6:.3f} MB. No new samples, fits, downloads or
GPU calls. Local research native wall times are not production-client runtime
qualification. No official submission occurred during this study.
'''
if not score['passed']:
    report += '''
Close this fixed full cubic-support candidate without a threshold, layer,
partial-support, source subset, rank, fit, precision or blend rescue. No panel,
client qualification, package or upload follows. This does not establish a
universal support or approximation bound. The previous eight-summary feature
test and all earlier negatives retain their own scopes.
'''
else:
    report += '\nThis development pass still needs separately declared panel, confirmation, client/runtime/contract and package checks before any upload.\n'
report += '''
Official331684 remains adjusted3.5844144212297072e-9,100/100 complete,zero
failures.4e-9 is banked;3/2/~1e-9 remain unmet. The goal remains active.
'''
with (HERE / 'RESULT.md').open('x') as f:
    f.write(report)
external = [p for p in read(HERE / 'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]
for r in read(HERE / 'inputs.json')['rows']:
    external += [r[k] for k in ['weights', 'truth', 'parent_result', 'prediction']]
external += bank_pins + [previous['study']]
external = list({p['path']: p for p in external}.values())
for p in external:
    check(p)
artifacts = [pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
write(HERE / 'closeout.json', dict(completed=True, negative=not score['passed'],
    created_at_utc=datetime.now(timezone.utc).isoformat(), artifacts=artifacts,
    external_pins=external, resources=resources, official_changed=False, goal_achieved=False,
    candidate=pin(HERE / 'candidate.py'), source_audit=pin(HERE / 'SOURCE_REVIEW.md'),
    next_numerical_study=None))
print(dict(closeout=pin(HERE / 'closeout.json'), artifacts=len(artifacts),
    external=len(external), resources=resources), flush=True)

