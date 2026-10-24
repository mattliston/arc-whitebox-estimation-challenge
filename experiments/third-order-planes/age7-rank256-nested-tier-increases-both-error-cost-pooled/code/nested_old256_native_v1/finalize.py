"""Verify and preserve this completed experiment; no numerical execution."""
from datetime import datetime,timezone
import shutil,subprocess
from common import *
seal_check()
for p in read(HERE/'finalize_sources_sealed.json')['pins']:check(p)
for p in read(HERE/'strict_sources_sealed.json')['pins']:check(p)
assert read(HERE/'completed.json')['completed']
score=read(HERE/'score.json');assert not score['passed']
assert all(r['raw_ratio']>1 and r['cost_ratio']>1 and r['adjusted_ratio']>1 for r in score['rows'])
assert read(HERE/'qualification.json')['passed'] and read(HERE/'unit.json')['passed'] and read(HERE/'strict_unit.json')['passed']
build=read(HERE/'build.json')
for v in build.values():
    if isinstance(v,dict) and 'path' in v:check(v)
assert (HERE/'candidate.py').read_bytes()==PARENT.read_bytes()+b'\n\n'+(HERE/'fragment.py').read_bytes()
names=['unit','disabled','enabled','cold100','qualify','score','strict_unit']
guards=[done(n) for n in names];groups=[g['pid'] for g in guards]
live=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines():
    row=line.strip().split(None,3)
    if len(row)==4 and (int(row[0]) in groups or int(row[1]) in groups):live.append(line)
assert not live,live
records=[r for n in ['disabled','enabled','cold100'] for r in read(HERE/n/'result.json')['records']]
for r in records:check(r['prediction']);check(r['weights'])
for r in read(HERE/'inputs.json')['rows']:
    for k in ['weights','truth','parent_result','prediction']:check(r[k])
strict=read(HERE/'strict_unit.json');errors=[]
for r in strict['records']:
    errors+=list(r.get('errors',{}).values())
    if 'error' in r:errors.append(r['error'])
artifact_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
assert artifact_bytes<30_000_000
resources=dict(guarded_groups=len(guards),all_groups_retired=True,full_native_calls=len(records),
    full_native_flops=sum(r['flops'] for r in records),peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_self_reported_rss_bytes=max(r['peak_rss_bytes'] for r in records),max_swap_growth_bytes=0,
    reference_networks_decoded=2,new_samples=0,fits=0,new_downloads=0,GPU_calls=0,
    artifact_bytes_before_closeout=artifact_bytes,disk_free_bytes=shutil.disk_usage(HERE).free)
write(HERE/'resources.json',resources)
summary=dict(completed=True,negative=True,score=score,strict_unit_max_error=max(errors),
    strict_unit_cases=len(strict['records']),whole_source_inverse=True,native_disabled_exact=True,
    enabled_first_eight_exact=True,repeat_and_warm_cold_bytes_exact=True,
    resources=resources,official_changed=False,goal_achieved=False,next_numerical_study=None)
write(HERE/'summary.json',summary)
report='''# Nested age-seven/rank-256 tier: worse error and complete cost

The fixed older-only second tier fails the actual adjusted objective on both
development networks. It is closed without a panel, fit, client run, package
or upload. This is new native evidence for this current-parent mechanism,
not a reinterpretation of the older source-only nested224 plan.

| Existing 1B development network | Raw error ratio | Complete cold cost ratio | Adjusted error ratio |
| --- | ---: | ---: | ---: |
'''
for r in score['rows']:
    report+=f"| {r['rank']} | {r['raw_ratio']:.12f} | {r['cost_ratio']:.12f} | {r['adjusted_ratio']:.12f} |\n"
report+=f'''
Pooled adjusted ratio is **{score['pooled_adjusted_ratio']:.12f}**, about
2.21% worse. Rank35 cold bill rises from413448696909 to415643617964 F;
rank100 rises from416264792412 to419125225959 F. All errors were recomputed
against the same existing ordinary1B references using each arm's complete
own-cold bill. These reused development networks are not pristine validation
or official scores. The failure is not merely a proxy or tolerance failure:
both actual raw errors AND actual complete costs increased.

The implementation keeps age-four/rank-384 for the first tier, moves cohorts
0..6 into an older rank-256 tier at layers8..14, and carries both at15. Every
signed column and coefficient, full1024-row reconstruction, young/newborn
factor, learned parameter, selector, physical term, kernel and cache owner
remains. The middle factors rotate explicitly after layer7, replacing the
old Lazy schedule. New QR, projections, core subtraction/restoration and
extra reads are all paid. The old nested224 source-price saving cannot be
added to this parent's already reduced rotation bill.

The parent is an exact byte prefix of the candidate; removing the appended
fragment restores the entire original source. Disabled cold35 matches all16
parent output bytes and its full bill. Enabled35/100/35 and independent
cold100 pass domain and numeric-cache retirement checks, preserve the first
eight parent layers exactly, and repeat all output bytes and cohort layouts.
Full repeated-call bills differ with context history and are retained; only
the cold bills are used for the paired score. No all-layer equality to the
parent or production client/runtime qualification is claimed.

Eleven actual-FNP small/projection/read fixtures pass. The primary unit's
max(1,norm) normalization was identified and preserved, then tightened in an
additional audit using true relative error for every nonzero reference and
absolute error for an exactly zero reference. Same fixtures and3e-5 threshold;
maximum strict error{max(errors):.12g}. This audit is not a new independent
implementation. Dense reference Gram, factor/core repair, full-rank identity
and explicit cubic tensor comparisons check the code's intended mathematics.
Tiny helper tests use classical FNP products; the full native calls exercise
the unchanged production kernels. Unit repeat bytes and bills are exact.

Five full native calls total{resources['full_native_flops']} F. All seven
guarded groups exited with zero added swap. Sampled peak RSS is
{resources['peak_group_rss_bytes']} B; self-reported peak is
{resources['peak_self_reported_rss_bytes']} B. Short-stage sampled RSS can
understate peak memory; use self-report or the whole-study maximum. About
{artifact_bytes/1e6:.3f} MB of artifacts before closeout; no new samples,
training, downloads or GPU. Controller14932 is terminal.

No rank, age, metric, precision or fit rescue is selected. This closes this
fixed candidate, not every possible older-source representation. Official
331684 remains adjusted3.5844144212297072e-9,100/100 complete,zero failures.
The4e-9 crossing is banked;3/2/~1e-9 remain unmet. Goal active.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(report)
external=read(HERE/'sources_sealed.json')['pins']
external=[p for p in external if not Path(p['path']).is_relative_to(HERE)]
for r in read(HERE/'inputs.json')['rows']:
    external += [r[k] for k in ['weights','truth','parent_result','prediction']]
external=list({p['path']:p for p in external}.values())
for p in external:check(p)
artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
write(HERE/'closeout.json',dict(completed=True,negative=True,created_at_utc=datetime.now(timezone.utc).isoformat(),
    artifacts=artifacts,external_pins=external,resources=resources,official_changed=False,goal_achieved=False,
    candidate=pin(HERE/'candidate.py'),source_audit=pin(HERE/'SOURCE_REVIEW.md'),next_numerical_study=None))
print(dict(closeout=pin(HERE/'closeout.json'),artifacts=len(artifacts),external=len(external),resources=resources),flush=True)
