"""Close the positive-leaf scalar study after the controller is terminal."""
from datetime import datetime, timezone
import os, shutil, subprocess
from common import *

seal_check()
for p in read(HERE/'finalize_sources_sealed.json')['pins']:check(p)
assert read(HERE/'completed.json')['completed']
result=read(HERE/'reprice_result.json');audit=read(HERE/'audit_result.json');paths=read(HERE/'paths_verified.json')
assert result['completed'] and audit['passed'] and paths['passed']
assert result['maximum_stages']==audit['structural_maximum_depth']==14
assert result['minimum_leaf_dimension']==1 and audit['complete_positive_leaf_family']
assert audit['lossless_frontier_pruning_verified'] and audit['all_old_call_savings_nonworsening']
assert audit['positive_shape_checks']==586 and audit['winograd_rational_identities']==15
assert len(audit['exhaustive_checks'])==2224 and audit['abstract_monotone_ancestor_checks']==864
assert paths['schedules_replayed']==2360 and paths['call_records_reconciled']==1180
assert paths['catalog_superset_monotonicity'] and paths['all_call_records_reconciled']
assert paths['result_sha256']==sha(HERE/'reprice_result.json')
assert result['new_model_calls']==result['truth_arrays_read']==result['fits']==0
assert not result['candidate'] and not result['native_runtime_qualification']
for key in ['producer','result','path_replay','old_result']:check(audit[key])
before=read(audit['old_result']['path'])
old_families={f['name']:f for f in before['families']}
new_families={f['name']:f for f in result['families']}
assert len(old_families)==len(before['families'])==len(new_families)==len(result['families'])==2
assert old_families.keys()==new_families.keys()
calls=choices=changed_calls=changed_choices=0
for name,new in new_families.items():
    old=old_families[name];assert old['programs']==new['programs']
    old_records={r['rank']:r for r in old['records']};new_records={r['rank']:r for r in new['records']}
    assert len(old_records)==len(old['records'])==len(new_records)==len(new['records'])==2
    assert old_records.keys()==new_records.keys()
    for rank,record in new_records.items():
        prior=old_records[rank];assert prior['complete_cold_flops']==record['complete_cold_flops']
        old_details={d['index']:d for d in prior['details']};new_details={d['index']:d for d in record['details']}
        assert len(old_details)==len(prior['details'])==len(new_details)==len(record['details'])==295
        assert old_details.keys()==new_details.keys()
        for index,entry in new_details.items():
            original=old_details[index]
            for field in ['shape','caller','actual_call_flops']:assert original[field]==entry[field]
            altered=False
            for field in ['optimistic_saved','leaf_storage_limited_saved']:
                assert entry[field]>=original[field]
                altered |= entry[field]!=original[field]
            changed_calls+=altered
            for field in ['optimistic_best','leaf_storage_limited_best']:
                changed_choices+=entry[field]!=original[field];choices+=1
            calls+=1
assert (calls,choices)==(1180,2360)
assert changed_calls==len(audit['changed_calls']) and changed_choices==audit['changed_chosen_entries']
build=read(HERE/'build.json')
for inverse in build['source_inverses']:
    check(inverse['old']);check(inverse['new'])
    recovered=Path(inverse['new']['path']).read_text()
    for a,b,count in reversed(inverse['edits']):
        assert recovered.count(b)==count
        recovered=recovered.replace(b,a)
    assert recovered==Path(inverse['old']['path']).read_text()
check(build['previous_checkpoint']);check(build['prior_study'])
guards=[done(n) for n in ['reprice','verify_paths','audit']]
groups=sorted(g['pid'] for g in guards);remaining=[];workers=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines():
    row=line.strip().split(None,3)
    if len(row)!=4:continue
    if int(row[0]) in groups or int(row[1]) in groups:remaining.append(line)
    if int(row[0])!=os.getpid() and any(s in row[3] for s in [str(RUNTIME),str(D/'run_guarded.py'),str(HERE/'run.py')]):workers.append(line)
assert not remaining and not workers,(remaining,workers)
bank=read(D/'official_best_331684.json')
assert sha(D/'official_best_331684.json')=='b6ef6e0284d1ac82eced28377b8ccf9846190797b23ad631de5aeed14f760d49'
bank_pins=[p for p in bank.values() if isinstance(p,dict) and {'path','sha256','bytes'}<=p.keys()]
assert len(bank_pins)==5
for p in bank_pins:check(p)
previous=read(D/'continuation_checkpoint_v79.json');check(previous['study']);check(previous['official_bank'])
write(HERE/'chosen_entries_verified.json',dict(passed=True,old_result=audit['old_result'],new_result=pin(HERE/'reprice_result.json'),
    call_records=calls,chosen_entries=choices,changed_saving_records=changed_calls,changed_chosen_entries=changed_choices,
    all_savings_nonworsening=True,all_cost_memory_path_entries_identical=changed_choices==0))
artifact_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file());assert artifact_bytes<20_000_000
resources=dict(guarded_groups=3,all_groups_retired=True,controller_terminal_session=46267,
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_self_reported_rss_bytes=max(r['peak_rss_bytes'] for r in [result,audit,paths]),
    max_swap_growth_bytes=0,full_native_calls=0,new_native_flops=0,reference_arrays_decoded=0,new_samples=0,
    fits=0,new_downloads=0,GPU_calls=0,artifact_bytes_before_closeout=artifact_bytes,disk_free_bytes=shutil.disk_usage(HERE).free)
write(HERE/'resources.json',resources)
families=[dict(name=f['name'],states=f['states'],wall_s=f['wall_s'],
    passes_necessary_3e9_gate=f['passes_necessary_3e9_gate'],
    leaf_storage_limit_passes_necessary_3e9_gate=f['leaf_storage_limit_passes_necessary_3e9_gate'],
    records=[{k:v for k,v in r.items() if k!='details'} for r in f['records']]) for f in result['families']]
negative=not any(f['passes_necessary_3e9_gate'] for f in families)
gate_sentence=('Both catalogs miss the gate.' if negative else 'At least one catalog passes the optimistic arithmetic gate.')
summary=dict(completed=True,negative=negative,candidate=False,
    scope='Complete all-positive-leaf uniform-child recursion for the same two catalogs, ceil padding or floor core with classical fringes; not all matrix algorithms or streaming schedules.',
    minimum_leaf_dimension=1,structural_maximum_depth=14,shape_only_cache=True,lossless_arena_frontier_pruning=True,
    families=families,required_uniform_fraction=result['required_uniform_fraction_to_cross_3e9_at_unchanged_raw'],
    comparison=audit['comparison'],changed_call_savings=changed_calls,changed_chosen_entries=changed_choices,
    current_small_leaf_dispatch_witnesses=len(audit['current_small_leaf_dispatch_witnesses']),chosen_leaf_histogram=audit['chosen_leaf_histogram'],
    verified_schedules=2360,verified_call_records=1180,exhaustive_fixture_comparisons=2224,
    abstract_ancestor_checks=864,winograd_rational_identities=15,source_inverses=2,
    resources=resources,official_changed=False,goal_achieved=False,selected_next_numerical_study=None)
write(HERE/'summary.json',summary)
report='''# Complete positive-leaf catalog search

This study removes the earlier minimum leaf dimension of eight and includes
every positive classical leaf in the same two finite multiplication catalogs.
It preserves the earlier source, results and correctly scoped negative.

| Fixed catalog | 35 optimistic saving | 100 optimistic saving | 35 with leaf limit | 100 with leaf limit |
| --- | ---: | ---: | ---: | ---: |
'''
for f in families:
    a,b=f['records']
    report+=f"| {f['name']} | {100*a['optimistic_fraction']:.9f}% | {100*b['optimistic_fraction']:.9f}% | {100*a['leaf_storage_limited_fraction']:.9f}% | {100*b['leaf_storage_limited_fraction']:.9f}% |\n"
report+=f'''
The unchanged-raw uniform planning cut is **{100*summary['required_uniform_fraction']:.9f}%**
for 3e-9. The fixed gate requires BOTH current development cases to cross.
{gate_sentence} Changed per-call saving records:
{changed_calls}/1180; changed cost/memory/path entries: {changed_choices}/2360.
The exact per-network additional savings and chosen leaf histograms are
recorded in summary.json; audit_result.json retains every changed call.
The two original cold bills remain 413448696909 / 416264792412 F.

All catalog partitions have axes at least two, so each nontrivial child's
largest axis strictly decreases and is at most ceil(parent maximum/2).
All 586 nonempty current shapes have maximum axis at most 16384. Fourteen
stages therefore cover the entire declared positive-leaf family; the search
terminates by shape, without an arbitrary remaining-stage cutoff.

The memoized frontier retains every Pareto entry with simultaneous leaf
storage at most 1168900096 bytes, plus the unrestricted minimum-cost entry.
An above-limit footprint cannot become feasible under an ancestor's rank
multiplier and fringe maximum. It can affect only the unrestricted cost,
whose cheapest child is retained. This is not a total-workspace bound.

Independent replay certifies all 2360 chosen schedules and 1180 call records,
including exact integer costs, memory, totals and catalog monotonicity.
The unpruned exhaustive reference, full producer Pareto frontier and compact
frontier agree in 2224 fixed fixture/arena/subordinate-limit comparisons.
There are 864 synthetic ancestor checks, 15 exact rational Winograd identities,
4664 inherited classical-fringe checks, two complete source inverses and
per-call nonworsening against the old minimum-eight search. Fixtures include
positive dimension one, nonpowers of two and nontrivial cost/memory tradeoffs.
The exact current native dispatch source identifies
{summary['current_small_leaf_dispatch_witnesses']} current-census tile witnesses
with a leaf axis below eight; this is metadata replay, not a new native bill.

These are optimistic arithmetic envelopes. They omit replacement transforms'
full workspace, copying, dispatch, initialization, runtime and numerical
error. Savings favorably replace entire measured old calls. No candidate,
native runtime qualification, new score or official forecast is established.
Earlier Smirnov runtime failures and the Read6 development failure stand.
Uniform recursive children, the fixed catalogs and declared padding/fringes
define the scope; heterogeneous children, structural-zero simplifications,
other algorithms, streaming and combined accuracy/cost work remain outside it.

Three guarded groups retired; controller 46267 terminal. Search state counts:
{families[0]['states']} / {families[1]['states']}. Peak sampled RSS
{resources['peak_group_rss_bytes']} B; self-report
{resources['peak_self_reported_rss_bytes']} B; zero added swap. About
{artifact_bytes/1e6:.3f} MB artifacts before closeout. No estimator calls,
array decoding, reference reads, fits, new samples, downloads or GPU calls.

Official 331684 remains adjusted 3.5844144212297072e-9, raw
1.949715441540434e-8, 100/100 complete with zero failures. The 4e-9 crossing
is banked; 3/2/~1e-9 remain unmet. The unchanged-raw multiplier floor
1.949715441540434e-9 rules out cost alone reaching ~1e-9, but not 2e-9.
The original goal remains active and qualified uploads remain authorized.
'''
if negative:
    report+='\nThis closes these all-positive-leaf uniform-child catalogs as a standalone\nuniform-cost route to 3e-9 under the fixed two-case gate.\n'
else:
    report+='\nThe arithmetic opportunity needs a separately specified implementation and\nnative accuracy/runtime study before any candidate can qualify.\n'
with (HERE/'RESULT.md').open('x') as f:f.write(report)
external=[p for p in read(HERE/'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]
external+=bank_pins+[previous['study']]
external=list({p['path']:p for p in external}.values())
for p in external:check(p)
artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
write(HERE/'closeout.json',dict(completed=True,negative=negative,created_at_utc=datetime.now(timezone.utc).isoformat(),
    artifacts=artifacts,external_pins=external,resources=resources,candidate=None,
    official_changed=False,goal_achieved=False,selected_next_numerical_study=None))
print(dict(closeout=pin(HERE/'closeout.json'),artifacts=len(artifacts),external=len(external),resources=resources),flush=True)
