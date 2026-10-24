"""Verify and preserve the completed finite-family search; no new calculation."""
from datetime import datetime, timezone
import ast
import os
import shutil
import subprocess
from common import *

seal_check()
for p in read(HERE / 'finalize_sources_sealed.json')['pins']:
    check(p)
assert read(HERE / 'completed.json')['completed']
result = read(HERE / 'reprice_result.json')
audit = read(HERE / 'audit_result.json')
paths = read(HERE / 'paths_verified.json')
assert result['completed'] and audit['passed'] and paths['passed']
assert audit['maximum_depth_seven_complete_for_specified_family']
assert audit['all_old_call_optima_nonworsening'] and not audit['changed_calls']
assert all(r['additional_optimistic_saved'] == r['additional_leaf_limited_saved'] == 0
           for r in audit['comparison'])
assert all(not f['passes_necessary_3e9_gate'] and not f['leaf_storage_limit_passes_necessary_3e9_gate']
           for f in result['families'])
assert result['new_model_calls'] == result['truth_arrays_read'] == result['fits'] == 0
assert result['candidate'] is False and result['native_runtime_qualification'] is False
assert paths['result_sha256'] == sha(HERE / 'reprice_result.json')
for name in ['producer', 'result', 'path_replay', 'old_result']:
    check(audit[name])
old_result = read(Path(audit['old_result']['path']))
old_families = {f['name']: f for f in old_result['families']}
new_families = {f['name']: f for f in result['families']}
assert len(old_families) == len(old_result['families']) == 2
assert len(new_families) == len(result['families']) == 2
assert old_families.keys() == new_families.keys()
chosen_entries_checked = call_records_checked = 0
for name, family in new_families.items():
    old_records = {r['rank']: r for r in old_families[name]['records']}
    new_records = {r['rank']: r for r in family['records']}
    assert len(old_records) == len(old_families[name]['records']) == 2
    assert len(new_records) == len(family['records']) == 2
    assert old_records.keys() == new_records.keys()
    for rank, record in new_records.items():
        old_details = {d['index']: d for d in old_records[rank]['details']}
        new_details = {d['index']: d for d in record['details']}
        assert len(old_details) == len(old_records[rank]['details']) == 295
        assert len(new_details) == len(record['details']) == 295
        assert old_details.keys() == new_details.keys()
        for index, detail in new_details.items():
            before = old_details[index]
            for field in ['shape', 'caller', 'actual_call_flops']:
                assert before[field] == detail[field], (name, rank, index, field)
            for choice in ['optimistic_best', 'leaf_storage_limited_best']:
                assert before[choice] == detail[choice], (name, rank, index, choice)
                chosen_entries_checked += 1
            call_records_checked += 1
assert (call_records_checked, chosen_entries_checked) == (1180, 2360)
chosen_certificate = dict(passed=True, old_result=audit['old_result'],
    new_result=pin(HERE / 'reprice_result.json'), call_records=call_records_checked,
    chosen_entries=chosen_entries_checked, all_cost_memory_path_entries_identical=True,
    call_identity_fields=['family_name', 'rank', 'index', 'shape', 'caller', 'actual_call_flops'])
build = read(HERE / 'build.json')
for inverse in build['source_inverses']:
    check(inverse['old']); check(inverse['new'])
    recovered = Path(inverse['new']['path']).read_text()
    for before, after in reversed(inverse['edits']):
        assert recovered.count(after) == 1
        recovered = recovered.replace(after, before, 1)
    assert recovered == Path(inverse['old']['path']).read_text()
check(build['old_study']); check(build['previous_checkpoint'])
names = ['reprice', 'verify_paths', 'audit']
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
bank = read(D / 'official_best_331684.json')
bank_pins = [v for v in bank.values() if isinstance(v, dict) and {'path', 'sha256', 'bytes'} <= v.keys()]
for p in bank_pins:
    check(p)
previous = read(D / 'continuation_checkpoint_v76.json')
check(previous['study']); check(previous['official_bank'])
write(HERE / 'chosen_entries_verified.json', chosen_certificate)
artifact_bytes = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
assert artifact_bytes < 20_000_000
resources = dict(guarded_groups=len(guards), all_groups_retired=True,
    controller_terminal_session=37572, full_native_calls=0, new_native_flops=0,
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_self_reported_rss_bytes=max(r['peak_rss_bytes'] for r in [result, audit, paths]),
    max_swap_growth_bytes=0, reference_arrays_decoded=0, new_samples=0,
    fits=0, new_downloads=0, GPU_calls=0, metadata_prepare_failures=1,
    artifact_bytes_before_closeout=artifact_bytes, disk_free_bytes=shutil.disk_usage(HERE).free)
write(HERE / 'resources.json', resources)
families = [dict(name=f['name'], states=f['states'],
    passes_necessary_3e9_gate=f['passes_necessary_3e9_gate'],
    leaf_storage_limit_passes_necessary_3e9_gate=f['leaf_storage_limit_passes_necessary_3e9_gate'],
    records=[{k:v for k,v in r.items() if k != 'details'} for r in f['records']]) for f in result['families']]
summary = dict(completed=True, negative=True, candidate=False,
    scope='Complete uniform-recursion depth/padding search for the two fixed catalogs, minimum leaf dimension8, classical core fringes; not all matrix algorithms or streaming schedules.',
    maximum_stages=7, child_minimum_dimension=8, removed_parent_dimension_restriction=True,
    depth_bound_verified=True, families=families,
    required_uniform_fraction=result['required_uniform_fraction_to_cross_3e9_at_unchanged_raw'],
    changed_current_call_optima=0, all_chosen_entries_identical=True,
    chosen_entries_verification=pin(HERE / 'chosen_entries_verified.json'),
    comparison=audit['comparison'],
    verified_schedules=paths['schedules_replayed'], verified_call_records=paths['call_records_reconciled'],
    exhaustive_small_checks=len(audit['exhaustive_small_checks']),
    small_padding_witnesses=audit['admitted_small_parent_padding_witnesses'],
    source_inverses=2, resources=resources, official_changed=False, goal_achieved=False,
    selected_next_numerical_study=None)
write(HERE / 'summary.json', summary)
report = '''# Complete finite catalog depth/padding search: no further saving

Completing the search from five to seven stages and admitting ceil-padded
children of width8 from parent dimensions below16 changes **none** of the
current-call cost optima. The previous five-stage search was narrower than
an unrestricted minimum-leaf-eight search. Its numerical optima survive
this new completion, with the original outputs preserved.

| Fixed catalog | 35 optimistic saving | 100 optimistic saving | 35 with leaf limit | 100 with leaf limit |
| --- | ---: | ---: | ---: | ---: |
'''
for f in families:
    a, b = f['records']
    report += f"| {f['name']} | {100*a['optimistic_fraction']:.6f}% | {100*b['optimistic_fraction']:.6f}% | {100*a['leaf_storage_limited_fraction']:.6f}% | {100*b['leaf_storage_limited_fraction']:.6f}% |\n"
report += f'''
Both catalogs still miss the **{100*summary['required_uniform_fraction']:.9f}%**
unchanged-raw uniform planning cut for3e-9 on BOTH current development cases.
Additional saving from the new search is exactly zero for every one of the
1180 call records (590 calls in each catalog). No deeper or newly admitted
small-parent padded path improves an actual current-call optimum. This is
an optimistic arithmetic screen, not a measured estimator cost or score.
The original cold bills remain413448696909 /416264792412 F.
An additional exact comparison verifies that all 2360 chosen entries have
identical cost, leaf memory and schedule path, with each call matched by
catalog, network, index, shape, caller and actual measured call bill.

The structural depth argument is explicit: all catalog partition dimensions
are at least2 and each current outer product has a minimum dimension at
most1024. Even with ceiling, eight stages make the minimum dimension at
most4, below the fixed8 leaf minimum. Therefore seven stages cover every
admissible path in this specified family. The family still chooses one
uniform recursive child program at each node, with the declared ceil or
floor-core/classical-fringe convention. It does not include arbitrary
algorithms, heterogeneous child schedules, alternative leaves or streaming.

Independent replay checks all{paths['schedules_replayed']} selected schedules,
their exact integer costs and simultaneous leaf footprints, and all
{paths['call_records_reconciled']} call records. An independent exhaustive
enumeration of all up-to-two-stage schedules agrees with the producer on
64 fixed fixture/leaf-limit comparisons. Eight rational Winograd identities
cover depths0..7. The old4664 classical-fringe checks also pass. Every old
per-call optimum is nonworsening and catalog-superset monotonicity holds.
The newly admitted parent15 case is not a vacuous code change: for shape
64x15x256 a padded Winograd option costs467712 versus475136 classical
operations. This tiny witness is not a current estimator improvement.

The1168900096-byte limit counts simultaneous leaf A/B/C storage only.
Transforms, full workspace, copies, dispatch, rounding and actual runtime
are not qualified; costs also favorably allow replacing measured old call
initialization. Earlier Smirnov streaming/global-batch runtime failures and
Read6's actual development gate failure remain intact. No implementation,
native qualification, candidate, package or upload follows automatically.

Preparation attempt1 stopped before numerical work because a strict pin
dictionary comparison rejected extra rank/call-count fields on two old
census records. Their path, SHA256 and byte counts matched exactly. The
original source, failure record and already written drafts are preserved.
Additive prepare_v2 normalizes only file-identity fields and verifies/adopts
those identical drafts. Producer formulas, inputs, protocol and gates did
not change. Two whole-source inverses independently verify the declared
producer and path-verifier changes against their preserved originals.

Three guarded groups retired; controller37572 terminal. Repricing visited
{families[0]['states']} /{families[1]['states']} cached states in the two catalogs,
using stdlib integer/rational arithmetic only. Peak sampled RSS
{resources['peak_group_rss_bytes']} B, self-report
{resources['peak_self_reported_rss_bytes']} B, zero added swap. About
{artifact_bytes/1e6:.3f} MB artifacts before closeout. No estimator calls,
target/network array decoding, fits, new samples, downloads or GPU.

This closes the specified completed finite family as a standalone3e-9
uniform-cost route, not smaller/composed improvements or all algorithms.
Official331684 stays adjusted3.5844144212297072e-9,100/100 complete,zero
failures.4e-9 is banked;3/2/~1e-9 remain unmet. The unchanged-raw floor
1.949715441540434e-9 prevents cost alone reaching~1e-9, but not2e-9.
The original goal remains active.
'''
with (HERE / 'RESULT.md').open('x') as f:
    f.write(report)
external = [p for p in read(HERE / 'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]
external += bank_pins + [previous['study']]
external = list({p['path']:p for p in external}.values())
for p in external:
    check(p)
artifacts = [pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
write(HERE / 'closeout.json', dict(completed=True, negative=True,
    created_at_utc=datetime.now(timezone.utc).isoformat(), artifacts=artifacts,
    external_pins=external, resources=resources, candidate=None,
    official_changed=False, goal_achieved=False, selected_next_numerical_study=None))
print(dict(closeout=pin(HERE / 'closeout.json'), artifacts=len(artifacts),
    external=len(external), resources=resources), flush=True)
