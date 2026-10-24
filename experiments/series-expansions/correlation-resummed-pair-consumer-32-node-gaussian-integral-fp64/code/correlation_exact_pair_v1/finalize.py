"""Metadata-only closeout for the fixed correlation-resummed consumer."""
import datetime
import json
import os
import shutil
import subprocess
from pathlib import Path
from common import HERE, D, PARENT, read, write, pin, sha, verify

ROOT = D.parents[4]


def main():
    seals = [verify(), verify('native_sources_sealed.json')]
    old = read(D / 'continuation_checkpoint_v54_verified.json')
    for key in ('checkpoint', 'handoff', 'index', 'closeout'):
        assert sha(old[key]['path']) == old[key]['sha256']
    previous_closeout = read(old['closeout']['path'])
    for key in ('artifacts', 'external_pins', 'own_source_pins'):
        for x in previous_closeout[key]:
            assert sha(x['path']) == x['sha256'], x['path']

    unit = read(HERE / 'unit.json')
    build = read(HERE / 'build.json')
    result = read(HERE / 'score.json')
    before = read(HERE / 'before_truth.json')
    assert unit['passed'] and not unit['truth_accessed']
    assert unit['max_probability_error'] <= 2e-11
    for key in ('max_moment_error', 'max_derivative_error', 'max_pair_error', 'max_fnp_error'):
        assert unit[key] <= 2e-10
    assert unit['domain_rejections'] == 4 and len(unit['fnp_tiny_bills']) == 4
    assert build['parent_byte_prefix_exact'] and build['post_centering_tail_exact']
    assert build['marginal_prefix_exact_except_domain_checks']
    for key in ('parent', 'candidate', 'null', 'gl32'):
        assert sha(build[key]['path']) == build[key]['sha256']
    assert Path(build['candidate']['path']).read_bytes().startswith(PARENT.read_bytes())
    assert Path(build['null']['path']).read_bytes().startswith(PARENT.read_bytes())
    assert result['completed'] and not result['passed']
    assert not result['fresh_generalization'] and not result['official_score']
    assert result['new_samples'] == 0 and not result['training']
    assert sha(result['before_truth']['path']) == result['before_truth']['sha256']
    assert before['ordinary_mean_arrays_not_yet_decoded']
    assert sha(before['source_seal']['path']) == before['source_seal']['sha256']

    all_records = []
    stages = {}
    for stage, ranks in [('null', [35, 100, 35]), ('dev', [35, 100, 35]), ('cold100', [100])]:
        receipt_path = HERE / stage / 'result.json'
        receipt = read(receipt_path)
        validation = read(HERE / (stage + '_validation.json'))
        assert validation['passed'] and not validation['truth_accessed']
        assert validation['repeat_or_warm_cold_exact']
        assert sha(receipt_path) == validation['result_sha256']
        assert not receipt['truth_accessed']
        assert receipt['candidate_sha256'] == build['null' if stage == 'null' else 'candidate']['sha256']
        assert [r['rank'] for r in receipt['records']] == ranks
        for i, r in enumerate(receipt['records']):
            assert sha(r['output']) == r['output_sha256']
            assert r['multiplier'] == max(.1, r['flops'] / 2**41)
            assert r['max_rss_bytes'] < 8_000_000_000
            assert r['includes_initialization'] == (i == 0) and r['input_flops_separate'] == 0
            assert r['weight5_cache']['completed'] and r['weight5_cache']['numeric_cache_retired']
            assert validation['proofs'][i]['first_row_exact']
            if stage == 'null':
                assert validation['proofs'][i]['all_parent_bytes_and_bill_exact']
            all_records.append(r)
        stages[stage] = receipt['records']
    assert stages['null'][0]['output_sha256'] == stages['null'][2]['output_sha256']
    assert stages['dev'][0]['output_sha256'] == stages['dev'][2]['output_sha256']
    assert stages['dev'][1]['output_sha256'] == stages['cold100'][0]['output_sha256']
    for pair, r in zip(before['pairs'], result['rows']):
        assert pair['rank'] == r['rank']
        for label in ('parent', 'candidate'):
            assert sha(pair[label]['output']) == pair[label]['output_sha256']
            assert pair[label]['flops'] == r[label]['flops']
            assert pair[label]['includes_initialization']
            assert r[label]['raw'] * r[label]['multiplier'] == r[label]['adjusted']
        for ratio, field in [('raw_ratio', 'raw'), ('cost_ratio', 'flops'), ('adjusted_ratio', 'adjusted')]:
            assert r['candidate'][field] / r['parent'][field] == r[ratio]
        assert r['adjusted_ratio'] > .90
    rows = result['rows']
    assert [r['rank'] for r in rows] == [35, 100]
    assert (rows[0]['candidate']['adjusted'] + rows[1]['candidate']['adjusted']) / (rows[0]['parent']['adjusted'] + rows[1]['parent']['adjusted']) == result['pooled_adjusted_ratio']

    guards = []
    for stage in ('unit', 'null', 'dev', 'cold100', 'score'):
        g = read(HERE / (stage + '_watch.json'))
        assert g['completed'] and g['whole_group_retired'] and g['exit_code'] == 0
        assert g['error'] is None and g['max_swap_growth_bytes'] == 0
        assert g['max_rss_bytes'] < 8_000_000_000
        guards.append(dict(file=stage + '_watch.json', **g))

    owned = []
    outside_count = 0
    proc = subprocess.run(['ps', '-Ao', 'pid,ppid,pgid,rss,etime,comm'], capture_output=True, text=True, check=True)
    for line in proc.stdout.splitlines()[1:]:
        fields = line.split(None, 5)
        if len(fields) != 6 or int(fields[0]) == os.getpid() or not any(s in fields[5].lower() for s in ('python', 'whest', 'tensor')):
            continue
        q = subprocess.run(['/usr/sbin/lsof', '-a', '-p', fields[0], '-d', 'cwd', '-Fn'], capture_output=True, text=True)
        dirs = [s[1:] for s in q.stdout.splitlines() if s.startswith('n/')]
        if dirs and not any(Path(x).is_relative_to(ROOT) for x in dirs):
            outside_count += 1
            continue
        alive = subprocess.run(['ps', '-p', fields[0], '-o', 'pid='], capture_output=True, text=True)
        if alive.stdout.strip():
            owned.append(dict(pid=int(fields[0]), executable=fields[5], cwd=dirs))
    assert not owned, owned

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    peak = max(unit['peak_rss_bytes'], result['peak_rss_bytes'], *(r['max_rss_bytes'] for r in all_records), *(g['max_rss_bytes'] for g in guards))
    elapsed = 0.0
    for g in guards:
        elapsed += g['elapsed_s']
    full_flops = sum(r['flops'] for r in all_records)
    resources = dict(created_utc=now, groups_completed=5, all_groups_retired=True,
        total_guard_elapsed_s=elapsed, peak_observed_rss_bytes=peak, max_swap_growth_bytes=0,
        full_estimator_calls=7, disabled_parent_calls=3, enabled_candidate_calls=4,
        metered_full_estimator_flops=full_flops, tiny_fnp_replays=4, tiny_fnp_flops=sum(unit['fnp_tiny_bills']),
        unit_reference_and_score_work_is_research=True, fitted_parameters=0, new_physical_samples=0,
        new_native_candidate=True, new_archive=False, new_upload=False,
        current_owned_numerical_processes=owned, unrelated_numerical_processes_count=outside_count,
        disk_free_GiB=shutil.disk_usage(D).free / 2**30, guards=guards)
    write(HERE / 'resources.json', resources)

    table = '\n'.join(f"| {r['rank']} | {r['raw_ratio']:.12f} | {r['cost_ratio']:.12f} | {r['adjusted_ratio']:.12f} |" for r in rows)
    report = f'''# Closed: fixed correlation-resummed pair consumer

The candidate passed mathematical and native implementation qualification,
then failed its fixed development accuracy/cost gate on both networks.

| Development rank | Raw MSE / Lazy | Own-cold FLOPs / Lazy | Adjusted / Lazy |
| --- | ---: | ---: | ---: |
{table}

Pooled adjusted ratio is {result['pooled_adjusted_ratio']:.16g}, about9.685% worse.
The prospective gate required <=.90 on each network. Even raw error rises
slightly, so added computation is not the sole reason for rejection. These
are reused ordinary1B development references, not NC14, fresh validation or
official scores. All candidate outputs and complete cold bills were fixed
and sealed before opening those reference means; parent scores match exactly.

This one candidate keeps Lazy's scalar marginal formulas, supplied cumulants,
CP/star/trace births, masks, rank384, age4, rotations, learned parameters and
final scalar head. It evaluates the same formal first-order L3+L4 pair map's
correlation dependence through conditional Gaussian boundary moments and a
fixed32-node Gaussian correlation integral, replacing Gaussian order6 and
cumulant correlation order2 truncations. Pair arithmetic also changes from
the fused FP32 series to FP64; the score difference cannot be attributed
exclusively to omitted correlation terms. The integral is a numerical
approximation, not symbolic exactness or a new positive joint-law model.

Independent small-fixture probability, physical conditional integration,
80-term Hermite derivative, assembled cumulant and actual FNP checks all
pass. Max probability error is {unit['max_probability_error']:.6g}; max scaled
moment error {unit['max_moment_error']:.6g}, derivative error
{unit['max_derivative_error']:.6g}, assembled pair error
{unit['max_pair_error']:.6g}, and FNP error {unit['max_fnp_error']:.6g}.
Four invalid-domain fixtures are rejected. Every actual call passes the
strict finite positive-variance and |rho|<=.85 domain checks, without clipping,
repair or fallback. The tested fixtures do not constitute a uniform proof
over every possible challenge network.

The disabled35/100/35 source reproduces all16 Lazy output rows and original
complete bills exactly. Enabled35/100/35 outputs are finite/nonnegative;
first returned rows remain byte-identical and repeated35 outputs are exact.
A separate cold100 output matches warm100 exactly. Cache completion and
retirement pass on all7 calls. Enabled cold bills are435,507,103,453 and
438,319,249,228 FLOPs, respectively. Native wall times are recorded; the
separate challenge-client/runtime contract was not run after this gate failed.

Five resource guards complete and retire in {elapsed:.6f}s, peak observed RSS
{peak:,} bytes, zero added swap. Seven full estimator calls cost
{full_flops:,} metered FLOPs; four tiny FNP replays cost
{resources['tiny_fnp_flops']:,}. Independent unit-reference/scoring work is
research. No fit, physical samples or GPU work. Disk free
{resources['disk_free_GiB']:.2f}GiB. The protocol's approximate0.8MB source-copy
estimate described archive scale; actual uncompressed candidate/null sources
are1,911,452/1,911,453 bytes. Total artifacts remain below30MB; no resource or
scientific threshold was relaxed.

This fixed consumer is closed: no order/window/coefficient/blend/refit/
precision/kernel rescue, wider panel, package or upload follows. It does not
rule out all correlation approximations or other cumulant representations.
Official Lazy#331599 remains adjusted3.8164492657795035e-9, raw
2.169264529783277e-8. The4e-9 bank is achieved;3/2/~1e-9 remain open.
The goal stays active and qualified future Phase2 submissions are preauthorized.
'''
    with (HERE / 'RESULT.md').open('x') as f:
        f.write(report)

    saved = {}
    backups = {}
    for key, name in [('handoff', 'handoff_before_checkpoint55.md'), ('index', 'index_before_checkpoint55.md')]:
        data = Path(old[key]['path']).read_bytes()
        saved[key] = data.decode()
        with (D / name).open('xb') as f:
            f.write(data)
        backups[key] = pin(D / name)
    artifacts = [pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
    assert sum(x['bytes'] for x in artifacts) < 30_000_000
    external_by_path = {}
    for seal in seals:
        for x in seal['pins']:
            if not Path(x['path']).is_relative_to(HERE):
                external_by_path[x['path']] = x
    external = [external_by_path[k] for k in sorted(external_by_path)]
    own = [pin(p) for p in sorted(HERE.glob('*.py'))]
    closeout = dict(completed=True, created_utc=now,
        status='Fixed correlation-resummed pair consumer qualified, then failed both adjusted development gates',
        artifacts=artifacts, external_pins=external, own_source_pins=own,
        artifact_bytes=sum(x['bytes'] for x in artifacts), result=pin(HERE / 'score.json'),
        raw_ratios=[r['raw_ratio'] for r in rows], cost_ratios=[r['cost_ratio'] for r in rows],
        adjusted_ratios=[r['adjusted_ratio'] for r in rows], pooled_adjusted_ratio=result['pooled_adjusted_ratio'],
        resources=resources, new_native_candidate=True, new_archive=False, new_upload=False,
        goal_achieved=False, next_numerical_study_selected=False)
    write(HERE / 'closeout.json', closeout)
    cpin = pin(HERE / 'closeout.json')

    heading = f'''## 2026-09-20 UTC — correlation-resummed pair consumer: qualified, adjusted error worsened

- Official Lazy#331599 remains adjusted3.8164492657795035e-9/raw2.169264529783277e-8.4e-9 bank achieved;3/2/~1e-9 open, goal active. Qualified future Phase2 uploads preauthorized; no approval, numerical worker or upload pending. Candidate below is closed, not pending.
- Checkpoint54, handoff/index and terminal covariance-supervision closeout artifacts/external/source pins reverified. That projection's negative result is unchanged. A bounded review found old Gaussian6/Edgeworth2 radial-Mehler experiments but no executed exact-correlation consumer in the inspected Astra sources; this is not an exhaustive novelty claim.
- New correlation_exact_pair_v1 builds one full-native candidate on unchanged Lazy prefix: same first-order L3+L4 cumulants, conditional Gaussian boundary moments, fixed GL32 integral streamed8, FP64 pair arithmetic, strict finite positive variance and |rho|<=.85. Scalar formulas/supplies/CP/star/trace/rank384/age4/masks/P/final head unchanged. No cumulant-order extension, fit, new physical samples, PSD repair, clipping or fallback.
- Independent probability/direct-moment/Hermite/assembled/FNP units pass (max scaled error9.13e-15), four domain rejections pass. Disabled35/100/35 all16 bytes and original bills exact. Enabled35/100/35 repeats, separate cold100 matches warm100, first output rows exact, all cache/domain/output checks pass. Seven full calls total{full_flops}F; four tiny FNP calls{resources['tiny_fnp_flops']}F.
- Sealed-before-truth own-cold ordinary1B dev raw ratios35/1001.001070729317/1.004728245689; cost1.094016528028/1.093357030007; adjusted1.095187923598/1.098526690670, pooled1.096850526741. Both fail fixed<=.90. FP64 arithmetic also changed, so no attribution solely to correlation truncation. No NC14/confirm/client/contract/package/upload; no order/window/coefficient/blend/refit/precision/kernel rescue.
- Five guards complete and retired{elapsed:.6f}s,peak{peak}B,zero added swap; about{closeout['artifact_bytes']/1e6:.2f}MB artifacts, disk{resources['disk_free_GiB']:.2f}GiB free. RESULT.md/score.json/resources.json/closeout{cpin['sha256']}; checkpoint55 supersedes54. Next independent study unselected; goal active and unachieved.

'''
    for key in ('handoff', 'index'):
        Path(old[key]['path']).write_text(heading + saved[key])
    cp = read(old['checkpoint']['path'])
    classification = 'numerical progress: fixed correlation-resummed pair consumer qualified then failed both own-cold adjusted development gates'
    cp.update(updated_at_utc=now, goal_status='active', goal_achieved=False,
        checkpoint_version=55, supersedes='continuation_checkpoint_v54.json',
        current_handoff_sha256=sha(ROOT / 'HANDOFF_CURRENT.md'),
        reexamination_index_sha256=sha(D / 'REEXAMINATION_INDEX_20260919.md'),
        source_review_index_sha256=sha(D / 'REEXAMINATION_INDEX_20260919.md'),
        numerical_workers_active=False, upload_worker_active=False, active_numerical_session=None,
        approval_pending=False, pending_candidate=None, next_numerical_study_selected=False,
        prior_checkpoint='continuation_checkpoint_v54.json', prior_checkpoint_sha256=old['checkpoint']['sha256'],
        previous_checkpoint=old['checkpoint'],
        previous_goal_turn_classification=cp['current_goal_turn_classification'],
        current_goal_turn_classification=classification, this_goal_turn_classification=classification,
        this_turn_resources=resources,
        local_work='Built and qualified one current-Lazy correlation-resummed pair candidate; own-cold adjusted dev error worsened on both fixed networks. Consumer closed, no package/upload; next independent study not selected.',
        preserved_previous_handoff_and_index=backups)
    cp['correlation_exact_pair'] = dict(status=closeout['status'], closeout=cpin,
        raw_ratios=closeout['raw_ratios'], adjusted_ratios=closeout['adjusted_ratios'],
        pooled_adjusted_ratio=result['pooled_adjusted_ratio'], resources=resources)
    write(D / 'continuation_checkpoint_v55.json', cp)
    for x in artifacts + external + own:
        assert sha(x['path']) == x['sha256'], x['path']
    verified = dict(checkpoint=pin(D / 'continuation_checkpoint_v55.json'),
        handoff=pin(ROOT / 'HANDOFF_CURRENT.md'), index=pin(D / 'REEXAMINATION_INDEX_20260919.md'),
        closeout=cpin, all_study_pins_verified=True, scalar_gates_and_replay_roster_verified=True,
        numerical_workers_active=False)
    write(D / 'continuation_checkpoint_v55_verified.json', verified)
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    main()
