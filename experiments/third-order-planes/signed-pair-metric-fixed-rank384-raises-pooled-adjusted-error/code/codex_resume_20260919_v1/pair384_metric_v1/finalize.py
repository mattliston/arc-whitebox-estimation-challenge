"""Preserve the fixed metric-only development failure and its resource evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
D = HERE.parent
A = D.parent / 'astra_phase2_reexamination_v1'
ROOT = Path('/Users/matt/arc-white-box-estimation')
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    seal = read(HERE / 'source_sealed.json')
    for name, expected in seal['files_sha256'].items():
        assert sha(HERE / name) == expected, name
    assert sha(D / seal['native_runner']) == seal['native_runner_sha256']
    for name, expected in read(HERE / 'source_review.json')['sources_sha256'].items():
        assert sha(A / name) == expected, name
    build = read(HERE / 'build.json')
    assert build['whole_source_inverse_byte_exact'] and build['rank'] == 384
    assert build['exact_metric_edits'] == 5 and build['omitted_rank_edits'] == 3
    assert build['parent_sha256'] == sha(D / 'lazy_rotation_objective_v1/candidate.py')
    unit = read(HERE / 'unit_v1/result.json')
    assert unit['passed'] and unit['repeated_arrays_byte_exact'] and unit['repeated_bills_exact']
    watches = [read(HERE / name) for name in
               ['unit_watch.json', 'dev_watch.json', 'cold100_watch.json', 'score_watch.json']]
    for w in watches:
        assert w['completed'] and w['exit_code'] == 0 and w['error'] is None
        assert w['whole_group_retired'] and w['max_swap_growth_bytes'] == 0
        assert w['max_rss_bytes'] < 8_000_000_000
    calls = []
    for name in ['dev', 'cold100']:
        result = read(HERE / name / 'result.json')
        assert result['candidate_sha256'] == build['candidate_sha256']
        assert result['runner_sha256'] == seal['native_runner_sha256']
        assert not result['truth_accessed']
        for record in result['records']:
            assert sha(record['output']) == record['output_sha256']
            assert record['weight5_cache']['completed'] and record['weight5_cache']['numeric_cache_retired']
            calls.append(record)
    assert len(calls) == 4
    assert calls[0]['output_sha256'] == calls[2]['output_sha256']
    assert calls[1]['output_sha256'] == calls[3]['output_sha256']
    score = read(HERE / 'dev_score.json')
    assert not score['development_gate_passed']
    assert score['repeat_byte_exact'] and score['warm_cold100_byte_exact'] and score['first_five_layers_byte_exact']
    assert [r['rank'] for r in score['rows']] == [35, 100]
    assert score['pooled_adjusted_ratio'] == sum(r['candidate']['adjusted'] for r in score['rows']) / sum(r['parent']['adjusted'] for r in score['rows'])
    peak = max([w['max_rss_bytes'] for w in watches] + [r['max_rss_bytes'] for r in calls] + [unit['max_rss_bytes']])
    assert peak < 8_000_000_000
    resources = dict(full_prediction_calls=4, guarded_stages=4,
                     billed_prediction_flops=sum(r['flops'] for r in calls),
                     peak_rss_bytes=peak, additional_swap_bytes=0, all_owned_groups_retired=True,
                     total_guarded_wall_s=sum(w['elapsed_s'] for w in watches))
    write_json(HERE / 'resources.json', resources)
    bank = read(D / 'official_best_331599.json')
    assert sha(D / 'lazy_rotation_objective_v1/submission-lazy-objective-20260919.tar.gz') == bank['archive_sha256']
    text = [
        '# Fixed rank384 signed-pair metric: development failure', '',
        'The entry-metric-only variant failed its prospective actual-objective gate.',
        'Rank384, age4, trained heads, physical factor coordinates, Lazy schedule,',
        'all original consumers and kernels were held fixed. This supplies the',
        'previously absent same-rank comparison; the original PAIR256 joint',
        'rank-and-metric failure remains unchanged.', '',
        '| Development rank | Lazy raw MSE | Pair384 raw MSE | Raw ratio | Own-cold adjusted ratio |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for row in score['rows']:
        text.append(f"| {row['rank']} | {row['parent']['raw']:.16g} | {row['candidate']['raw']:.16g} | {row['raw_ratio']:.12g} | {row['adjusted_ratio']:.12g} |")
    text += ['', f"Pooled adjusted ratio: **{score['pooled_adjusted_ratio']:.16g}**.",
             'Rank35 improves slightly; rank100 worsens. Both raw and adjusted',
             'improvement on both networks and pooled adjusted<=.99 were required.',
             'These are two reused ordinary1B development networks, not an',
             'independent panel, official result, or universal metric-capacity test.', '']
    for row in score['rows']:
        increment = row['candidate']['flops'] - row['parent']['flops']
        text.append(f"Rank{row['rank']} complete cold FLOPs: {row['parent']['flops']} -> {row['candidate']['flops']} (+{increment}, {100*increment/row['parent']['flops']:.6f}%).")
    text += ['',
             'Independent explicit small third tensors certify the Gram and range',
             'action with maximum relative error1.5600663146273522e-7. Signed,',
             'near-coalesced, coalesced and zero-coefficient cases, original physical',
             'projection bytes, exact repeat bills and four input rejections pass.',
             'Full outputs are finite/nonnegative; first5 parent layers, own repeat,',
             'warm/cold100 output bytes and cache retirement pass. Changed later',
             'outputs use their own actual complete bills. All predictions and bills',
             'were sealed before the two original truth files were opened.', '',
             f"Four full calls total {resources['billed_prediction_flops']} billed prediction FLOPs.",
             f'Peak RSS including self-reports: {peak} bytes. Zero added swap.',
             'All four supervised stages exit0 and retire their process groups.',
             'No NumPy-free client, larger panel, fit, metric refresh, scaling/blending,',
             'rank/age/precision/QR sweep, package or submission follows this failure.', '',
             'The entry-frozen per-pair Gram omits cross-pair tensor terms and is not',
             'refreshed after transport. The result closes this exact fixed variant,',
             'not every cancellation-aware or endpoint-aware representation.', '',
             f"Official bank remains Lazy #331599 at {bank['adjusted']:.16g} adjusted.",
             'The4e-9 milestone is confirmed. Approximately1e-9 remains active and unmet.', '']
    with (HERE / 'NEGATIVE_RESULT.md').open('x') as stream:
        stream.write('\n'.join(text))
    artifacts = {str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
                 if p.is_file() and '__pycache__' not in p.parts}
    size = sum((HERE/name).stat().st_size for name in artifacts)
    assert size < 10_000_000
    closeout = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    status='closed negative at fixed same-rank development gate',
                    goal_status='active', goal_achieved=False, official_bank=bank,
                    candidate_sha256=build['candidate_sha256'], parent_sha256=build['parent_sha256'],
                    source_inverse_byte_exact=True, rank=384, age=4, exact_metric_edits=5,
                    unit_passed=True, development_gate_passed=False,
                    pooled_adjusted_ratio=score['pooled_adjusted_ratio'],
                    raw_ratios=[r['raw_ratio'] for r in score['rows']],
                    adjusted_ratios=[r['adjusted_ratio'] for r in score['rows']],
                    resources=resources, artifact_bytes_before_closeout=size,
                    artifacts_sha256=artifacts, remaining_disk_GiB=shutil.disk_usage(ROOT).free/2**30)
    write_json(HERE / 'closeout.json', closeout)
    print(json.dumps({k:v for k,v in closeout.items() if k not in ['artifacts_sha256','official_bank']}))


if __name__ == '__main__':
    main()
