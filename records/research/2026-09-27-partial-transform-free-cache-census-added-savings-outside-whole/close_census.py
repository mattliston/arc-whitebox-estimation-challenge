"""Reconcile operation receipts, prior trace, and preserved predictions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

D = Path(__file__).resolve().parent
H = D.parent
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


r = read(D / 'result.json')
g = read(D / 'census_v2_guard.json')
failed = read(D / 'census_guard.json')
assert not r['passes_ceiling_gate'] and r['qualification']['passed']
assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows']
assert g['exit_code'] == 0 and g['max_swap_growth_bytes'] == 0
assert not failed['completed'] and failed['whole_group_retired']
assert not failed['remaining_rows'] and failed['max_swap_growth_bytes'] == 0
for seal in ['launch_seal.json', 'launch_seal_v2.json']:
    for p, digest in read(D / seal)['files'].items():
        assert sha(Path(p)) == digest, p
previous = read(H / 'factor_transform_reuse_v1/result.json')
rows = []
for row in r['networks']:
    rank = row['rank']
    operations = read(D / f'operations_{rank}.json')
    transforms = read(D / f'transforms_{rank}.json')
    old = read(H / 'factor_transform_reuse_v1' / f'transforms_{rank}.json')
    assert len(transforms) == len(old) == row['transforms']
    assert len(operations) == row['operations']
    assert [o['index'] for o in operations] == list(range(len(operations)))
    for key, opkey in [('optimistic_removable_flops', 'removable_arithmetic'),
                       ('total_chunks', 'chunks'), ('repeated_chunks', 'repeated_chunks'),
                       ('observed_transform_arithmetic', 'paid')]:
        assert sum(o[opkey] for o in operations) == row[key]
    assert sum(t['flops'] for t in transforms) == row['input_transform_flops']
    assert 100 * row['optimistic_removable_flops'] / row['flops'] == row['optimistic_saving_pct']
    groups = {t['index']: [] for t in transforms}
    for o in operations:
        assert 0 <= o['removable_arithmetic'] <= o['paid']
        groups[o['transform']].append(o)
    seen = set()
    extra = 0
    for t, before in zip(transforms, old):
        assert all(t[k] == before[k] for k in ['side', 'depth', 'dims', 'flops'])
        ops = groups[t['index']]
        assert sum(o['paid'] for o in ops) == t['arithmetic_flops']
        assert sum(o['removable_arithmetic'] for o in ops) == t['removable_arithmetic']
        key = json.dumps([before['side'], before['depth'], before['dims'], before['input']], sort_keys=True)
        if key not in seen:
            extra += t['removable_arithmetic']
        seen.add(key)
    prior = next(n for n in previous['networks'] if n['rank'] == rank)
    assert prior['flops'] == row['flops']
    union = prior['optimistic_removable_flops'] + extra
    pred = D / f'pred_{rank}.npy'
    control = H / 'both_tail_train2_v6' / pred.name
    assert pred.read_bytes() == control.read_bytes()
    rows.append(dict(rank=rank, operations_reconciled=True, full_prediction_file_bytes_exact=True,
        complete_bill_exact=True, exact_prior_transform_trace=True,
        partial_saving_on_first_whole_input_occurrences=extra,
        additional_to_whole_cache_pct=100 * extra / row['flops'],
        free_union_ceiling_flops=union, free_union_ceiling_pct=100 * union / row['flops'],
        union_scope='Descriptive union without double counting; unchanged original partial-only gate',
        prediction_sha256=sha(pred)))
asof = datetime.now(timezone.utc).isoformat()
once(D / 'receipt_audit.json', dict(asof_utc=asof, source_seals_verified=True, records=rows,
    scope='Independent scalar operation/transform reconciliation and prediction byte checks; full operand hashes were not persisted'))
table = '\n'.join(f"| {n['rank']} | {n['optimistic_saving_pct']:.6f}% | {a['additional_to_whole_cache_pct']:.6f}% | {a['free_union_ceiling_pct']:.6f}% |"
                  for n, a in zip(r['networks'], rows))
peak = max(g['max_rss_bytes'], r['peak_rss_bytes'])
once(D / 'RESULT.md', f'''# Partial arithmetic-block reuse: insufficient saving

The fixed4096-scalar arithmetic-block cache fails the prospective1.5%
complete-bill ceiling gate on both unchanged S31 TRAIN networks.

| TRAIN rank | Free arithmetic-block reuse | Added saving outside whole-input repeats | Union with earlier free whole-input cache |
| --- | ---: | ---: | ---: |
{table}

The union is a descriptive, optimistic combination without double counting;
it does not change either failed gate. All lookup, copy, storage and lifetime
costs were set to zero. Neither number is a deployable saving. No cache
implementation follows, and no block-size or other parameter is selected.
The retained thin-star component remains below the submission bar.

The test observes only paid additions/subtractions inside the two existing
global A/B input-transform bindings. Exact repeated ordered operand blocks
produce identical output hashes. Existing free resident reuse contributes
no arithmetic. This closes this fixed partition's cache rationale, not all
partitions, scalar common-subexpressions, output transforms or new algebra.

All16 prediction files and complete bills equal the independent controls;
both transform sequences, shapes and individual transform bills exactly
match the prior whole-input census. Twelve scalar-byte fixtures, exhaustive
tiny duplicate grouping, signed-zero/operation distinction and an in-place
case pass. Reconciled operation sums agree with all network/transform receipts.
The catalog itself was held in memory; full operand/hash records were not
persisted, so the receipt audit is not an independent full-data regrouping.

The first attempt failed only its synthetic strided-buffer qualification,
before native inputs. Its version and guard remain preserved; REPAIR.md
explains the exact-block repair. The repaired guard completed and retired
in{g['elapsed_s']:.2f}s, peak RSS{peak / 1e9:.3f}GB, zero added swap.
Exec session69990 was reaped with exit0. No targets, training, VALID40,
new samples, changed estimator, package or upload were used.

S31 #332517 remains2.1736465056908115e-9 adjusted. This test provides no
banked gain. Standing submission authorization and the overall goal persist.
''')
files = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(D.iterdir()) if p.is_file()}
once(D / 'artifact_manifest.json', dict(asof_utc=asof, files=files,
    total_bytes=sum(v['bytes'] for v in files.values()), scope='Files before manifest and closeout'))
once(D / 'closeout.json', dict(asof_utc=asof, status='closed_insufficient_partial_reuse',
    all_diagnostic_groups_retired=True, added_swap_bytes=0,
    result_sha256=sha(D / 'result.json'), receipt_audit_sha256=sha(D / 'receipt_audit.json'),
    manifest_sha256=sha(D / 'artifact_manifest.json'), candidate_built=False,
    goal='Active; 2e-9 and approximately1e-9 unmet'))
print(json.dumps(dict(status='closed_insufficient_partial_reuse', networks=rows, peak_rss_bytes=peak)))
