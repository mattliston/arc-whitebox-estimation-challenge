"""Reconcile the scalar receipts and preserve the fixed negative diagnostic."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

D = Path(__file__).resolve().parent
H = D.parent


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def once(p, value):
    with p.open('x') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False) + '\n')


read = lambda p: json.loads(p.read_text())
result = read(D / 'result.json')
guard = read(D / 'diagnostic_guard.json')
assert not result['passes_source_gate']
assert guard['completed'] and guard['whole_group_retired'] and not guard['remaining_rows']
assert guard['exit_code'] == 0 and guard['max_swap_growth_bytes'] == 0
assert result['qualification']['passed']
assert not result['fits_run'] and not result['changed_candidate_built']
pins = read(D / 'pre_array_source_seal.json')['files']
for name, expected in pins.items():
    assert sha(Path(name)) == expected, name
audits = []
for row in result['networks']:
    rank = row['rank']
    layers = read(D / f'layers_{rank}.json')
    assert [r['layer'] for r in layers] == list(range(2, 15))
    assert row['half_counts'] == [16009216, 15990784]
    assert sum(row['half_counts']) == 32000000
    for key in ['offdiagonal_covariance', 'weighted_next_variance']:
        for layer in layers:
            s = layer[key]
            expected = math.fsum([s['base_cross'], 2 * s['error_dot_delta'], s['delta_energy']])
            assert math.isclose(s['changed_cross'], expected, rel_tol=1e-13, abs_tol=1e-20)
        base = math.fsum(r[key]['base_cross'] for r in layers)
        changed = math.fsum(r[key]['changed_cross'] for r in layers)
        assert math.isclose(row[key]['base_cross'], base, rel_tol=1e-14)
        assert math.isclose(row[key]['changed_cross'], changed, rel_tol=1e-14)
        assert math.isclose(row[key]['ratio'], changed / base, rel_tol=1e-14)
        assert row[key]['improved_layers'] == sum(r[key]['changed_cross'] < r[key]['base_cross'] for r in layers)
        assert row[key]['all_base_energies_positive']
    assert not row['gates']['covariance_reduction'] and not row['gates']['weighted_variance_reduction']
    assert not row['gates']['weighted_variance_layer_wins']
    pred = D / f'pred_{rank}.npy'
    control = H / 'both_tail_train2_v6' / pred.name
    assert pred.read_bytes() == control.read_bytes()
    old = [json.loads(line) for line in (control.parent / 'rows.jsonl').read_text().splitlines()]
    assert row['flops'] == next(r['flops'] for r in old if r['rank'] == rank)
    audits.append(dict(rank=rank, scalar_statistics_reconciled=True,
                       full_prediction_file_bytes_exact=True, complete_bill_exact=True,
                       prediction_sha256=sha(pred)))
asof = datetime.now(timezone.utc).isoformat()
once(D / 'receipt_audit.json', dict(asof_utc=asof, source_seal_verified=True, records=audits,
    scope='Independent scalar receipt reconciliation and file parity, not a second native covariance capture'))
rows = result['networks']
table = '\n'.join(f"| {r['rank']} | {r['offdiagonal_covariance']['change_pct']:+.6f}% | "
                  f"{r['weighted_next_variance']['change_pct']:+.6f}% | "
                  f"{r['weighted_next_variance']['improved_layers']}/13 |" for r in rows)
once(D / 'RESULT.md', f'''# Fixed quadratic covariance-source diagnostic: closed

The source gate fails on both unchanged S31 TRAIN networks. Adding the
previous fixed increment at coefficient1 worsens both pooled error measures:

| TRAIN rank | Off-diagonal covariance error energy | Weighted next-variance error energy | Weighted layer wins |
| --- | ---: | ---: | ---: |
{table}

Every base cross-half energy is positive. The prospective requirements were
at least2% reduction in both pooled measures on both networks, and10/13
weighted layer wins. No sign, scale, subset or refit is selected. This closes
the covariance-alignment rationale for retraining this already failed
candidate. The earlier broader matched-fit failure is also unchanged.

The diagnostic uses existing independent even/odd-batch P32M covariance
halves with exact half sizes16,009,216 and15,990,784 and unbiased denominator
corrections. It observes the actual post-head carried covariance; its delta
never enters inference. Cross-half energies remove independent sampling
variance in exact arithmetic, not shared FP32/MPS bias. These are two reused
TRAIN networks, not independent validation or a population impossibility
claim. No final-mean labels were opened.

All16 prediction files and full operation bills exactly match independent
S31 controls. Four statistic/congruence fixtures pass; source seals and
scalar receipts reconcile. One guard completed and retired in
{guard['elapsed_s']:.2f}s, peak RSS{max(guard['max_rss_bytes'], result['peak_rss_bytes']) / 1e9:.3f}GB,
zero added swap. Exec session75412 was reaped with exit0.

No candidate, training, VALID40, sampling, package or upload follows.
Official S31 #332517 remains2.1736465056908115e-9 adjusted. This is a
negative research result and supplies no banked score improvement.
''')
files = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(D.iterdir()) if p.is_file()}
once(D / 'artifact_manifest.json', dict(asof_utc=asof, files=files,
    total_bytes=sum(r['bytes'] for r in files.values()), scope='Files before manifest and closeout'))
once(D / 'closeout.json', dict(asof_utc=asof, status='closed_source_alignment_failed',
    source_gate_passed=False, diagnostic_group_retired=True, added_swap_bytes=0,
    result_sha256=sha(D / 'result.json'), receipt_audit_sha256=sha(D / 'receipt_audit.json'),
    manifest_sha256=sha(D / 'artifact_manifest.json'), no_candidate_or_refit=True,
    goal='Active, 2e-9 and approximately1e-9 unmet'))
print(json.dumps(dict(status='closed_source_alignment_failed', ranks=[r['rank'] for r in rows],
                     source_files_verified=len(pins), manifest_files=len(files))))
