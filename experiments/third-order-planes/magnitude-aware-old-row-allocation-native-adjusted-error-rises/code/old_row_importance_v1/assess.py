"""Independent offline score reconciliation; no further inference or fitting."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

D = Path(__file__).resolve().parent
H = D.parent
P2 = H.parent
OUT = H / 'old_row_importance_train2_v1'
BASE = H / 'both_tail_train2_v6'
read = lambda p: json.loads(p.read_text())
rows = lambda p: [json.loads(s) for s in (p / 'rows.jsonl').read_text().splitlines()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


for name, expected in read(D / 'pre_endpoint_seal.json')['files'].items():
    assert sha(Path(name)) == expected, name
assert read(D / 'qualification.json')['passed']
candidate_rows, base_rows = rows(OUT), rows(BASE)
assert [r['rank'] for r in candidate_rows] == [35, 100]
assert [r['rank'] for r in base_rows] == [35, 100]
assert read(OUT / 'manifest.json')['sha256'] == sha(D / 'candidate.py')
assert (D / 'candidate.py').read_bytes().startswith((H / 'candidate_old_both_tail_trace_bounded_v6.py').read_bytes())
for filename in ['qualification_guard.json', 'train2_guard.json']:
    g = read(D / filename)
    assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows']
    assert g['exit_code'] == 0 and g['max_swap_growth_bytes'] == 0
records = []
previous_telemetry = 0
for c, b in zip(candidate_rows, base_rows):
    rank = c['rank']
    cy = np.load(OUT / f'pred_{rank}.npy')
    by = np.load(BASE / f'pred_{rank}.npy')
    assert cy.shape == by.shape == (16, 1024)
    assert cy.dtype == by.dtype == np.float64
    assert np.isfinite(cy).all()
    assert cy[:7].tobytes() == by[:7].tobytes()
    truth = np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
    independent = []
    for row, pred in [(c, cy), (b, by)]:
        mse = math.fsum((float(x) - float(y)) ** 2 for x, y in zip(pred[15], truth)) / 1024
        util = row['flops'] / 2**41
        adjusted = mse * max(.1, util)
        assert math.isclose(row['mse'], mse, rel_tol=2e-15)
        assert row['util'] == util
        assert math.isclose(row['adjusted'], adjusted, rel_tol=2e-15)
        independent.append(dict(mse=mse, util=util, adjusted=adjusted, flops=row['flops']))
    telemetry = c['old_row_importance'][previous_telemetry:]
    previous_telemetry = len(c['old_row_importance'])
    assert len(telemetry) == 8
    assert all(t['kept'] >= t['mandatory'] and t['kept'] <= t['rows'] for t in telemetry)
    assert all(t['base_kept'] - t['kept'] == 128 for t in telemetry)
    records.append(dict(rank=rank, candidate=independent[0], baseline=independent[1],
        raw_change_pct=100 * (c['mse'] / b['mse'] - 1),
        bill_change_pct=100 * (c['flops'] / b['flops'] - 1),
        adjusted_change_pct=100 * (c['adjusted'] / b['adjusted'] - 1),
        adjusted_win=c['adjusted'] < b['adjusted'], first7_output_bytes_exact=True,
        finite_all16=True, local_recon_calls=8, fewer_local_rows=1024,
        actual_local_selection_counts=telemetry,
        candidate_prediction_sha256=sha(OUT / f'pred_{rank}.npy'),
        baseline_prediction_sha256=sha(BASE / f'pred_{rank}.npy')))
pooled = {key: 100 * (math.fsum(r[key] for r in candidate_rows) / math.fsum(r[key] for r in base_rows) - 1)
          for key in ['mse', 'flops', 'adjusted']}
gates = dict(two_adjusted_wins=all(r['adjusted_win'] for r in records),
             pooled_adjusted_improvement_at_least_1p5=pooled['adjusted'] <= -1.5,
             first7_parity=all(r['first7_output_bytes_exact'] for r in records))
result = dict(networks=records, pooled_change_pct=pooled, gates=gates,
    passes_train2_gate=all(gates.values()), source_seal_verified=True,
    candidate_sha256=sha(D / 'candidate.py'), scorer_sha256=sha(Path(__file__)),
    scores_independently_reconciled=True,
    scope='Fixed TRAIN2 native endpoint and full own bill; reused development networks',
    candidate_built=True, final_mean_training_truth_read=True, valid40_accessed=False,
    fits_run=False, new_samples=False, board_submitted=False,
    next_action='Expand only on the predeclared pass; otherwise close without rescue')
with (D / 'assessment.json').open('x') as f:
    json.dump(result, f, indent=2, allow_nan=False)
print(json.dumps(dict(pooled_change_pct=pooled, gates=gates,
                     passes_train2_gate=result['passes_train2_gate'])), flush=True)
