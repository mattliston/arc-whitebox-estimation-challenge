"""Correct the bridge's comparison between full and active repeated planes.

V1 stopped before any result: it compared a full physical tensor query to
the bank's active-only off-diagonal CP/star plane. Compare active entries
and all diagonals, and apply the exact same active support to the new
off-diagonal residual. No numerical tolerance or gain gate is loosened.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent = HERE / 'check_one_high_actual_value_v1.py'
source = parent.read_text()
changes = (
    ('one_high_actual_value_r35_l12_v1', 'one_high_actual_value_r35_l12_v2'),
    ('from current_pair_oracle_math_v1 import exact_change',
     'from current_pair_oracle_math_v1 import exact_change\nfrom radial_binary_compact import select'),
    ("parity = float(np.max(np.abs(K[f['indices']] - f['model_next_K21'])))",
     """_, alive = select(ndtr(mu / np.sqrt(np.maximum(np.diag(C), 1e-10))), .01)
live = alive[:, None] * alive[None, :]
np.fill_diagonal(live, True)
mask = live[f['indices']]
parity = float(np.max(np.abs(K[f['indices']] - f['model_next_K21'])[mask]))"""),
    ("        results[arm], dc, dv = measure(delta)",
     "        delta *= live\n        results[arm], dc, dv = measure(delta)"),
    ("Full1024-row exact distinct q4 projection, natural-strength finite pair response.",
     "Full1024-row exact distinct q4 projection, then actual active-support mask with all diagonals retained, natural-strength finite pair response. V1 full-versus-active parity mismatch corrected without changing a tolerance or gain gate."),
)
for old, new in changes:
    assert source.count(old) == 1, old
    source = source.replace(old, new)
exec(compile(source, str(parent), 'exec'), dict(__file__=__file__, __name__='__main__'))
