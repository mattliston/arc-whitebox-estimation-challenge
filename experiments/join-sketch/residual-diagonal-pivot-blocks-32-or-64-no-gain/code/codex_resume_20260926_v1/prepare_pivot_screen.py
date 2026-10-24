"""Accuracy ceiling for block residual-diagonal Nystrom row selection.

The research screen materializes the Gram matrix. Deployment must either pay for
that matrix or price a column-at-a-time implementation; no bill claim is made here.
"""
from pathlib import Path
import json
import hashlib

HERE = Path(__file__).resolve().parent
source = HERE / 'interpolation_screen_s29.py'
text = source.read_text()
start = text.index('            rows = t.argsort(-d.to(t.float64), stable=True)[:r]')
end = text.index('            K = (Y[rows, :] + Y[rows, :].T) * 0.5', start)
text = text[:start] + '''            block = int(os.environ.get('PIVOT_BLOCK', '64'))
            G = (J * weight[None, :]) @ J.T
            if old_columns: G = G + Qp @ core @ Qp.T
            G = (G + G.T) * 0.5
            residual = G.diag().clone()
            selected = []
            Z = G[:, :0]
            used = t.zeros(G.shape[0], dtype=t.bool, device=G.device)
            for offset in range(0, r, block):
                width = min(block, r - offset)
                ids = t.argsort(-residual.masked_fill(used, -float('inf')), stable=True)[:width]
                R = G[:, ids]
                if offset: R = R - Z @ Z[ids, :].T
                Kb = (R[ids, :] + R[ids, :].T) * 0.5
                jitter = t.clamp(Kb.diag().abs().max(), min=1e-30) * 1e-12
                Lb = t.linalg.cholesky(Kb + jitter * t.eye(width, dtype=G.dtype, device=G.device))
                Nb = t.linalg.solve_triangular(Lb, R.T, upper=False).T
                Z = t.cat((Z, Nb), dim=1)
                residual = residual - (Nb * Nb).sum(1)
                used[ids] = True
                selected.append(ids)
            rows = t.cat(selected)
            Y = G[:, rows]
            del G, Z, Nb, R, Kb, Lb, residual, used
''' + text[end:]
text = text.replace("rec['kind'] = 'interpolation'", "rec['kind'] = 'pivot' + os.environ.get('PIVOT_BLOCK', '64')")
target = HERE / 'pivot_screen_s29.py'
with target.open('x') as f: f.write(text)
with (HERE / 'pivot_screen_manifest.json').open('x') as f:
    json.dump(dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   target_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   scope='target-free accuracy ceiling; full Gram cost not included'), f, indent=2)
print(target)
