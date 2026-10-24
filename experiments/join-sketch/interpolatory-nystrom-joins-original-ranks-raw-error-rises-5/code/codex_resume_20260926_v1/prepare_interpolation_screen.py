"""Nyström interpolation avoids projecting all columns into an orthogonal basis."""
from pathlib import Path
import json
import hashlib
HERE=Path(__file__).resolve().parent
source=HERE/'spectral_screen_s29.py'
text=source.read_text()
start=text.index("            if os.environ.get('SPECTRAL_KIND', 'eigh') == 'eigh':")
end=text.index("            basis, old_columns = Qn, stop",start)
text=text[:start]+'''            r = min(want, int(J.shape[0]))
            rows = t.argsort(-d.to(t.float64), stable=True)[:r]
            Y = J @ (weight[:, None] * J[rows, :].T)
            if old_columns: Y = Y + Qp @ (core @ Qp[rows, :].T)
            K = (Y[rows, :] + Y[rows, :].T) * 0.5
            # Solve is a legitimate paid operation, with no reference targets.
            Qn = t.linalg.solve(K, Y.T).T
            Fj = J[rows, :]
            if old_columns:
                F = t.cat((Qp[rows, :] @ F, Fj), dim=1)
            else:
                F = Fj
            core = K
            E = {}
''' + text[end:]
text=text.replace("os.environ.get('SPECTRAL_KIND', 'eigh')", "'interpolation'")
target=HERE/'interpolation_screen_s29.py'
with target.open('x') as f:f.write(text)
(HERE/'interpolation_screen_manifest.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),method='Q = G[:,I] inv(G[I,I]), F = old factors at I; weighted core G[I,I].'),indent=2)+'\n')
print(target)
