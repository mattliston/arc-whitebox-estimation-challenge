"""Target-free best-subspace diagnostic: can a better range finder buy rank?"""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
source = P2 / 'fable_analyst_v1/r320_features.py'
text = source.read_text()
text = text.replace("import os, sys, json, time", "import os, sys, json, time, gc")
text = text.replace("'fable_physicist_v1/refit6b/s6b_loopR3_joint_s23'", "'fable_physicist_v1/refit6b/s6b_loopR7s_joint_3ep'")
text = text.replace("os.environ.get('R320_EPOCH', '0')", "os.environ.get('R320_EPOCH', '1')")
text = text.replace("open(HERE / 'split_train640.json')", "open(P2 / 'fable_analyst_v1/split_train640.json')")
start = text.index("            rows = torch.argsort(-d.to(t.float64)")
end = text.index("            Fj = Qn.T @ J", start)
text = text[:start] + '''            if os.environ.get('SPECTRAL_KIND', 'eigh') == 'eigh':
                G = (J * weight[None, :]) @ J.T
                if old_columns: G = G + Qp @ core @ Qp.T
                ev, vec = t.linalg.eigh((G + G.T) * 0.5)
                Qn = vec[:, -max(rank, want):].flip(1)
                del G, ev, vec
            else:
                rows = t.argsort(-d.to(t.float64), stable=True)[:max(rank, want)]
                Y = J @ (weight[:, None] * J[rows, :].T)
                if old_columns: Y = Y + Qp @ (core @ Qp[rows, :].T)
                Qn = qr(t, Y)
''' + text[end:]
text = text.replace("            fo.write(json.dumps(rec) + '\\n'); fo.flush()", "            rec['kind'] = os.environ.get('SPECTRAL_KIND', 'eigh')\n            np.save(OUT.with_name(OUT.stem + '_' + str(r_) + '_' + arm + '_pred.npy'), pred)\n            fo.write(json.dumps(rec) + '\\n'); fo.flush()")
text += '\n'
target = HERE / 'spectral_screen_s29.py'
with target.open('x') as out: out.write(text)
(HERE / 'spectral_screen_manifest.json').write_text(json.dumps(dict(source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    claim='Optimal Gram subspace diagnostic using estimator state only; native cost not yet measured.'), indent=2)+'\n')
print(target)
