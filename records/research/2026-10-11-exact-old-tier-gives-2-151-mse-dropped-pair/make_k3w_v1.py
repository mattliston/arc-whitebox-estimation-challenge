"""TAG opus_engineer a62 (2026-10-11 22:0xZ; physicist's carry proposal (f) 21:21Z, T120 arm 'colk3'): the join sketch's column weights
lam^2 |p|^PEXP in place of lam^2 (|p|^2 = the joined column's sum of squares over the rows previous, as t120_worker_pairsketch.py's
weight * (J * J).sum(0) ** (PEXP / 2)); everything else of the Nystrom join unchanged (row selection by the weighted Gram diagonal, the old
frame's core). ONE text edit in the ACTIVE old-tier push (the definition the OTT wrapper wraps, '# Approximate range-finder core'), on the
line 'weight = coefficient * coefficient', reading a global _K3W_PEXP (0 = the parent bit for bit) appended at the end.
usage: make_k3w_v1.py <parent.py> <out name> <PEXP>   -> a62_modules/s49/<out name>/candidate.py + build.json"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent, name, pexp = Path(sys.argv[1]).resolve(), sys.argv[2], float(sys.argv[3])
out = HERE/'s49'/name
out.mkdir(parents=True)
text = parent.read_text()
anchor = "# Approximate range-finder core, original orthogonal factor projection.\ndef _shared_old384_push(layer, w32, p, previous, newborn, lam, state, rows=None):\n"
assert text.count(anchor) == 1
at = text.index(anchor)
end = text.index('\ndef ', at + len(anchor))
old = "        weight = coefficient * coefficient\n"
body = text[at:end]
assert body.count(old) == 1, body.count(old)
new = old + ("        if _K3W_PEXP:   # opus-engineer a62 K3W: the physicist's colk3 sketch weights lam^2 |p|^PEXP (|p| over the rows previous)\n"
             "            weight = weight * fnp.sum(J * J, axis=0) ** (_K3W_PEXP / 2)\n")
text = text[:at] + body.replace(old, new) + text[end:]
text = text.rstrip('\n') + '\n\n_K3W_PEXP = %r   # opus-engineer a62 make_k3w_v1: the join sketch weight exponent (0 = lam^2 only)\n' % pexp
(out/'candidate.py').write_text(text)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
(out/'build.json').write_text(json.dumps(dict(parent=str(parent), parent_sha256=sha(parent), edit=dict(old=old, new=new), pexp=pexp,
                                               candidate=str(out/'candidate.py'), candidate_sha256=sha(out/'candidate.py')), indent=1) + '\n')
print(name, sha(out/'candidate.py')[:8])
