#!/usr/bin/env python3
"""TAG fable-porter (PT-2): refit .pt -> native candidate. Builds the gain-mixture b-fix variant of the bank at <gamma>
(build_bfix_variant.patch on the pinned bank; gamma 0 = the bank's loop) and writes the learned literals from <params.pt>
into it with fable_refit_v1/export_params.py (run under the torch runtime). Refuses to overwrite <out.py> and refuses a .pt
with keys other than p, a, c, e (export_params.py would silently drop them).
usage: build_refit_candidate.py <params.pt> <gamma> <out.py>"""
import sys, json, hashlib, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(HERE))
from build_bfix_variant import patch
BANK = P2 / 'codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py'
BANK_SHA = 'd79463f389a6b6e4172415b183601d327bc13b560fd7096ddbf781dce7a94d0e'
TORCH_PY = '/Users/matt/anaconda3/bin/python'
pt, gamma, out = Path(sys.argv[1]).resolve(), float(sys.argv[2]), Path(sys.argv[3]).resolve()
subprocess.run([TORCH_PY, '-c', "import sys, torch; extra = set(torch.load(sys.argv[1])) - {'p', 'a', 'c', 'e'}; "
                "assert not extra, f'params keys not exported by export_params.py: {extra}'", str(pt)], check=True)
base = BANK.read_bytes()
assert hashlib.sha256(base).hexdigest() == BANK_SHA, 'bank candidate changed'
with out.open('x') as f:
    f.write(patch(base.decode(), gamma))
subprocess.run([TORCH_PY, str(P2 / 'fable_refit_v1/export_params.py'), str(out), str(pt), str(out)], check=True, stdout=subprocess.DEVNULL)
assert f"\n_BFIX_GAMMA = {gamma!r}\n" in out.read_text()
print(json.dumps(dict(out=str(out), params=str(pt), gamma=gamma, out_sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
