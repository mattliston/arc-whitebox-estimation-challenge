"""[opus-pubgov] pubmax A with u64's GOV v4h LEAN/SHED walls (122 s / 125 s, his build 0bc7bd5d) plus the public-fire residual/EXACT
thresholds (FULL residual 0.40, EXACT residual 0.42, EXACT wall 116 s). On u64's refined wall model it beats his v4h on BOTH
metrics: board gain per public fire 0.347/0.303 (raw) and 0.375/0.339 (post) vs 0.309/0.250 and 0.334/0.273, and linear
net -0.163/-0.193% vs -0.116% (no SHED of the record's 132.6 s network). Built from 0bc7bd5d by three constant edits;
parity copy = A's 42c60af4. Public fires only; never a designation candidate."""
import hashlib
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
G = Path(__file__).resolve().parent; W = O/'lead/runs/opus_wallx_v1'; P = O/'lead/runs/opus_pubmax_v1'; ST = O/'lead/runs/opus_stack_v1'
sha = lambda b: hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()
rel, par = (W/'cand_pubmax_A_v4h_release.py').read_text(), (P/'cand_pubmax_A_parity.py').read_text()
assert sha(rel).startswith('0bc7bd5d') and sha(par).startswith('42c60af4')
edits = {'_GOV_CAP_S = 0.39\n': '_GOV_CAP_S = 0.4\n', '_GOV_CAP2_S = 0.41\n': '_GOV_CAP2_S = 0.42\n', '_GOV_WALL1_S = 112.0\n': '_GOV_WALL1_S = 116.0\n'}
new = rel
for a, b in edits.items():
    assert new.count('\n' + a) == 1, a
    new = new.replace('\n' + a, '\n' + b)
ns = {'__name__': 'asm_lib', '__file__': str(ST/'assemble_final_v2.py')}
exec(compile((ST/'assemble_final_v2.py').read_text().split("\nif __name__ == '__main__':")[0], 'asm', 'exec'), ns)
_, hits = ns['parity_of'](new)
# the parity copy keeps u64's comments on the two wall lines; the release gate compares threshold lines by name and value
a, b = new.split('\n'), par.split('\n'); assert len(a) == len(b)
import re
TH = re.compile(r'^(_GOV_[A-Z0-9_]*(?:CAP|WALL)[A-Z0-9_]*_S) = (.+?)(\s*#.*)?$')
for x, y in zip(a, b):
    if x != y:
        mx, my = TH.match(x), TH.match(y); assert mx and my and mx.group(1) == my.group(1) and float(my.group(2)) == 1e9, (x, y)
compile(new, 'pg2', 'exec'); (G/'cand_pubmaxa_pg2_v1_release.py').write_text(new)
print('release', sha(new)[:12], hits)
