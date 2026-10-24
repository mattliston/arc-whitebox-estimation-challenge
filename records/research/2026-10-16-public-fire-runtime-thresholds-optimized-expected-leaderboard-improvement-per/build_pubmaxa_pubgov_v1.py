"""[opus-pubgov] PUBLIC-ONLY max with the public-fire governor optimum: pubmax A (release 75ccf4d8 = swap + uncapped read grid) with
the same four constants as swappub (FULL residual 0.40, EXACT residual 0.42, EXACT wall 116 s, LEAN-wall tier off; LEAN residual 0.46
and SHED 123 s unchanged), the best set for A on u64's refined wall model in both LEAN-projection modes (pubgov_scan_v2.py).
Parity copy = A's parity copy 42c60af4 byte for byte. Public fires only; never a designation candidate."""
import hashlib
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
G = Path(__file__).resolve().parent; P = O/'lead/runs/opus_pubmax_v1'; ST = O/'lead/runs/opus_stack_v1'
sha = lambda b: hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()
rel, par = (P/'cand_pubmax_A_release.py').read_text(), (P/'cand_pubmax_A_parity.py').read_text()
assert sha(rel).startswith('75ccf4d8') and sha(par).startswith('42c60af4')
edits = {'_GOV_CAP_S = 0.39\n': '_GOV_CAP_S = 0.4\n', '_GOV_CAP2_S = 0.41\n': '_GOV_CAP2_S = 0.42\n',
         '_GOV_WALL1_S = 112.0\n': '_GOV_WALL1_S = 116.0\n', '_GOV_WALL_S = 118.0\n': '_GOV_WALL_S = 1e9\n'}
new = rel
for a, b in edits.items():
    assert new.count('\n' + a) == 1, a
    new = new.replace('\n' + a, '\n' + b)
ns = {'__name__': 'asm_lib', '__file__': str(ST/'assemble_final_v2.py')}
exec(compile((ST/'assemble_final_v2.py').read_text().split("\nif __name__ == '__main__':")[0], 'asm', 'exec'), ns)
par2, hits = ns['parity_of'](new); assert par2 == par
compile(new, 'pubmaxa_pubgov', 'exec'); (G/'cand_pubmaxa_pubgov_v1_release.py').write_text(new)
print('release', sha(new)[:12], 'parity = A parity', sha(par2)[:12], hits)
