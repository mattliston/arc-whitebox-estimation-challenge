"""[opus-pubgov] PUBLIC-FIRE variant of the swap (constants only): governor thresholds chosen on u64's replay model to maximize the
expected board improvement per public fire (pubgov_scan_v1.py): FULL residual 0.39 -> 0.40, EXACT residual 0.41 -> 0.42, EXACT wall
112 -> 116 s, LEAN wall tier OFF (1e9); LEAN residual 0.46 and SHED 123 s unchanged. Its parity copy (all six thresholds 1e9) is the
swap's parity copy bc189784 byte for byte, so the swap's official120 / boundary / pairs evidence applies unchanged.
Private designations keep the default (or lower) thresholds: this file is for public fires only."""
import hashlib, importlib.util, re
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
G = Path(__file__).resolve().parent; X = O/'lead/runs/opus_releaseidea_v1'; ST = O/'lead/runs/opus_stack_v1'
sha = lambda b: hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()
rel = (X/'cand_swap_v1_release.py').read_text(); par = (X/'cand_swap_v1_parity.py').read_text()
assert sha(rel).startswith('d81105ed') and sha(par).startswith('bc189784')
edits = {'_GOV_CAP_S = 0.39\n': '_GOV_CAP_S = 0.4\n', '_GOV_CAP2_S = 0.41\n': '_GOV_CAP2_S = 0.42\n',
         '_GOV_WALL1_S = 112.0\n': '_GOV_WALL1_S = 116.0\n', '_GOV_WALL_S = 118.0\n': '_GOV_WALL_S = 1e9\n'}
new = rel
for a, b in edits.items():
    assert new.count('\n' + a) == 1, a
    new = new.replace('\n' + a, '\n' + b)
ns = {'__name__': 'asm_lib', '__file__': str(ST/'assemble_final_v2.py')}; src = (ST/'assemble_final_v2.py').read_text()
exec(compile(src.split("\nif __name__ == '__main__':")[0], 'asm', 'exec'), ns)
par2, hits = ns['parity_of'](new)
assert par2 == par, 'parity copy differs from the swap parity copy'
compile(new, 'swappub', 'exec')
(G/'cand_swappub_v1_release.py').write_text(new)
print('release', sha(new)[:12], 'parity = swap parity', sha(par2)[:12], 'thresholds', hits)
