"""[opus-pubmax] builder v1: increments on top of the SWAP (d81105ed = final 3786ca8b + cap table + drop12), each as release + parity copy
(parity = release with every governor threshold at 1e9; the same assembler step order as the final, smallsm inserted before lazybind):
  A  = swap + the rest of the uncapped read grid (opus_smtable_v1 v1 grid: 1,710 keys = the 375 cap keys + 1,335 more; extracted
       byte-exact from cand_GOV3_smtable_v1.py be57709e = GOV_v3 0640c0cf + grid fragment), dictionary-only, appended last
  B  = assemble(GOV v4g 805c3598 -> dm2 -> pushsm_v2 -> smallsm_v2 95ea4353 -> lazybind v2) + cap table + drop12 (= swap + join/old-bank SM)
  AB = B + the grid remainder
Writes cand_pubmax_<v>_{release,parity}.py and BUILD_pubmax_v1.json; asserts that B's assembly without smallsm reproduces the final 3786ca8b."""
import hashlib, importlib.util, json, re, subprocess, sys
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
D = Path(__file__).resolve().parent; ST = O/'lead/runs/opus_stack_v1'; SM = O/'lead/runs/opus_smtable_v1'; X = O/'lead/runs/opus_releaseidea_v1'
sha = lambda b: hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()
spec = importlib.util.spec_from_file_location('asm', ST/'assemble_final_v2.py'); asm = importlib.util.module_from_spec(spec)
src = (ST/'assemble_final_v2.py').read_text(); main_at = src.index('\ndef main(') if '\ndef main(' in src else None
# reuse the assembler's own functions (dm2, parity_of) without running its CLI
ns = {'__name__': 'asm_lib', '__file__': str(ST/'assemble_final_v2.py')}
exec(compile(src.split("\nif __name__ == '__main__':")[0], str(ST/'assemble_final_v2.py'), 'exec'), ns)
gov = (O/'lead/runs/opus_governor_v1/candidate_GOV_v4g.py').read_text(); assert sha(gov).startswith('805c3598')
psm = (ST/'fragment_pushsm_v2_withrule_from_PSM18e8f16b.py.txt').read_text(); assert sha(psm).startswith('bf3746db')
osm = (O/'lead/runs/opus_small_v1/fragment_smallsm_v2.py.txt').read_text(); assert sha(osm).startswith('95ea4353')
lazy = (O/'lead/runs/opus_lazybind_v1/fragment_lazybind_v2.py.txt').read_text(); assert sha(lazy).startswith('65ccb979')
cap = (SM/'fragment_smtable_cap_v2.py.txt').read_text(); drop = (X/'fragment_swap_drop12_v1.py.txt').read_text()
g3 = (O/'lead/runs/opus_governor_v1/candidate_GOV_v3.py').read_text(); grid_c = (SM/'cand_GOV3_smtable_v1.py').read_text()
assert sha(g3).startswith('0640c0cf') and sha(grid_c).startswith('be57709e') and grid_c.startswith(g3)
grid = grid_c[len(g3):]
gk = set(eval(grid[grid.index('_ASTRA_SM40_PLANS.update(') + 25:grid.rindex(')')]).keys())
ck = set(eval(cap[cap.index('_ASTRA_SM40_PLANS.update(') + 25:cap.rindex(')')]).keys())
assert ck <= gk and len(gk) == 1710, (len(gk), len(ck - gk))
dk = set(eval(drop[drop.index('for _sw_key in ') + 15:drop.index(':\n    assert')]))
assert not (gk & dk)
# the grid remainder as its own dictionary-only fragment: the v1 grid's plans for the keys the cap table does not carry
G = eval(grid[grid.index('_ASTRA_SM40_PLANS.update(') + 25:grid.rindex(')')])
rem = {k: G[k] for k in sorted(gk - ck)}
frag_rem = ("\n\n# BEGIN [opus-pubmax] GRID REMAINDER v1 (append LAST; dictionary-only): the 1,335 keys of the uncapped v1 read grid\n"
            "# (opus_smtable_v1, cand_GOV3_smtable_v1 be57709e) that the wall-capped table does not carry, with their v1 plans.\n"
            f"_ASTRA_SM40_PLANS.update({rem!r})\n# END [opus-pubmax] GRID REMAINDER v1\n")
assert len(rem) == 1335
def assemble(frags):
    t = ns['dm2'](gov)
    for f in frags:
        t += f
    return t + lazy
base_final = assemble([psm]); assert sha(base_final).startswith('3786ca8b'), sha(base_final)[:8]   # self-test: reproduces the final
swap = base_final + cap + drop; assert sha(swap).startswith('d81105ed')
variants = {'A': swap + frag_rem, 'B': assemble([psm, osm]) + cap + drop}
variants['AB'] = variants['B'] + frag_rem
out = dict(grid_remainder_sha256=sha(frag_rem), grid_keys=len(gk), cap_keys=len(ck), remainder_keys=len(rem))
for v, rel in variants.items():
    par, hits = ns['parity_of'](rel)
    compile(rel, v, 'exec'); compile(par, v + 'p', 'exec')
    (D/f'cand_pubmax_{v}_release.py').write_text(rel); (D/f'cand_pubmax_{v}_parity.py').write_text(par)
    out[v] = dict(release=sha(rel), parity=sha(par), thresholds=hits)
(D/'fragment_grid_remainder_v1.py.txt').write_text(frag_rem)
json.dump(out, open(D/'BUILD_pubmax_v1.json', 'w'), indent=1)
print({v: (out[v]['release'][:8], out[v]['parity'][:8]) for v in variants}, 'remainder', len(rem))
