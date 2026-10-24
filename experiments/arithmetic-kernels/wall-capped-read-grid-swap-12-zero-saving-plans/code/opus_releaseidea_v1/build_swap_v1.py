"""[opus-releaseidea] SWAP v1 builder: final_v4g_psm2_dm2_lb2 (release 3786ca8b / parity b0d96393) + the wall-capped Smirnov grid table
(opus_smtable_v1/fragment_smtable_cap_v2: 375 new young-read keys) + a dictionary-only drop of the 12 table167 read plans whose net FLOP
saving vs the incumbent read is ~0 (bench_existing_v1.jsonl: saving / extra instructions < 3.1 FLOP per kinstr; -62..+45 MFLOP each,
sum +1.5 MFLOP, +149 G instructions over one call each). Both fragments appended LAST to the release and to its parity copy (same bytes),
so the parity copy still differs from the release only in the governor threshold lines. Writes cand_swap_v1_{release,parity}.py + BUILD_swap_v1.json."""
import hashlib, json, re
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
D = Path(__file__).resolve().parent; ST = O/'lead/runs/opus_stack_v1'; SM = O/'lead/runs/opus_smtable_v1'
sha = lambda b: hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()
rel, par = (ST/'cand_final_v4g_psm2_dm2_lb2_release.py').read_text(), (ST/'cand_final_v4g_psm2_dm2_lb2_parity.py').read_text()
assert sha(rel).startswith('3786ca8b') and sha(par).startswith('b0d96393')
cap = (SM/'fragment_smtable_cap_v2.py.txt').read_text()
rows = [json.loads(l) for l in (SM/'bench_existing_v1.jsonl').read_text().splitlines()]
ratio = lambda r: (r['incumbent']['flops'] - r['smirnov']['flops']) / max(r['smirnov']['instr'] - r['incumbent']['instr'], 1) * 1e3
drop = sorted(tuple(r['shape']) for r in rows if ratio(r) < 3.1)
assert len(drop) == 12, drop
dF = sum(r['incumbent']['flops'] - r['smirnov']['flops'] for r in rows if tuple(r['shape']) in drop)
dI = sum(r['smirnov']['instr'] - r['incumbent']['instr'] for r in rows if tuple(r['shape']) in drop)
frag = ("\n\n# BEGIN [opus-releaseidea] SWAP v1 DROP (append LAST, after fragment_smtable_cap_v2; dictionary-only): the 12 table167 young-read\n"
        "# Smirnov plans whose net FLOP saving vs the incumbent read is ~0 (real-compute bench, opus_smtable_v1/bench_existing_v1.jsonl:\n"
        f"# saving per extra instruction < 3.1 FLOP/kinstr; sum over one call each {dF/1e6:+.1f} MFLOP for {dI/1e9:.0f} G extra instructions)\n"
        "# fall back to the incumbent read, paying for the wall of the wall-capped grid table appended before this.\n"
        f"for _sw_key in {tuple(drop)!r}:\n    assert _sw_key in _ASTRA_SM40_PLANS, _sw_key\n    del _ASTRA_SM40_PLANS[_sw_key]\n"
        "del _sw_key\n# END [opus-releaseidea] SWAP v1 DROP\n")
# the cap keys must be new (disjoint from the base table) and none of them may be dropped
base_keys = set(eval(rel[rel.index('\n_ASTRA_SM40_PLANS = ') + len('\n_ASTRA_SM40_PLANS = '):rel.index('\n', rel.index('\n_ASTRA_SM40_PLANS = ') + 1)]).keys())
cap_keys = set(eval(cap[cap.index('_ASTRA_SM40_PLANS.update(') + len('_ASTRA_SM40_PLANS.update('):cap.rindex(')')]).keys())
assert set(drop) <= base_keys and not (cap_keys & base_keys), (len(set(drop) - base_keys), len(cap_keys & base_keys))
assert rel.count('\n_ASTRA_SM40_PLANS = ') == 1 and '_ASTRA_SM40_PLANS =' not in rel[rel.index('\n_ASTRA_SM40_PLANS = ') + 30:]
out = {}
for name, text in (('release', rel), ('parity', par)):
    t = text + cap + frag
    compile(t, name, 'exec')
    p = D/f'cand_swap_v1_{name}.py'; p.write_text(t); out[name] = sha(t)
TH = re.compile(r'^_GOV_[A-Z0-9_]*(?:CAP|WALL)[A-Z0-9_]*_S *= ')
a, b = (D/'cand_swap_v1_release.py').read_text().split('\n'), (D/'cand_swap_v1_parity.py').read_text().split('\n')
assert len(a) == len(b) and all(x == y or (TH.match(x) and TH.match(y)) for x, y in zip(a, b))
(D/'fragment_swap_drop12_v1.py.txt').write_text(frag)
json.dump(dict(base_release=sha(rel), base_parity=sha(par), cap_fragment=sha(cap), cap_keys=len(cap_keys), drop_fragment=sha(frag), drop=drop,
               drop_bench_dF=dF, drop_bench_dI=dI, release_sha256=out['release'], parity_sha256=out['parity']), open(D/'BUILD_swap_v1.json', 'w'), indent=1)
print('release', out['release'][:8], 'parity', out['parity'][:8], 'cap keys', len(cap_keys), 'drop', len(drop), f'dF {dF/1e6:+.1f}M dI {dI/1e9:.0f}G')
