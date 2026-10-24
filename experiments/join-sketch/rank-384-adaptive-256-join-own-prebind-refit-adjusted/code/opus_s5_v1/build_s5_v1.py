"""[opus-s5] S5 = VN (7fd8f6aa) + Full384 (Astra's rank-base hunks as exact unique text replacements + build_fixed_dispatch_v2 transport
re-run on VN's own table: PLANS (m,k,n,320)->(m,k,n,384) with conflict check, L4_all k 80->81 in FAMILIES and PREBIND) + R256 greedy join
fragment (0c795483) appended last. Writes candidate_VN_F384_v1.py (no join fragment) and candidate_S5raw_v1.py (with R256)."""
import ast, hashlib, json
from pathlib import Path
L = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs')
W = Path(__file__).resolve().parent; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
VN = L/'opus_vn_v1/candidate_VN_v1.py'; PATCH = L/'astra_full384_smirnov_v1/rank_base_source_diff_v1.patch'; R256 = L/'opus_gbfix_v1/fragment_join_greedy_r256_2b108lineage_v1.py.txt'
assert sha(VN).startswith('7fd8f6aa') and sha(R256).startswith('0c79548308ad')
s = VN.read_text()
# 1) rank-base hunks
hunks, cur = [], None
for line in PATCH.read_text().splitlines(keepends=True):
    if line.startswith('@@'):
        cur = ([], []); hunks.append(cur); continue
    if cur is None or line.startswith(('---', '+++')): continue
    tag, body = line[0], line[1:]
    if tag in ' -': cur[0].append(body)
    if tag in ' +': cur[1].append(body)
applied = []
for old, new in hunks:
    o, n = ''.join(old), ''.join(new); c = s.count(o); assert c == 1, ('hunk not unique in VN', c, o[:120]); s = s.replace(o, n); applied.append(len(old))
# 2) fixed dispatch transport on VN's own table (Astra build_fixed_dispatch_v2 logic)
lines = s.splitlines(keepends=True); nodes = {}
for nd in ast.parse(s).body:
    if isinstance(nd, ast.Assign) and len(nd.targets) == 1 and isinstance(nd.targets[0], ast.Name) and nd.targets[0].id in ('_ASTRA_SM40_PLANS', '_ASTRA_SM40_FAMILIES', '_ASTRA_SM40_PREBIND'):
        nodes[nd.targets[0].id] = nd     # last definition wins (module order)
assert len(nodes) == 3
old = {k: ast.literal_eval(nd.value) for k, nd in nodes.items()}; new = dict(old)
plans, moved = {}, []
for key, plan in old['_ASTRA_SM40_PLANS'].items():
    target = key[:3] + (384,) if key[3] == 320 else key
    if target in plans: assert plans[target] == plan, ('conflicting inherited plans', target)
    plans[target] = plan
    if target != key: moved.append([list(key), list(target)])
new['_ASTRA_SM40_PLANS'] = plans
new['_ASTRA_SM40_FAMILIES'] = tuple((('L4_all', 4, 40, 13, 81, 16) if f[0] == 'L4_all' else f) for f in old['_ASTRA_SM40_FAMILIES'])
assert ('L4_all', 4, 40, 13, 80, 16) in old['_ASTRA_SM40_FAMILIES']
new['_ASTRA_SM40_PREBIND'] = tuple(((('L4_all', 4, 40, 13, 81, 16) if f[0] == 'L4_all' else f), a) for f, a in old['_ASTRA_SM40_PREBIND'])
for k, nd in sorted(nodes.items(), key=lambda kv: -kv[1].lineno):
    lines[nd.lineno - 1:nd.end_lineno] = [k + ' = ' + repr(new[k]) + '\n']
vf = ''.join(lines); compile(vf, 'vnf384', 'exec')
p1 = W/'candidate_VN_F384_v1.py'; p1.write_text(vf)
s5 = vf + '\n\n' + R256.read_text(); p2 = W/'candidate_S5raw_v1.py'; p2.write_text(s5); compile(s5, str(p2), 'exec')
m = dict(VN=dict(path=str(VN), sha256=sha(VN)), rank_patch=dict(path=str(PATCH), sha256=sha(PATCH), hunks=len(hunks)), R256=dict(path=str(R256), sha256=sha(R256)),
         plans_before=len(old['_ASTRA_SM40_PLANS']), plans_after=len(plans), moved_320_to_384=len(moved),
         VN_F384=dict(path=str(p1), sha256=sha(p1)), S5raw=dict(path=str(p2), sha256=sha(p2)))
(W/'S5_BUILD_v1.json').write_text(json.dumps(m, indent=2) + '\n')
print(json.dumps({k: (v['sha256'][:12] if isinstance(v, dict) and 'sha256' in v else v) for k, v in m.items()}))
