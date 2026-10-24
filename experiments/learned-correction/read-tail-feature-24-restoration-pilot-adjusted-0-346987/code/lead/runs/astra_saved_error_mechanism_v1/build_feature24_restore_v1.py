from pathlib import Path
import ast,hashlib,json,difflib
D=Path(__file__).resolve().parent
parent=D.parent/'astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(parent)=='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
source=parent.read_text();lines=source.splitlines(keepends=True)
node=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='_rtt_read'][-1]
old=''.join(lines[node.lineno-1:node.end_lineno])
needle="        if old:\n            assert context['inner_calls']==1 and context['inner']is not None\n            inner=context['inner'];assert tuple(inner.shape)==(m,rank)\n"
assert old.count(needle)==1
insert="        _f24_tail_restore(hold(left[:,old:]), hold(young_right[c15:c2]),\n            hold(lam[old:]), tail, rows, hold(live_indices[c15:c2]), hold)\n"
new=old.replace(needle,insert+needle)
fragment=(D/'fragment_feature24_restore_v1.py').read_text()
out=''.join(lines[:node.lineno-1])+new+''.join(lines[node.end_lineno:])+fragment
ast.parse(out)
candidate=D/'candidate_feature24_tail_restore_all_v1.py';candidate.open('x').write(out)
diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='parent active _rtt_read',tofile='candidate active _rtt_read'))+'\nAPPENDED:\n'+fragment
(D/'feature24_restore_diff_v1.patch').open('x').write(diff)
profile=json.loads((D.parent/'astra_rtt2_layer_restore_audit_v1/TRAIN16_ERROR_ONSET_AND_TARIFF_v1.json').read_text())['per_layer_warm_tariff']
# Shape-based upper bound: at most every tail column has a response-row diagonal.
# Includes three matvecs and diagonal einsum; gathers/casts/put need native billing.
cost=[]
for p in profile:
 m,k,nt=p['m'],p['k'],p['c2']-p['c15']
 flops=2*k*(nt+m)+2*m*nt+3*k*nt+100*(m+nt+1024)
 cost.append(dict(layer=p['layer'],flops_upper=flops))
manifest=dict(parent=str(parent),parent_sha256=sha(parent),candidate=str(candidate),candidate_sha256=sha(candidate),fragment_sha256=sha(D/'fragment_feature24_restore_v1.py'),scope='All RTT-changed hidden layers 2..14, fixed exact young-tail feature24 restoration; first two layers and terminal remain unchanged; no fitted constants.',formula='sum_j((L_i * lambda) dot R_j - current_young_tail_ij)/sigma_j / (sqrt(n)*variance_i), excluding i=j',no_covariance_or_K21_plane_mutation=True,model_calls=0,train16_geometry_shape_cost_upper=sum(p['flops_upper']for p in cost),train16_geometry_cost_pct_upper=100*sum(p['flops_upper']for p in cost)/238299111296,per_layer=cost,limits='Cost is shape arithmetic only, not a native FNP bill; exact old-tail read contribution is untouched. Original float32 K21 construction and accumulation rounding are not reproduced by the analytic float64 moment correction.')
(D/'FEATURE24_CANDIDATE_MANIFEST_v1.json').open('x').write(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
