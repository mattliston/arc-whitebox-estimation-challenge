from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent
P=Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py")
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P)=='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
t=P.read_text();tree=ast.parse(t);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_astra_bcc_push_impl')
lines=t.splitlines(keepends=True);body=''.join(lines[node.lineno-1:node.end_lineno])
old="            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])"
new="            _jn_rows = _jrs_select(J, weight, _jn_d, rank) if _JRS_ACTIVE[0] and layer == 7 and old_columns == 0 else fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])"
assert body.count(old)==1
updated=body.replace(old,new)
assert updated.count('\n')==body.count('\n')
prefix=''.join(lines[:node.lineno-1])+updated+''.join(lines[node.end_lineno:])
fragment=(D/'fragment_jrs_HELD_v1.py').read_text()
out=prefix+'\n\n'+fragment
assert out[:-len('\n\n'+fragment)].replace(new,old)==t
ast.parse(out)
path=D/'candidate_jrs_HELD_v1.py'
with path.open('x')as f:f.write(out)
result=dict(parent_path=str(P),parent_sha256=sha(P),candidate_path=str(path),candidate_sha256=sha(path),fragment_sha256=sha(D/'fragment_jrs_HELD_v1.py'),source_inverse_exact=True,changed_existing_function='_astra_bcc_push_impl',changed_existing_line=node.lineno+body[:body.index(old)].count('\n'),same_existing_line_numbers=True,scope='L7 firstjoin only, no old columns; exact original selector otherwise; unchanged Y/QR/core/ranks/cohort/cumulants.',new_setup_keys=0,new_einsum_signatures=0,stats='_JRS_STATS: calls=1,records=[layer7,m,columns,rank384,original_anchorTrue,fallbackBool,reason],retiredTrue; _JRS_ACTIVE=[False]afterpredict.')
with (D/'SOURCE_MANIFEST_HELD_v1.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))

