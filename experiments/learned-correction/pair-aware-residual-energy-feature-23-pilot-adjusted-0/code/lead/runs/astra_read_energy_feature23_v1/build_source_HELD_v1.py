from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent;P=D.parent/'astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P)=='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
p=P.read_text();old='_e2_H(fnp.sqrt(_e2_H(fnp.sum(skplane_square, axis=0))))';new='_e2_H(fnp.sqrt(_e2_H(_re23_add(l, _e2_H(fnp.sum(skplane_square, axis=0)), _e2_H))))';assert p.count(old)==1
body=p.replace(old,new)
start=body.rindex('def _rtt_read(');end=body.index('\n_RC_PARENT_READ=_rtt_read',start)
read=body[start:end];anchor='        if old:\n            assert context[\'inner_calls\']==1 and context[\'inner\']is not None\n            inner=context[\'inner\']';assert read.count(anchor)==1
addition="        _re23_publish(layer,left,young_right,lam,live_indices,state,rows,Q,scores if int(Q.shape[1]) else None,weighted if int(Q.shape[1]) else None,c15,hold)\n"
read=read.replace(anchor,addition+anchor);body=body[:start]+read+body[end:]
K=D/'kernel_energy23_HELD_v1.py';L=D/'fragment_lifecycle_HELD_v1.py';suffix='\n'+K.read_text()+'\n'+L.read_text();S=D/'candidate_energy23_HELD_v1.py';S.write_text(body+suffix)
assert (S.read_text()[:-len(suffix)].replace(addition,'').replace(new,old))==p
info=dict(parent_path=str(P),parent_sha256=sha(P),candidate_path=str(S),candidate_sha256=sha(S),kernel_path=str(K),kernel_sha256=sha(K),lifecycle_path=str(L),lifecycle_sha256=sha(L),source_inverse_exact=True,feature_line_count_preserved=True,source_edits=dict(feature_old=old,feature_new=new,read_addition=addition),fixed_scope='L7–14 changed RTT tail, feature23 pre-sqrt energy only; no coefficient/floor/rank tuning',zero_diagonal='subtract sum of per-pair diagonal squares',no_new_einsum_signatures=True)
(D/'SOURCE_MANIFEST_HELD_v1.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
