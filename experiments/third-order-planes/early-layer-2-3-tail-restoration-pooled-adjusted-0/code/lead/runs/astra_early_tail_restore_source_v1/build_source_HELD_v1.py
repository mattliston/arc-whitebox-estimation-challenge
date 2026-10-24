from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent;P=D.parent/'astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py';F=D/'fragment_early_tail_restore_HELD_v1.py';C=D/'candidate_early_tail_restore_HELD_v1.py'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert h(P)=='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
s=P.read_text()+'\n\n'+F.read_text();C.write_text(s);assert s[:-len('\n\n'+F.read_text())]==P.read_text()
r=dict(status='SOURCE_FROZEN_PENDING_TINY',candidate=dict(path=str(C),sha256=h(C)),parent=dict(path=str(P),sha256=h(P)),fragment=dict(path=str(F),sha256=h(F)),exact_parent_prefix=True,zero_based_layers=[2,3],outer_c2_and_forms_unchanged=True,all_other_layers_parent_delegate=True,natural_D7_policy_unchanged=True,source_only=True,models=0)
(D/'SOURCE_BUILD_v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
