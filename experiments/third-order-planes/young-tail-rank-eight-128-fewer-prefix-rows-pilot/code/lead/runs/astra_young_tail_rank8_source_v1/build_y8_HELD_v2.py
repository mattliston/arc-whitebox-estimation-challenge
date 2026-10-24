from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent
parent_path=D.parent/'astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py'
parent=parent_path.read_bytes()
assert hashlib.sha256(parent).hexdigest()=='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
old=(D/'fragment_y8_HELD_v1.py').read_text()
before='    assert bool(hold(fnp.all(hold(fnp.isfinite(columns64)))))\n'
after='    if not bool(hold(fnp.all(hold(fnp.isfinite(columns64))))):\n        return hold(_rtp_qr_basis(vc, mu, hold)), True\n'
assert old.count(before)==1
fragment=old.replace(before,after)
candidate=parent+b'\n'+fragment.encode()
assert candidate.removesuffix(b'\n'+fragment.encode())==parent
for name,data in [('fragment_y8_HELD_v2.py',fragment.encode()),('candidate_y8_HELD_v2.py',candidate)]:
 (D/name).open('xb').write(data)
ready=json.loads((D/'SOURCE_BUILD_HELD_v1.json').read_text())
ready.update(candidate_sha256=hashlib.sha256(candidate).hexdigest(),fragment_sha256=hashlib.sha256(fragment.encode()).hexdigest(),amendment='Before any model: nonfinite sketch columns delegate to inherited physical basis, matching prospective fallback rule. No numerical outcomes used.',candidate_path=str(D/'candidate_y8_HELD_v2.py'),fragment_path=str(D/'fragment_y8_HELD_v2.py'))
(D/'SOURCE_BUILD_HELD_v2.json').open('x').write(json.dumps(ready,indent=2)+'\n')
print(json.dumps(ready))
