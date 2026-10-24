from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent
parent=D.parent/'opus_fix336687_v1/candidate_v61j.py'
fragment=D/'fragment_pair_rotation_v1.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='ac93b10f2c8fa541b94cf0787b7f573d36cf1233df219d00a22335c4f9a47129'
out=D/'candidate_v61j_pair_rotation_HELD_v1.py'
text=parent.read_text()+'\n\n'+fragment.read_text()
ast.parse(text)
with out.open('x') as f:f.write(text)
receipt=dict(parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),fragment=str(fragment),fragment_sha256=hashlib.sha256(fragment.read_bytes()).hexdigest(),candidate=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),status='HELD_SOURCE_ONLY',mechanism='Exact two-channel rotation of young-read operands only; original prefix/suffix and old read preserved; <=32-row sampled power-of-two balancing. No channels dropped.')
(D/'PAIR_ROTATION_BUILD_v1.json').open('x').write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
