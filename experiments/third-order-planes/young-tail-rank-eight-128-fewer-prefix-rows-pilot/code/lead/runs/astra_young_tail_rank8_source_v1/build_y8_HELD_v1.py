from pathlib import Path
import base64,hashlib,json,zlib
import numpy as np
D=Path(__file__).resolve().parent
P=D.parent/'astra_hermite_group_view_repair_v1/candidate_hermite_group_views_QR_HELD_v2.py'
parent=P.read_bytes();assert hashlib.sha256(parent).hexdigest()=='1c913b92d6fd231593224d423dd0278be66390583bd68d6eb6c4b89eb1f79c9a'
probes=np.random.default_rng(20261015).standard_normal((1024,6)).astype('float32')
literal={'probes':((1024,6),base64.b64encode(zlib.compress(probes.tobytes())).decode())}
fragment=(D/'fragment_y8_template_HELD_v1.py').read_text().replace('__PROBE_LITERAL__',repr(literal))
candidate=parent+b'\n'+fragment.encode()
assert candidate.removesuffix(b'\n'+fragment.encode())==parent
for name,data in [('fragment_y8_HELD_v1.py',fragment.encode()),('candidate_y8_HELD_v1.py',candidate)]:
 (D/name).open('xb').write(data)
out=dict(status='SOURCE_FROZEN_PENDING_TINY_PEER',candidate_sha256=hashlib.sha256(candidate).hexdigest(),fragment_sha256=hashlib.sha256(fragment.encode()).hexdigest(),parent_path=str(P),parent_sha256=hashlib.sha256(parent).hexdigest(),source_inverse_exact=True,seed=20261015,probe_shape=[1024,6],probe_sha256=hashlib.sha256(probes.tobytes()).hexdigest(),scope='Only existing changed RTT reads; original c15==c2 bypass retained; c=max(32,oldc15-128); all layers otherwise unchanged.',basis='[vc,mu,(exact young reduced prefix)@six fixed probes], F64 column normalization and one QR; fixed numerical-rank fallback to original physical2 basis.',setup='One new6144elementF32 literal dictionary in existing _cs_build/_CS_DICTS ownership; no new PB tree keys.',models=0)
(D/'SOURCE_BUILD_HELD_v1.json').open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
