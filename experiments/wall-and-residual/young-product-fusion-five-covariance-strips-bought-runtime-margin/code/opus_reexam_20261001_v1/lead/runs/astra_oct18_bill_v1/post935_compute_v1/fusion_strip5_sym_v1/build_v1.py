import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent
P=D.parent/'fusion_strip5_v1/candidate_95c_fusion_strip5_HELD_v1.py'
source=P.read_text()
assert hashlib.sha256(P.read_bytes()).hexdigest()=='66fd76ceb1bb4ad8c90505d049a22dc794457ff041df1f8c1679d937697ac4f2'
fragment=(D/'fragment_v1.py').read_text()
candidate=D/'candidate_95c_fusion_strip5_sym_HELD_v1.py'
combined=source+'\n'+fragment
ast.parse(combined)
with candidate.open('x') as f:f.write(combined)
receipt=dict(status='HELD_SOURCE_ONLY_FIXED_WHOLE_STRIP_OUTPUT_SYMMETRIZATION',parent=str(P),parent_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),candidate=str(candidate),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    arithmetic='For the existing five selected strip outputs only: G <- float32(0.5*float32(G+G.T)); parent source retained exactly.',
    dtype='float32 operands, result and in-place scale; Python0.5 is the same scalar form used throughout parent.',
    added_arithmetic_flops_per_call=2*1024*1024,max_calls=5,added_flops_per_network=10*1024*1024,
    added_bill_percent_at_rank35=100*(10*1024*1024)/238836882388,
    nominal_public_calls_per_network=15,max_additional_numeric_owner_bytes=4*1024*1024,
    retained=['Identical fixed first-five896/960 strip selection','All fusion rules and output content','All existing PB649 order/classes and setup','All original owner sizes and covariance diagonal in exact arithmetic'],
    caveats=['No SDK/numerical test performed. No accuracy recovery claimed.','Added frees/aliases require actual request measurement;15 is the arithmetic/view-call count only.','For finite covariance values well belowfloat32overflow, diagonal self-averaging is exact; this is not an overflow-proof general operation.','Only local memories/handoff, TEAM_LOG and historical strip family source were searched; no exhaustive claim about unrelated experiments.'])
with (D/'BUILD_v1.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({k:receipt[k] for k in ('status','candidate_sha256','added_flops_per_network','added_bill_percent_at_rank35','nominal_public_calls_per_network')}))
