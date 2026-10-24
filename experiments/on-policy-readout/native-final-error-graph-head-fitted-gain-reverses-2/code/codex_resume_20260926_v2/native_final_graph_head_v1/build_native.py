"""Prospective graph builder; refuses execution unless graph is the fixed survivor."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import zlib
import numpy as np

HERE=Path(__file__).resolve().parent;H2=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((HERE/'protocol.json').read_text())
assessment=json.loads((HERE/'assessment.json').read_text())
survivor=json.loads((HERE/'survivor.json').read_text())
assert survivor['survivor']=='graph' and survivor['assessment_sha256']==sha(HERE/'assessment.json')
assert assessment['protocol_sha256']==sha(HERE/'protocol.json')
assert not assessment['valid40_accessed']
guard=json.loads((HERE/'assessment_guard.json').read_text())
assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
fit=HERE/'fit_graph';result=json.loads((fit/'result.json').read_text())
assert result['weights_sha256']==sha(fit/'weights.npz')
parent=H2/'candidate_old_both_tail_trace_bounded_v6.py'
assert sha(parent)==protocol['parent_sha256']
shapes={'mean':(151,),'scale':(151,),'layers.0.weight':(512,151),'layers.0.bias':(512,),
        'layers.2.weight':(256,512),'layers.2.bias':(256,),'layers.4.weight':(1,256),'layers.4.bias':(1,)}
with np.load(fit/'weights.npz') as z:
    assert set(z.files)==set(shapes)
    arrays={name:np.array(z[name],dtype=np.float32,order='C') for name in z.files}
for name,shape in shapes.items():assert arrays[name].shape==shape and np.isfinite(arrays[name]).all(),name
assert (arrays['scale']>=protocol['training']['feature_scale_floor']).all()
arrays['output_scale']=np.asarray([protocol['model']['output_scale']],dtype=np.float32)
encoded={name:(a.shape,base64.b64encode(zlib.compress(a.tobytes(),9)).decode('ascii')) for name,a in arrays.items()}
helper=HERE/'native_helper_draft.py'
source=parent.read_text()+'\n\n'+helper.read_text()+'\n\n_NF_LITERAL = '+repr(encoded)+'\n'
ast.parse(source)
candidate=H2/'candidate_native_final_graph_head_v1.py'
with candidate.open('x') as f:f.write(source)
record=dict(candidate=str(candidate),sha256=sha(candidate),parent_sha256=sha(parent),append_only=True,
            helper_sha256=sha(helper),weights_sha256=sha(fit/'weights.npz'),
            protocol_sha256=sha(HERE/'protocol.json'),assessment_sha256=sha(HERE/'assessment.json'),
            survivor_sha256=sha(HERE/'survivor.json'),
            first15_outputs_intended_unchanged=True,setup_intended_unchanged=True,
            metered_native_qualified=False,valid40_accessed=False,submission_qualified=False)
with (HERE/'native_build.json').open('x') as f:json.dump(record,f,indent=2,allow_nan=False)
print(json.dumps(record,indent=2))
