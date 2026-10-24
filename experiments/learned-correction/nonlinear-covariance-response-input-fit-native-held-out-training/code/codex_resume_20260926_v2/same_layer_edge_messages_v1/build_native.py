"""Append the fixed feature extension to S31, preserving the trained gain wrapper."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;H2=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mode,stage=sys.argv[1:];assert mode in ('message','control') and stage in ('zero','terminal')
protocol=json.loads((HERE/'protocol.json').read_text());cal=json.loads((HERE/'calibration.json').read_text())
guard=json.loads((HERE/'capture_guard.json').read_text())
assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
assert cal['protocol_sha256']==sha(HERE/'protocol.json')
stats=cal['statistics'][mode]
if stage=='zero':theta=np.zeros((2,12,16),dtype=np.float32)
else:
    guard=json.loads((HERE/f'fit_{mode}_guard.json').read_text())
    assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
    fit=HERE/f'fit_{mode}';result=json.loads((fit/'result.json').read_text());manifest=json.loads((fit/'manifest.json').read_text())
    assert result['completed'] and result['updates']==12 and result['terminal_sha256']==sha(fit/'terminal.npz')
    assert manifest['protocol_sha256']==sha(HERE/'protocol.json') and manifest['calibration_sha256']==sha(HERE/'calibration.json')
    with np.load(fit/'terminal.npz') as payload:
        theta=payload['theta'].copy()
        assert np.array_equal(payload['mean'],np.array(stats['mean'],dtype=np.float32))
        assert np.array_equal(payload['scale'],np.array(stats['scale'],dtype=np.float32))
assert theta.shape==(2,12,16) and np.isfinite(theta).all()
parent=H2/'candidate_old_both_tail_trace_bounded_v6.py';assert sha(parent)==protocol['parent_sha256']
source=parent.read_text();lines=source.splitlines()
node=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_message_edge_source'][0]
body='\n'.join(lines[node.lineno-1:node.end_lineno])
marker="    F=F/t['feature_scale'];";assert body.count(marker)==1
body=body.replace(marker,'    nf=_rnm_input(F,R,S,a,p,t)\n'+marker)
for side in ('left','right'):
    marker=f"    {side} = _e2_H(F @ t['{side}']) + _e2_H(messages @ context['{side}'])"
    assert body.count(marker)==1
    body=body.replace(marker,marker+f"\n    {side} = {side} + _e2_H(nf @ _RNM_CONST['{side}'])")
body=body.replace('def _message_edge_source(', 'def _gn_message_edge_source_base(',1)
literal=dict(mean=stats['mean'],scale=stats['scale'],left=theta[0].tolist(),right=theta[1].tolist())
helper='''
_RNM_CONST = None
def _rnm_input(F,R,T,a,p,edge):
    global _RNM_CONST
    if _RNM_CONST is None:
        _RNM_CONST={k:fnp.asarray(v,dtype=fnp.float32) for k,v in _RNM_LITERAL.items()}
    hold=_E2Hold()
    if _RNM_MODE=='control':
        fields=hold(F/edge['feature_scale'])[:,:12]
    else:
        phi=fnp.stack((p,hold(a*p),hold(hold(hold(a*a)-1)*p)),axis=1,dtype=fnp.float32)
        rr=R.copy();tt=T.copy()
        fnp.fill_diagonal(rr,0);fnp.fill_diagonal(tt,0)
        fields=fnp.concatenate((hold(hold(rr@phi)/32),hold(hold(rr*rr)@phi),
                                hold(hold(tt@phi)/32),hold(hold(tt.T@phi)/32)),axis=1)
    return fnp.clip(hold(hold(fields-_RNM_CONST['mean'])/_RNM_CONST['scale']),-8,8)
'''
out=H2/f'candidate_same_layer_edge_{mode}_{stage}_v1.py'
extension='\n\n# Fixed same-layer edge input pilot. All original parameters remain frozen.\n'
extension+=f'_RNM_MODE = {mode!r}\n_RNM_LITERAL = '+repr(literal)+'\n'+helper+'\n'+body+'\n'
assembled=source+extension;ast.parse(assembled)
with out.open('x') as fp:fp.write(assembled)
receipt=dict(mode=mode,stage=stage,candidate=str(out),sha256=sha(out),parent_sha256=sha(parent),
    calibration_sha256=sha(HERE/'calibration.json'),builder_sha256=sha(HERE/'build_native.py'),
    terminal_sha256=None if stage=='zero' else sha(HERE/f'fit_{mode}'/'terminal.npz'),
    unchanged_parent_prefix=True)
with (HERE/f'build_{mode}_{stage}.json').open('x') as fp:json.dump(receipt,fp,indent=2)
print(json.dumps(receipt))
