from pathlib import Path
import ast,base64,zlib,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
SUP=H2/'fourth_matrix_learned_supplier_v1';PARENT=H2/'candidate_old_both_tail_trace_bounded_v6.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(PARENT)=='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
assert not json.loads((SUP/'result.json').read_text())['passed']
assert json.loads((SUP/'verification.json').read_text())['passed']
frozen=json.loads((SUP/'model_frozen.json').read_text());payload={}
for key in ('model','normalizers'):
    path=Path(frozen[key]['path']);assert sha(path)==frozen[key]['sha256']
    with np.load(path,allow_pickle=False) as z:
        for k in z.files:
            assert k not in payload;payload[k]=z[k]
assert frozen['model']['sha256']=='e51e47c18af641fc906c5d681ab8df55af32c397338cdac7bfee21442e6f0b15'
packed={}
for key,value in payload.items():
    assert value.dtype==np.float32 and np.isfinite(value).all()
    packed[key]=(list(value.shape),base64.b64encode(zlib.compress(value.tobytes())).decode())
    assert np.frombuffer(zlib.decompress(base64.b64decode(packed[key][1])),np.float32).reshape(value.shape).tobytes()==value.tobytes()
s=PARENT.read_text();tree=ast.parse(s);changed=s;inverses=[];proof=[]
edits={
'_predict_learned_compact':[(
 '                           final_hook=scalar_terminal_hook,',
 '                           final_hook=scalar_terminal_hook, matrix_state_hook=(lambda: state),')],
'_predict_binary':[(
 'coefficient_hook=None, covariance_hook=None):',
 'coefficient_hook=None, covariance_hook=None, matrix_state_hook=None):'),(
 '            _PHYS_LAYER[0] = l',
 '''            _PHYS_LAYER[0] = l
            matrix31 = None
            matrix_gn_k4 = (_s19_k4u if _S19_GN_TORCH else k4)
            if _FM_ENABLED and 8 <= l <= 14:
                if matrix_state_hook is None:
                    raise RuntimeError("Missing causal fourth-matrix state")
                matrix4, matrix31, matrix22 = _fm_moments(l,mu,Cpre,K21,radial,w32,matrix_state_hook())
                k4 = k4 + matrix4
                k22 = k22 + matrix22
                del matrix4, matrix22'''),(
 '            if l in _T31_LAYERS:',
 '''            if matrix31 is not None:
                if l in _T31_LAYERS:
                    matrix31 = matrix31 + _t31_extra(l,locals())
                m, C, target, radial_new, aux = pair_map(mu,Cpre,K21,k4,k22,metric,extra31=matrix31)
            elif l in _T31_LAYERS:'''),(
 '(_s19_k4u if _S19_GN_TORCH else k4), aux))',
 '(matrix_gn_k4 if matrix31 is not None else (_s19_k4u if _S19_GN_TORCH else k4)), aux))'),(
 '            host = m',
 '            del matrix31, matrix_gn_k4\n            host = m')]
}
for name,pairs in edits.items():
    node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
    old=ast.get_source_segment(s,node);new=old
    for a,b in pairs:assert new.count(a)==1,(name,a);new=new.replace(a,b,1)
    back=new
    for a,b in reversed(pairs):assert back.count(b)==1;back=back.replace(b,a,1)
    assert back==old and changed.count(old)==1;changed=changed.replace(old,new,1)
    inverses.append((old,new));proof.append(dict(function=name,edits=pairs,exact_inverse=True))
hp=P2/'codex_resume_20260919_v1/joint_incoming_moment_head_v1/head.py'
node=next(n for n in ast.parse(hp.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='features')
helper=ast.get_source_segment(hp.read_text(),node).replace('def features(', 'def _fm_features(',1)
extra='\n_FM_LITERAL='+repr(packed)+'\n_FM_PARAMETERS=None\n_FM_ENABLED=True\n'+helper+'\n'+(HERE/'fragment_v2.py').read_text()
candidate=changed+extra;ast.parse(candidate);back=candidate[:-len(extra)]
for old,new in reversed(inverses):assert back.count(new)==1;back=back.replace(new,old,1)
assert back==s
for name,source in [('candidate_v2.py',candidate),('disabled_v2.py',candidate+'\n_FM_ENABLED=False\n')]:
    with (HERE/name).open('x') as f:f.write(source)
record=dict(parent_sha256=sha(PARENT),candidate_sha256=sha(HERE/'candidate_v2.py'),disabled_sha256=sha(HERE/'disabled_v2.py'),
            model=frozen['model'],normalizers=frozen['normalizers'],all_parameter_bytes_exact=True,exact_source_inverse=True,
            feature_source_sha256=sha(hp),fragment_sha256=sha(HERE/'fragment_v2.py'),proof=proof,
            original_proxy_gate_passed=False,separate_endpoint_followup=True,radial31_overlap_removed=True)
with (HERE/'build_v2.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps({k:v for k,v in record.items() if k!='proof'}),flush=True)
