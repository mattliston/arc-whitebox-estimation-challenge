"""Export one frozen per-layer nonlinear student while preserving message cache identity."""
import fit_frontier_v1 as f
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    for key in ('fit','base-science','base-release','out'):ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    report=json.loads(a.fit.read_text());assert report['status']=='NONLINEAR_TEACHER_FIT_NO_NETWORK_SCORE'
    spec=report['source_spec'];assert f.sha(a.base_science)==spec['science']['parent_sha256']
    assert f.sha(a.base_release)==spec['release']['parent_sha256']
    oldtail=Path(spec['tail']);assert f.sha(oldtail)==spec['tail_sha256']
    literals,functions=f.literals_and_functions(a.base_science.read_text())
    oldlit,_=f.literals_and_functions(oldtail.read_text());Q=f.decode(oldlit['_GNN_LITERAL'])
    layers={int(r['layer']):r['parameters'] for r in report['layers']};assert set(layers)==set(range(1,16))
    for values in layers.values():
        params=f.decode(values)
        assert set(params)==set(('enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b'))
        assert all(params[k].shape==Q[k].shape for k in params)
    step=functions['_gop2_gnn_step']
    insert="    c = _e2_gnn_layer(k, l)"
    assert step.count(insert)==1
    step=step.replace(insert,insert+"\n    _vmnl = _vmfr_nonlinear_parameters(l)")
    for prefix in ('enc0','enc2','enc4'):
        assert step.count("k['%s_wT']"%prefix)==1 and step.count("P['%s_b']"%prefix)==1
        step=step.replace("k['%s_wT']"%prefix,"_vmnl['%s_wT']"%prefix).replace("P['%s_b']"%prefix,"_vmnl['%s_b']"%prefix)
    assert "k = _e2_gnn_consts(P," in step and "M = _e2_messages(k, W, state)" in step
    tail=oldtail.read_text()+'\n# Fixed target-free nonlinear refinement, layer-specific encoder weights.\n'
    tail+='assert _GNN_PARAMS is None and not _E2_GNN_STATE\n'
    tail+='_VMFR_NL_LITERALS = '+repr(layers)+'\n_VMFR_NL_CACHE = {}\n'
    tail+='''
def _vmfr_nonlinear_parameters(layer):
    got = _VMFR_NL_CACHE.get(layer)
    if got is None:
        got = _unpack_static_fp32(_VMFR_NL_LITERALS[layer])
        for name in ('enc0', 'enc2', 'enc4'):
            got[name + '_wT'] = got[name + '_w'].T
        _VMFR_NL_CACHE[layer] = got
    return got
'''
    tail+='\n'+step+'\n_FAST_RESCUE_FRAC = {}\n'
    name='enc64_msg8_nonlinear_teacher';tp=a.out/('tail_'+name+'_v1.py');f.write(tp,tail)
    record={k:v for k,v in spec.items() if k not in ('name','science','release','tail','tail_sha256','teacher_errors')}
    record.update(name=name,kind='per_layer_nonlinear_teacher_refinement',tail=str(tp),tail_sha256=f.sha(tp),
                  nonlinear_fit_sha256=f.sha(a.fit),nonlinear_layers=2,layer_parameter_count=15,
                  message_cache_identity_preserved=True,teacher_fit_metrics=[{k:v for k,v in r.items() if k!='parameters'} for r in report['layers']],
                  qualification='NOT NATIVE TESTED; teacher-input fidelity does not qualify propagated estimator accuracy')
    for role,parent in [('science',a.base_science),('release',a.base_release)]:
        source=parent.read_text()+tail;p=a.out/('candidate_'+name+('_sci_v1.py' if role=='science' else '_rel_v1.py'))
        compile(source,str(p),'exec');f.write(p,source);record[role]=dict(path=str(p),sha256=f.sha(p),parent_sha256=f.sha(parent))
    result=dict(utc=f.utc(),status='IMMUTABLE_NONLINEAR_TEACHER_CANDIDATE_NO_NETWORK_SCORE',candidates=[record],
                reference_submission=337393,fit_sha256=f.sha(a.fit),source_sha256=f.sha(__file__))
    f.jwrite(a.out/'CANDIDATES_v1.json',result)
    print(json.dumps(dict(manifest=str(a.out/'CANDIDATES_v1.json'),science=record['science'],release=record['release']),indent=2))


if __name__=='__main__':main()
