"""First hidden128 preserved; second hidden128->64, preserve the current message8 function.
One fixed train4 selector/output ridge, separate teacher-input validation3.
"""
import fit_frontier_v1 as f
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    for key in ('captures','message-manifest','parent-science','parent-release','composed-parent-tail','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args()
    assert f.sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
    assert f.sha(a.parent_science)=='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
    assert f.sha(a.parent_release)=='e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
    a.out.mkdir(parents=True,exist_ok=False)
    f.jwrite(a.out/'PREREG_v1.json',dict(utc=f.utc(),source_sha256=f.sha(__file__),train=f.TRAIN,valid=f.VALID,
       first_hidden=128,second_hidden=64,message_width=8,ridge=f.RIDGE,selection='Train variance times outgoing squared norm; fixed64',
       target='Frozen current message8 state64 reconstructed on immutable teacher features; no network truth',changes='Retain current first hidden and selected second features exactly; refit only final affine output',
       validation_inputs='Separate teacher-validation3; off-policy',reference_submission=337393,no_native_gain_claim=True))
    message=json.loads(a.message_manifest.read_text());spec=next(c for c in message['candidates'] if c['name']=='msg8_teacher')
    assert spec['science']['sha256']=='23ff032f6816b013de0556a04ce57ef3780470755c1d36f815367abc032eb4dd'
    literals,_=f.literals_and_functions(a.parent_science.read_text());Q=f.decode(literals['_GNN_LITERAL'])
    rows,paths,meta=f.collect(a.captures);P=f.decode(meta['gnn_literal'])
    for key in ('enc2_w','enc2_b','enc4_w','enc4_b','head','head_b','rms','gain'):
        assert np.array_equal(P[key],Q[key]),('unexpected parent change',key)
    assert Q['enc0_w'].shape==(128,78) and Q['enc2_w'].shape==(128,128) and Q['enc4_w'].shape==(64,128)
    keep=[np.asarray(x,dtype=int) for x in spec['retained_by_power']]
    cols=np.concatenate([np.arange(30)]+[30+16*g+keep[g%3] for g in range(6)])
    datasets={}
    for label,ranks in [('train',f.TRAIN),('valid',f.VALID)]:
        rr=[r for r in rows if r['rank'] in ranks];data=f.matrices(rr,P)
        z=data['H'][:,cols]@Q['enc0_w'].T+data['bias']+Q['enc0_b']-P['enc0_b']
        h1=f.silu(z);h2=f.silu(h1@Q['enc2_w'].T+Q['enc2_b'])
        datasets[label]=dict(rows=rr,h2=h2,target=data['state'],target_msg8=h2@Q['enc4_w'].T+Q['enc4_b'])
    retained,selection_scores=f.choose(datasets['train']['h2'],Q['enc4_w'],64)
    w4,b4=f.ridge_affine(datasets['train']['h2'][:,retained],datasets['train']['target_msg8'])
    new={k:x.copy() for k,x in Q.items()}
    new.update(enc2_w=Q['enc2_w'][retained],enc2_b=Q['enc2_b'][retained],enc4_w=w4.T,enc4_b=b4)
    stats={}
    for label,data in datasets.items():
        before=data['h2']@Q['enc4_w'].T+Q['enc4_b']
        after=data['h2'][:,retained]@w4+b4
        stats[label]=dict(before=f.state_error(data['rows'],before,data['target'],P),after=f.state_error(data['rows'],after,data['target'],P),
            fidelity_to_current_msg8=f.state_error(data['rows'],after,data['target_msg8'],P))
    for key in Q:
        if key not in ('enc2_w','enc2_b','enc4_w','enc4_b'):assert np.array_equal(Q[key],new[key])
    detail=dict(kind='second_hidden_only_current_msg8_self_distillation',first_hidden=128,second_hidden=64,message_width=8,
        retained_first=list(range(128)),retained_second=retained.tolist(),retained_by_power=spec['retained_by_power'],
        selection_scores=selection_scores.tolist(),teacher_errors=stats,
        original_message_science_sha256=spec['science']['sha256'],loader_revert_sha256='67ec471178d413c61112268c050bcaeb23434d252edab2f4abe47262264a567b',
        fixed_geometry_bill_note='Approx0.16pp vs337393 before effects; native bill required',
        qualification='NOT NATIVE TESTED; held25 and original32 required. No accuracy inference from teacher fit.')
    rec=f.export(a.out,'msg8_second64_selfteacher',dict(science=a.parent_science,release=a.parent_release),new,'','',detail)
    actual_tail=Path(rec['tail'])
    composed=a.out/'composed_tail_msg8_second64_selfteacher_v1.py'
    f.write(composed,a.composed_parent_tail.read_text()+actual_tail.read_text())
    rec.update(append_tail=str(actual_tail),append_tail_sha256=rec['tail_sha256'],tail=str(composed),tail_sha256=f.sha(composed))
    f.jwrite(a.out/'CANDIDATES_v1.json',dict(utc=f.utc(),status='IMMUTABLE_SECOND_HIDDEN_ONLY_NO_NETWORK_SCORE',candidates=[rec],
        source_sha256=f.sha(__file__),capture_files=paths,train=f.TRAIN,valid=f.VALID,reference_submission=337393))
    print(json.dumps(dict(candidate=rec['name'],science=rec['science'],release=rec['release'],
       valid_before=stats['valid']['before']['local_head_delta_rms'],valid_after=stats['valid']['after']['local_head_delta_rms'],valid_fidelity_to_msg8=stats['valid']['fidelity_to_current_msg8']['local_head_delta_rms']),indent=2))

if __name__=='__main__':
    main()

