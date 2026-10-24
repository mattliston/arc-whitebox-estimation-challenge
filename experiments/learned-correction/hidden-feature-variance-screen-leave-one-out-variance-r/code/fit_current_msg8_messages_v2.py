"""Two predeclared message4/6 students of actual current-msg8 trajectories.

Fixed train4 selection/ridge, separate validation3. No network truth is read.
Only current777a native H/bias/state/messages/sigma captures are accepted.
"""
import fit_frontier_v1 as f
import argparse
import ast
import gzip
import json
from pathlib import Path
import numpy as np
SCI='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
REL='e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05'
WIDTHS=(4,6)
HOOK='07ae3f9ff87827ab4bb569de929af9a3ded296730d8b46be23b407bf677b856a'

def collect(root):
    complete=json.loads((root/'COMPLETE.json').read_text());assert complete['successful']
    rows=[];files=[];meta=None
    for group in (1,2,3):
        name='part%02d'%group
        guard=json.loads((root/(name+'_GUARD.json')).read_text())
        assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit']==0
        folder=root/name/'gnn_current_msg8_v1'
        current=json.loads((folder/'METADATA_v2.json').read_text())
        assert current['source_sha256']==SCI and current['hook_sha256']==HOOK
        assert current['H_width']==78 and current['message_width']==8 and current['first_width']==128
        receipt=json.loads((folder/'CAPTURE_COMPLETE_v2.json').read_text())
        assert receipt['complete'] and receipt['source_sha256']==SCI and receipt['native_handles_retained'] is False
        if meta is None:meta=current
        for key in ('gnn_literal','hc_u','hcf_u'):assert current[key]==meta[key]
        for entry in receipt['files']:
            path=folder/entry['path'];assert f.sha(path)==entry['sha256']
            rec=json.load(gzip.open(path,'rt'))
            assert rec['phase']=='scored' and rec['rank'] in f.TRAIN+f.VALID
            assert rec['rows']==list(range(rec['rank']%8,1024,8))
            rows.append(rec);files.append(dict(path=str(path),sha256=f.sha(path)))
    keys=[(r['rank'],r['layer']) for r in rows]
    assert len(set(keys))==len(keys)==105
    assert set(keys)=={(r,l) for r in f.TRAIN+f.VALID for l in range(1,16)}
    return rows,files,meta

def matrices(rows,P):
    H=np.concatenate([f.array(r,'H') for r in rows])
    bias=np.concatenate([np.broadcast_to(f.array(r,'bias'),(128,128)) for r in rows])
    assert H.shape[1]==78 and P['enc0_w'].shape==(128,78)
    h1=f.silu(H@P['enc0_w'].T+bias)
    h2=f.silu(h1@P['enc2_w'].T+P['enc2_b'])
    state=h2@P['enc4_w'].T+P['enc4_b']
    saved_h2=np.concatenate([f.array(r,'hidden2') for r in rows])
    saved_state=np.concatenate([f.array(r,'state') for r in rows])
    np.testing.assert_allclose(h2,saved_h2,atol=5e-5,rtol=3e-5)
    np.testing.assert_allclose(state,saved_state,atol=5e-5,rtol=3e-5)
    messages=np.concatenate([f.array(r,'messages').reshape(128,24) for r in rows])
    source=np.concatenate([f.array(r,'source') for r in rows])
    assert source.shape[1]==24
    return dict(H=H,bias=bias,h1=h1,h2=h2,state=saved_state,messages=messages,source=source,
                replay_h2_max_abs=float(np.max(np.abs(h2-saved_h2))),replay_state_max_abs=float(np.max(np.abs(state-saved_state))))

def main():
    ap=argparse.ArgumentParser()
    for key in ('captures','parent-science','parent-release','composed-parent-tail','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();assert f.sha(f.__file__)=='b9bc4ed11d70170db6d3909bd670e97616c38e8ab455bdc99ca80513c9faf363'
    assert f.sha(a.parent_science)==SCI and f.sha(a.parent_release)==REL
    a.out.mkdir(parents=True,exist_ok=False)
    f.jwrite(a.out/'PREREG_v1.json',dict(utc=f.utc(),source_sha256=f.sha(__file__),message_widths=WIDTHS,
        train=f.TRAIN,valid=f.VALID,ridge=f.RIDGE,teacher_source_sha256=SCI,hook_sha256=HOOK,
        teacher_trajectory='ACTUAL777a; not old EH14-input replay',no_network_truth=True,no_validation_selection=True,
        fit='Shared first-affine preactivation ridge plus current learned-edge message-context ridge',
        selection='Train H variance times current first-encoder outgoing squared weights; two predeclared widths',
        preserved='All128 nonlinear features, state64, current head/HC/HCF and non-message edge branches',
        reference_submission=337421,qualification='Native full32 plus exact777a and337421 controls required'))
    rows,paths,meta=collect(a.captures)
    literals,functions=f.literals_and_functions(a.parent_science.read_text());P=f.decode(literals['_GNN_LITERAL'])
    captured=f.decode(meta['gnn_literal'])
    assert set(P)==set(captured)
    for key in P:assert np.array_equal(P[key],captured[key]),key
    assert np.array_equal(np.asarray(literals['_HC_U']),np.asarray(meta['hc_u']))
    assert np.array_equal(np.asarray(literals['_HCF_U']),np.asarray(meta['hcf_u']))
    train_rows=[r for r in rows if r['rank'] in f.TRAIN];valid_rows=[r for r in rows if r['rank'] in f.VALID]
    train=matrices(train_rows,P);valid=matrices(valid_rows,P)
    special={}
    for node in ast.parse(a.parent_science.read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ('_MESSAGE_EDGE_LITERAL','_VMFR_ORIGINAL_W3_ENCODER'):
                special[node.targets[0].id]=node.value
    special={k:ast.literal_eval(node) for k,node in special.items()}
    edge={k:np.asarray(v,dtype='f4') for k,v in special['_MESSAGE_EDGE_LITERAL'].items()}
    originalE=np.asarray(special['_VMFR_ORIGINAL_W3_ENCODER'],dtype='f4')
    assert originalE.shape==(128,8) and all(v.shape==(24,14) for v in edge.values())
    assert '_VMFR_CM' in functions['_gop2_message_product'] and '_VMFR_CM' in functions['_g5_message_edge_base']
    records=[]
    for width in WIDTHS:
        keep=[];scores=[]
        for group in range(3):
            score=np.zeros(8)
            for block in (group,group+3):
                cols=slice(30+8*block,30+8*(block+1))
                score+=train['H'][:,cols].astype('f8').var(axis=0)*np.sum(P['enc0_w'][:,cols].astype('f8')**2,axis=0)
            keep.append(np.sort(np.argsort(-score,kind='stable')[:width]));scores.append(score.tolist())
        midx=np.concatenate([8*g+keep[g] for g in range(3)])
        hidx=np.concatenate([np.arange(30)]+[30+8*g+keep[g%3] for g in range(6)])
        w0,b0=f.ridge_affine(train['H'][:,hidx],train['H']@P['enc0_w'].T)
        Q={k:v.copy() for k,v in P.items()}
        Q.update(msg_w=P['msg_w'][midx],msg_b=P['msg_b'][midx],enc0_w=w0.T,enc0_b=P['enc0_b']+b0)
        mt=np.clip(train['messages'],-8,8).astype('f8');xx=mt[:,midx]
        gram=xx.T@xx+f.RIDGE*len(xx)*np.eye(len(midx))
        edge_new={k:np.linalg.solve(gram,xx.T@(mt@value)).astype('f4') for k,value in edge.items()}
        final=f.decode(literals['_FINAL_LITERAL']);assert final['w0'].shape[1]==78
        final['w0']=final['w0'][:,hidx]
        extra='_VMFR_CM = '+str(width)+'\n_MESSAGE_EDGE_LITERAL = '+repr({k:v.tolist() for k,v in edge_new.items()})+'\n'
        extra+='_FINAL_LITERAL = '+repr(f.encode(final))+'\n'
        code=f.metric_function(originalE[:,keep[2]])
        stats={}
        for label,data,rr in [('train',train,train_rows),('valid',valid,valid_rows)]:
            z=data['H'][:,hidx]@w0+data['bias']+b0
            h2=f.silu(f.silu(z)@P['enc2_w'].T+P['enc2_b'])
            got=h2@P['enc4_w'].T+P['enc4_b']
            stats[label]=f.state_error(rr,got,data['state'],P)
            mm=np.clip(data['messages'],-8,8)
            stats[label]['edge_projection_relative_l2']={k:float(np.linalg.norm(mm[:,midx]@edge_new[k]-mm@edge[k])/max(np.linalg.norm(mm@edge[k]),1e-30)) for k in edge}
        detail=dict(kind='actual_current_msg8_message_self_distillation',message_width=width,
            retained_by_power=[k.tolist() for k in keep],selection_scores=scores,teacher_errors=stats,
            teacher_source_sha256=SCI,teacher_trajectory='ACTUAL777a native inputs and targets',
            approximation_note='GOP2 subspaces change under channel pruning; native propagated validation required',
            reference_submission=337421,no_native_gain_claim=True)
        name='msg%d_current_selfteacher'%width
        rec=f.export(a.out,name,dict(science=a.parent_science,release=a.parent_release),Q,extra,code,detail)
        append=Path(rec['tail']);composed=a.out/('composed_tail_'+name+'_v1.py')
        f.write(composed,a.composed_parent_tail.read_text()+append.read_text())
        rec.update(append_tail=str(append),append_tail_sha256=rec['tail_sha256'],tail=str(composed),tail_sha256=f.sha(composed))
        records.append(rec)
    report=dict(utc=f.utc(),status='CURRENT_MSG8_ACTUAL_TRAJECTORY_DISTILLATION_NO_NATIVE_SCORE',
        train=f.TRAIN,valid=f.VALID,capture_files=paths,capture_metadata=meta,candidates=records,
        replay_checks=dict(train={k:v for k,v in train.items() if k.startswith('replay')},valid={k:v for k,v in valid.items() if k.startswith('replay')}),
        source_sha256=f.sha(__file__),reference_submission=337421)
    f.jwrite(a.out/'CANDIDATES_v1.json',report)
    print(json.dumps(dict(utc=f.utc(),candidates=[dict(name=c['name'],science=c['science'],release=c['release'],
        teacher_valid_head_rms=c['teacher_errors']['valid']['local_head_delta_rms']) for c in records]),indent=2))
if __name__=='__main__':main()

