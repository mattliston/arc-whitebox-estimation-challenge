"""One fixed ridge correction; emit the only candidate without a replay gate."""
import ast
import resource
import time
import numpy as np
from common import *
from readout import replay
from source_edit import edit,decoded_payload,terminal_head


def solve(B,y,w):
    w=w/w.sum()
    scale=np.sqrt(np.sum(w[:,None]*B*B,axis=0))
    assert np.isfinite(scale).all() and np.min(scale)>1e-20
    X=B/scale
    gram=X.T@(w[:,None]*X)+RIDGE*np.eye(B.shape[1])
    rhs=X.T@(w*y)
    coefficient=np.linalg.solve(gram,rhs)
    residual=float(np.linalg.norm(gram@coefficient-rhs)/max(np.linalg.norm(rhs),1e-30))
    assert residual<1e-9
    return coefficient/scale,scale,dict(normal_equation_relative_error=residual,
                                      condition=float(np.linalg.cond(gram)))


def main():
    verify();guard('parity');guard('capture');start=time.monotonic()
    for entry in read(HERE/'fit_seal.json')['sources']:verify_pin(entry)
    captured=read(HERE/'capture/complete.json')
    assert captured['passed'] and captured['roster']==list(TRAIN) and not captured['targets_accessed']
    assert captured['source']==pin(COMPONENT) and captured['all_capture_references_retired']
    assert read(HERE/'parity/complete.json')['passed']
    # Small independent solver certificate, executed only by root after review.
    rng=np.random.Generator(np.random.PCG64DXSM(2026092062))
    B=rng.normal(size=(113,7));y=rng.normal(size=113);w=np.linspace(.1,1,113)
    got,scale,certificate=solve(B,y,w);wn=w/w.sum()
    independent=np.linalg.lstsq(np.vstack((np.sqrt(wn)[:,None]*B/scale,np.sqrt(RIDGE)*np.eye(7))),
                                np.r_[np.sqrt(wn)*y,np.zeros(7)],rcond=None)[0]/scale
    assert np.allclose(got,independent,rtol=1e-10,atol=1e-12)
    refs=read(HERE/'data_refs.json');data=[]
    for row,fixture in zip(captured['rows'],refs['train']):
        assert row['mini']==fixture['mini'] and row['identity']==fixture['identity']
        assert row['weights']==fixture['weights'] and not row['targets_accessed']
        verify_pin(row['packet'])
        with np.load(row['packet']['path'],allow_pickle=False) as z:
            packet={key:z[key].copy() for key in z.files}
        old=np.r_[packet['head'],packet['head_b']].astype(np.float64)
        prediction,delta=replay(packet,old)
        assert np.array_equal(delta,packet['delta'])
        assert np.array_equal(prediction.astype(np.float64),packet['prediction'][-1])
        physical_scale=(packet['sigma'].astype(np.float32)*np.float32(packet['rms'])).astype(np.float64)*packet['effective_gain'].astype(np.float64)
        basis=physical_scale[:,None]*np.column_stack((packet['hidden'],np.ones(1024))).astype(np.float64)
        data.append(dict(row=row,packet=packet,old=old,B=basis,
                         unclipped=packet['host'].astype(np.float64)+packet['delta'].astype(np.float64)))
    assert [d['row']['mini'] for d in data]==list(TRAIN)
    assert all(np.array_equal(d['old'],data[0]['old']) for d in data)
    once(HERE/'features_frozen_before_targets.json',dict(captures=pin(HERE/'capture/complete.json'),
          train=list(TRAIN),ridge=RIDGE,all65_retained=True,solver_certificate=certificate,
          independent_solver_max_error=float(np.max(np.abs(got-independent))),
          original_native_readout_replay_exact=True,targets_decoded_in_this_study=False,
          old_targets_already_used_by_checkpoint52=True))
    verify_pin(refs['target_manifest']);verify_pin(refs['target_packet'])
    target_manifest=read(refs['target_manifest']['path'])
    assert target_manifest['train']==list(TRAIN) and target_manifest['packet']==refs['target_packet']
    assert [r['weights'] for r in target_manifest['rows']]==[r['weights'] for r in refs['train']]
    assert [r['identity'] for r in target_manifest['rows']]==[r['identity'] for r in refs['train']]
    with np.load(refs['target_packet']['path'],allow_pickle=False) as z:
        assert tuple(z['train'])==TRAIN
        targets=z['targets'].astype(np.float64)
    assert targets.shape==(6,1024) and np.isfinite(targets).all()
    delta,scale,equations=solve(np.concatenate([d['B'] for d in data]),
                              np.concatenate([targets[i]-d['unclipped'] for i,d in enumerate(data)]),
                              np.concatenate([np.full(1024,d['row']['multiplier']) for d in data]))
    old=data[0]['old'];head=(old+delta).astype(np.float32)
    assert head.shape==(65,) and np.isfinite(head).all()
    model_path=HERE/'model.npz'
    with model_path.open('xb') as stream:np.savez(stream,head=head,delta=delta,old=old,column_scale=scale)
    candidate,disabled,patches,literal_old=edit(COMPONENT.read_text(),head)
    assert np.array_equal(np.asarray(literal_old,dtype=np.float32),old.astype(np.float32))
    assert disabled==COMPONENT.read_bytes()
    # Verify every original learned scalar except the terminal65 remains equal.
    before=decoded_payload(COMPONENT.read_text());after=decoded_payload(candidate)
    assert terminal_head(candidate)==[float(value) for value in head]
    for key,(shape,raw) in before.items():
        assert after[key][0]==shape
        prefix=15*64*4 if key=='head' else (15*4 if key=='head_b' else len(raw))
        assert after[key][1][:prefix]==raw[:prefix]
    with (HERE/'candidate.py').open('xb') as stream:stream.write(candidate)
    with (HERE/'disabled_control.py').open('xb') as stream:stream.write(disabled)
    diagnostics=[]
    for i,d in enumerate(data):
        prediction,_=replay(d['packet'],head)
        before_mse=float(np.mean((d['packet']['prediction'][-1]-targets[i])**2))
        after_mse=float(np.mean((prediction.astype(np.float64)-targets[i])**2))
        diagnostics.append(dict(mini=TRAIN[i],parent_mse=before_mse,fitted_replay_mse=after_mse,
                                not_an_advance_gate=True))
    once(HERE/'fit_result.json',dict(completed=True,fit_count=1,train=list(TRAIN),ridge=RIDGE,
          model=pin(model_path),candidate=pin(HERE/'candidate.py'),component=pin(COMPONENT),
          disabled_control=pin(HERE/'disabled_control.py'),source_byte_inverse_exact=True,
          allowed_parameter_count=65,all_other_learned_coefficients_exact=True,
          executable_ast_unchanged=True,added_inference_operations=0,all_other_decoded_parameter_bytes_exact=True,
          equations=equations,patches=patches,diagnostics=diagnostics,
          surrogate_selection=False,amplitude_sweep=False,actual_native_score_measured=False,
          targets=refs['target_packet'],elapsed_s=time.monotonic()-start,
          peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
    once(HERE/'candidates.json',dict(bank=pin(BANK),component=pin(COMPONENT),protocol=pin(HERE/'PROTOCOL.md'),
         arms=[dict(name='terminal_head',candidate=pin(HERE/'candidate.py'))]))
    print(__import__('json').dumps(dict(event='single_terminal_head_fit',candidate=pin(HERE/'candidate.py'),
          equations=equations,actual_score_pending=True,advance_by_generic_bank_gates_only=True)),flush=True)


if __name__=='__main__':main()
