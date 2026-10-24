"""One five-epoch current-GNN warm fit on fresh angular teacher-forced rows."""
import base64,gc,time,zlib
from common import *

def main():
    assert len(sys.argv)==3
    pp,c=plan(sys.argv[1],sys.argv[2]);environment(c,torch=True);rows,receipts,prerequisites=capture_inputs(c,pp)
    passive,passive_pin,_=passive_inputs(c,pp)
    from gnn import GNN,runtime,payload,tensors,loss,np,torch
    runtime();initial=payload(checked(passive['parameters']));model=GNN(initial)
    assert set(k for k,v in model.p.items() if v.requires_grad)==set(TRAINABLE)
    fixtures={r['rank']:r for r in c['data']};data={}
    for row in rows:
        rank=row['rank'];assert row['weights']==fixtures[rank]['weights'] and row['truth']==fixtures[rank]['truth']
        arrays=payload(checked(row['packet']))
        with np.load(checked(fixtures[rank]['truth']),allow_pickle=False) as z:truth=z['mc_post'].astype(np.float64)
        target=(truth-arrays['host_gaussian']).astype(np.float32)
        assert np.isfinite(target).all() and np.isfinite(arrays['x']).all() and np.isfinite(arrays['sigma_gaussian']).all()
        data[rank]=dict(x=arrays['x'],sigma_gaussian=arrays['sigma_gaussian'],target=target)
    def row_tensors(rank):
        W=np.load(checked(fixtures[rank]['weights']),allow_pickle=False)
        value=tensors(data[rank],W,gaussian=True)
        return (*value,torch.from_numpy(np.array(data[rank]['target'],copy=True)))
    def evaluate(ranks):
        model.eval();values=[]
        with torch.no_grad():
            for rank in ranks:
                x,s,W,t=row_tensors(rank);pred=model(x,s,W)
                assert torch.isfinite(pred).all()
                mse=((pred-t)**2).mean(1).numpy()
                values.append(dict(rank=rank,loss=float(loss(pred,t,model)),final_mse=float(mse[-1]),layer_mse=[float(v) for v in mse]))
                del x,s,W,t,pred
        model.train();return dict(rows=values,loss=sum(v['loss'] for v in values)/len(values),final_mse=sum(v['final_mse'] for v in values)/len(values))
    start=time.monotonic();before=evaluate(c['heldout_ranks'])
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=OPT['lr'],weight_decay=OPT['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=OPT['epochs']*len(c['fit_ranks']),eta_min=OPT['lr_min'])
    rng=np.random.default_rng(OPT['seed']);epochs=[];updates=0
    for epoch in range(OPT['epochs']):
        total=0.
        for rank in rng.permutation(c['fit_ranks']).tolist():
            x,s,W,t=row_tensors(rank);pred=model(x,s,W);objective=loss(pred,t,model)
            assert torch.isfinite(objective)
            optimizer.zero_grad();objective.backward()
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),OPT['clip_norm']);assert torch.isfinite(norm)
            optimizer.step();scheduler.step();total+=float(objective.detach());updates+=1
            del x,s,W,t,pred,objective
        epochs.append(dict(epoch=epoch+1,train_loss=total/32));print(json.dumps(epochs[-1]),flush=True);gc.collect()
    assert updates==160
    after=evaluate(c['heldout_ranks']);final={k:v.detach().numpy().copy() for k,v in model.p.items()}
    assert all(np.isfinite(v).all() for v in final.values())
    for k in FROZEN:assert final[k].tobytes()==initial[k].tobytes()
    admitted=after['final_mse']<before['final_mse'];export=None
    # Single fixed checkpoint; heldout loss does not select epochs or coefficients.
    payload_path=D/'final_checkpoint.npz'
    with payload_path.open('xb') as f:np.savez(f,**final)
    if admitted:
        literal={k:(tuple(v.shape),base64.b64encode(zlib.compress(v.astype(np.float32).tobytes(),9)).decode()) for k,v in final.items()}
        source,proof=edit_literal(checked(c['candidate']).read_text(),literal);path=D/'candidate.py'
        with path.open('x') as f:f.write(source)
        export=dict(candidate=pin(path),source_inverse=proof,frozen_parameter_bytes_exact=True,only_gnn_parameters_changed=True)
    result=dict(completed=True,passed_integrity=True,protocol=pp,trainer=pin(__file__),gnn=pin(D/'gnn.py'),source=c['candidate'],parameters=passive['parameters'],passive_complete=passive_pin,capture_receipts=receipts,prerequisites=prerequisites,fit_ranks=c['fit_ranks'],heldout_ranks=c['heldout_ranks'],optimizer=OPT,updates=updates,epochs=epochs,baseline_heldout=before,fitted_heldout=after,heldout_final_mse_change_pct=100*(after['final_mse']/before['final_mse']-1),endpoint_admitted=admitted,export=export,checkpoint=pin(payload_path),wall_s=time.monotonic()-start,torch_version=torch.__version__,numpy_version=np.__version__,teacher_forced_fixed_physics_rows=True,end_to_end_physics_gradient=False,bank_qualified=False,native_fit4_check8_resources_full40_and_serving_required=True)
    once(D/'fit_result.json',result);print(json.dumps({k:result[k] for k in ('completed','updates','heldout_final_mse_change_pct','endpoint_admitted','export','wall_s')}),flush=True)

if __name__=='__main__':main()
