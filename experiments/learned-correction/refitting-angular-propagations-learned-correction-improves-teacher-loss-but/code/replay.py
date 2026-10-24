"""Target-free CPU Torch replay of the actual passive native C64 trajectory."""
from common import *

def main():
    assert len(sys.argv)==3
    pp,c=plan(sys.argv[1],sys.argv[2]);environment(c,torch=True);native_prerequisite(c)
    r,rp,gp=passive_inputs(c,pp)
    from gnn import GNN,runtime,payload,tensors,np,torch
    runtime();p=payload(checked(r['parameters']));model=GNN(p).eval()
    row=r['rows'][0];arrays=payload(checked(row['packet']));W=np.load(checked(c['qualification_fixture']['weights']),allow_pickle=False)
    with torch.no_grad():pred,hidden=model(*tensors(arrays,W),return_states=True)
    actual={'delta':pred.numpy(),'hidden':hidden.numpy()};stats=[]
    for name in actual:
        for l in range(16):
            a=actual[name][l].astype(np.float64);b=arrays[name][l].astype(np.float64);diff=a-b
            rms=float(np.sqrt(np.mean(b*b)));err=float(np.sqrt(np.mean(diff*diff)))
            relative=err/max(rms,1e-10);limit=c['replay_gates'][name+'_relative_rms_max']
            passed=bool(np.isfinite(a).all() and relative<=limit)
            stats.append(dict(field=name,layer=l,rms_native=rms,rms_error=err,max_abs_error=float(np.max(np.abs(diff))),relative_rms_error=relative,passed=passed))
    result=dict(passed=all(x['passed'] for x in stats),plan=pp,source=pin(__file__),gnn=pin(D/'gnn.py'),passive_complete=rp,passive_guard=gp,parameters=r['parameters'],metrics=stats,torch_version=torch.__version__,numpy_version=np.__version__,device='cpu',targets_accessed=False,torch_byte_parity_required=False,scope='Current passive angular x and recurrent GNN only; teacher-forced acquisition changes x but uses this same GNN.')
    once(D/'replay_result.json',result);print(json.dumps(result),flush=True)
    assert result['passed'],'Native/Torch GNN replay did not qualify; no acquisition or fit.'

if __name__=='__main__':main()
