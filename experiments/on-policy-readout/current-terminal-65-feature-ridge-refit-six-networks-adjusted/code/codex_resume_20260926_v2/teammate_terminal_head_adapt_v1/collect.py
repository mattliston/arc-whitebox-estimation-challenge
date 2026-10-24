"""Root-only passive qualification/capture; no fitted or label arithmetic."""
import ctypes
import gc
import resource
import sys
import time
import types
import numpy as np
import flopscope as flops
from common import *
from readout import replay


def main():
    mode=sys.argv[1];assert mode in ('parity','capture')
    verify()
    if mode=='capture':
        guard('parity')
        assert read(HERE/'parity/complete.json')['passed']
    out=HERE/mode;out.mkdir()
    refs=read(HERE/'data_refs.json')
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    mod=types.ModuleType('terminal_head_capture_'+mode);mod.__file__=str(COMPONENT);sys.modules[mod.__name__]=mod
    exec(compile(COMPONENT.read_text(),str(COMPONENT),'exec'),mod.__dict__)
    original_final,original_step,original_layer=mod.final_scalar_features,mod.gnn_step,mod._e2_gnn_layer
    held={};calls=[];terminal_calls=[]
    def final(l,m,aux,w):
        value=original_final(l,m,aux,w)
        assert l==15 and 'host' not in held
        held['host']=m;terminal_calls.append(l)
        return value
    def layer(k,l):
        value=original_layer(k,l)
        if l==15:
            assert 'effective_gain' not in held
            held['effective_gain']=value['gain'];held['rms']=value['rms']
            held['head']=value['head'];held['head_b']=value['head_b']
        return value
    def step(l,x,sigma,w,state,p):
        value=original_step(l,x,sigma,w,state,p);calls.append(l)
        if l==15:
            assert 'hidden' not in held
            held.update(hidden=value[1],delta=value[0],sigma=sigma)
        return value
    estimator=mod.Estimator()
    estimator.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,
                    api_version='2',scratch_dir=None,submission_dir=None,seed=0))
    mod.final_scalar_features,mod.gnn_step,mod._e2_gnn_layer=final,step,layer
    controls={r['rank']:r for r in map(__import__('json').loads,(H2/'thin_star_route_train2_v1/rows.jsonl').read_text().splitlines())}
    relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
    relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
    records=[];started=time.monotonic()
    for item in fixture_rows(mode):
        assert not held and not calls and not terminal_calls
        identity=item['mini'] if mode=='capture' else item['rank']
        wp=Path(item['weights']['path'])
        if mode=='capture':verify_pin(item['weights'])
        weights=np.load(wp,allow_pickle=False)
        assert weights.shape==(16,1024,1024)
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=[weights[l] for l in range(16)],
                                  seed=0,name=item['identity']['mlp_name'] if mode=='capture' else item['mlp_name'])
        begin=time.monotonic();cpu0=time.process_time()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as context:
            result=estimator.predict(mlp,2**41)
            prediction=np.asarray(result,dtype=np.float64).copy()
            # Harness copies only. No new FNP arithmetic is called by wrappers.
            packet={key:np.asarray(value).copy() for key,value in held.items()}
            held.clear();del result
        bill=int(context.flops_used)
        assert calls==list(range(16)) and terminal_calls==[15],(calls,terminal_calls)
        assert prediction.shape==(16,1024) and np.isfinite(prediction).all() and np.min(prediction)>=0
        assert packet['hidden'].shape==(1024,64) and packet['head'].shape==(64,)
        assert set(packet)=={'host','effective_gain','rms','head','head_b','hidden','delta','sigma'}
        assert all(np.isfinite(value).all() for value in packet.values())
        head=np.r_[packet['head'],packet['head_b']].astype(np.float32)
        expected,delta=replay(packet,head)
        assert np.array_equal(delta,packet['delta'])
        assert np.array_equal(expected.astype(np.float64),prediction[-1])
        assert getattr(mod,'_W5_PORT_SCOPE',None) is None
        last=getattr(mod,'_W5_PORT_LAST',None)
        if last is not None:
            assert last['completed'] and last['numeric_cache_retired']
        row=dict(mini=identity if mode=='capture' else None,rank=identity if mode=='parity' else None,
                 weights=pin(wp),flops=bill,multiplier=max(.1,bill/2**41),
                 wall_s=time.monotonic()-begin,cpu_s=time.process_time()-cpu0,
                 terminal_readout_replay_exact=True,gnn_layers=calls.copy(),terminal_calls=terminal_calls.copy(),
                 capture_references_cleared=True,w5_scope_retired=True,
                 source=pin(COMPONENT),thin_stats=dict(getattr(mod,'_TSR_STATS',{})),
                 targets_accessed=False,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if mode=='parity':
            expected_path=H2/'thin_star_route_train2_v1'/f'pred_{identity}.npy'
            expected=np.load(expected_path,allow_pickle=False)
            assert prediction.tobytes()==expected.tobytes() and bill==controls[identity]['flops']
            row.update(all16_and_bill_exact=True,control=pin(expected_path),features_persisted=False)
            del expected
        else:
            packet['prediction']=prediction
            path=out/f'mini{identity:04d}.npz'
            with path.open('xb') as stream:np.savez(stream,**packet)
            row.update(identity=item['identity'],packet=pin(path))
        records.append(row);once(out/f'row{len(records):02d}.json',row)
        print(__import__('json').dumps(dict(event='terminal_capture',mode=mode,**row)),flush=True)
        calls.clear();terminal_calls.clear()
        del weights,mlp,packet,prediction,head,delta
        gc.collect();relief(None,0)
        assert not held and not calls and not terminal_calls
    once(out/'complete.json',dict(passed=True,mode=mode,rows=records,source=pin(COMPONENT),
         targets_accessed=False,all_capture_references_retired=True,elapsed_s=time.monotonic()-started,
         roster=list(TRAIN if mode=='capture' else PARITY),
         source_edit=False,numerical_candidate_change=False))


if __name__=='__main__':main()
