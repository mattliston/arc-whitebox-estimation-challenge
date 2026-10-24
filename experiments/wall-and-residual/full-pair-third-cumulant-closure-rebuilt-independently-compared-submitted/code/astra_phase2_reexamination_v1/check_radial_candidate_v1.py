"""Official-environment 35/100/35 state-reset, resource, and head-parity gate."""
import hashlib,importlib.metadata,importlib.util,json,resource,sys,time
from pathlib import Path
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed
flops.configure(symmetry_warnings=False)
HERE=Path(__file__).resolve().parent
STORE=HERE.parent/"learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
source=HERE/"candidate_radial_gnn_splitread_v1.py"
outfile=HERE/"radial_splitread_native_parity_v1.json"
if outfile.exists():raise FileExistsError(outfile)
spec=importlib.util.spec_from_file_location("astra_native_radial",source)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
man=json.loads((STORE/"manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
trials=[];outputs=[]
for call,rank in enumerate((35,100,35)):
    rp=STORE/man[rank]["path"];row=json.loads(rp.read_text());W=np.load(rp.parent/"weights.npy",mmap_mode="r")
    mlp=MLP(width=1024,depth=16,weights=list(W),seed=int(derive_estimator_seed(int(row["mlp_seed"]))),name=f"native-radial-{rank}")
    rows=[];deltas=[];original=mod.gnn_step
    def wrapped(l,x,sigma,w,state,P):
        result=original(l,x,sigma,w,state,P)
        rows.append(np.asarray(x).copy());deltas.append(np.asarray(result[0]).copy())
        return result
    if call<2:mod.gnn_step=wrapped
    t0=time.time()
    with threadpool_limits(limits=4):
        with flops.BudgetContext(flop_budget=2**41,wall_time_limit_s=120,quiet=True) as ctx:
            result=mod.Estimator().predict(mlp,2**41)
    mod.gnn_step=original
    out=np.asarray(result,dtype=np.float64);outputs.append(out.copy())
    ref=np.load(HERE/f"radial_closedloop_r{rank}_v1.npz")
    r=dict(call=call,rank=rank,flops=int(ctx.flops_used),util=int(ctx.flops_used)/2**41,
           wall_s=time.time()-t0,residual_s=float(ctx.residual_wall_time_s),
           max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
           numpy_output_max_abs=float(np.max(np.abs(out-np.maximum(ref["mean"],0)))),
           output_sha256=hashlib.sha256(out.tobytes()).hexdigest(),
           instrumented=call<2)
    if rows:
        r["feature_max_abs"]=float(np.max(np.abs(np.stack(rows)-ref["x"])))
        r["delta_max_abs"]=float(np.max(np.abs(np.stack(deltas)-ref["delta"])))
    trials.append(r);print(json.dumps(r),flush=True)
repeat=bool(np.array_equal(outputs[0],outputs[2]))
passed=repeat and all(t["numpy_output_max_abs"]<5e-6 and t["residual_s"]<.4 and t["max_rss_bytes"]<8*1024**3 and t.get("delta_max_abs",0)<1e-6 for t in trials)
receipt=dict(pass_gate=passed,repeat_byte_identical=repeat,trials=trials,
             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             versions={k:importlib.metadata.version(k) for k in ("flopscope","whestbench","numpy")})
outfile.write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(dict(pass_gate=passed)),flush=True)
