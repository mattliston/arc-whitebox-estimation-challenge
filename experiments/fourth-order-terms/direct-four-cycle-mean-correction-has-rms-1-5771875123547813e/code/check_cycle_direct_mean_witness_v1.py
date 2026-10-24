"""INERT: fixed existing-data4cycle direct mean13 witness, NOT a candidate.

--acquire seals source-derived arrays/bill without truth; --score compares
one natural-strength direct vector with the existing development reference.
No model/old source test is imported or executed. No fit or parameter knob.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE / "cycle_direct_mean_witness_v1"
OLD = "gaussian_fourth_r1_source_test_v1/"
HELPER = "check_response_rows_adjusted_v1.py"
SCHEMA = "cycle_direct_mean_witness_v1"
BLAS = ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS")
PINS = {
    HELPER: "8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304",
    "check_gaussian_fourth_r1_source_v1.py": "e88da777239fee2b03a0c74d84ef772eaf51d3d76fce55a5f026f6c878ce40f9",
    "GAUSSIAN_FOUR_CYCLE_EXISTING_MEAN_WITNESS_V1.md": "69627f228d07d951f23652d162c4605644bb115eaaabb4e6963a601de78b6886",
    OLD+"result.json": "f08efe2a6367085ce0bf7db6a964d38d61d56fb1638e8ca15e379c14af29db0f",
    OLD+"capture_sealed.json": "d079f5cd686e10d2183aaf1f43607370741847f934adf0cb14adf53084df8769",
    OLD+"causal_inputs.npz": "2143d18e35f7bdf06101b718d16df13ed3d5c0ce68d6d3f3dff04929870265a0",
    OLD+"proxy_values.npz": "fb1d82abb8433e862abb66c979eedfadad6820bd284332e83c971a15c0c070c7",
    OLD+"bank_output.npy": "43f66a6fbb1cd0a362a8d2b1251235322a53994650be2da3c0262c53ad26a5b5",
    "candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py": "3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757",
    "response_rows_adjusted_reevaluation_v1/score/result.json": "a5e4a8aa64968218931d3b11aa80141580371ec17931f7a1532b7381ab5437d0",
}
TRUTH_SHA = "ec47c040198070cfe936126d18d24d02a94a6fef1da14795173c23d199e05f61"
KEYS = ("alive12","pdf12","sigma12","W13","pre_v13","alpha13","pdf13","sigma13")


def helper():
    path = HERE / HELPER
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[HELPER]:
        raise RuntimeError("Pure-helper hash changed")
    spec = importlib.util.spec_from_file_location("cycle_pure_helpers",path)
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    return h


def bindings(h):
    result = {name:h.binding(HERE/name) for name in PINS}
    for name, expected in PINS.items():
        h.require(result[name]["sha256"] == expected,"Frozen source/data changed: "+name)
    for name in (Path(__file__).name,"CYCLE_DIRECT_MEAN_WITNESS_PROTOCOL_V1.md"):
        result[name] = h.binding(HERE/name)
    old = h.load_json(HERE/OLD/"result.json")
    seal = h.load_json(HERE/OLD/"capture_sealed.json")
    h.require(old["completed"] and old["qualification_passed"] and old["hashes_unchanged"]
              and not old["truth_files_accessed"] and old["capture"]["all16_served_byte_parity"]
              and old["capture_seal_sha256"] == PINS[OLD+"capture_sealed.json"]
              and seal["capture"]["reference_sha256"] == PINS[OLD+"bank_output.npy"],
              "Old causal/unconditional-output ownership failed")
    for name in ("causal_inputs.npz","proxy_values.npz","bank_output.npy"):
        h.require(old["artifacts"][name]["sha256"] == PINS[OLD+name],"Old artifact binding mismatch")
    return result,old


def source_arithmetic(xp,inputs,u,theta,Q):
    x = {key:xp.asarray(value,dtype=xp.float64) for key,value in inputs.items()}
    u = xp.asarray(u,dtype=xp.float64)
    theta,Q = xp.asarray(theta,dtype=xp.float64),xp.asarray(Q,dtype=xp.float64)
    d = x["alive12"]*x["pdf12"]/x["sigma12"]
    f = theta*d*xp.square(u)
    a = x["W13"].T @ f
    rho = 9*xp.square(xp.sum(xp.square(f)))/Q
    positive = 3*xp.square(xp.square(a))
    radial = rho*xp.square(x["pre_v13"])
    dk4 = positive-radial
    omega = (xp.square(x["alpha13"])-1)*x["pdf13"]/(24*x["pre_v13"]*x["sigma13"])
    delta = omega*dk4
    return dict(f=f,a=a,positive=positive,radial=radial,delta_k4=dk4,omega=omega,delta=delta),dict(
        rho=rho,unit_norm=xp.sum(xp.square(u)),delta_rms=xp.sqrt(xp.mean(xp.square(delta))))


def environment(h):
    h.require(Path(sys.prefix).resolve()==h.GATE_ENV.resolve() and "anaconda3" not in sys.executable,
              "Use designated OFFPY, never AP/Torch")
    h.require(os.getpriority(os.PRIO_PROCESS,0)>=10,"Use nice>=10")
    env={name:os.environ.get(name) for name in BLAS}
    h.require(env==dict.fromkeys(BLAS,"1"),"Set all BLAS controls1 before imports")
    versions={name:importlib.metadata.version(name) for name in ("flopscope","whestbench","numpy","threadpoolctl")}
    h.require(versions["flopscope"]=="0.12.0" and versions["whestbench"]=="0.16.0","Wrong native versions")
    return dict(pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),executable=sys.executable,
                blas_env=env,versions=versions)


def acquire(h,directory):
    runtime=environment(h); source_set,old=bindings(h)
    h.require("numpy" not in sys.modules and "torch" not in sys.modules,"Fresh process required")
    start=dict(schema=SCHEMA,utc=h.utc(),runtime=runtime,source_bindings=source_set,truth_files_accessed=False,
               rank=35,source_layer=12,comparison_layer=13,model_executed=False,fits=0)
    start_sha=h.write_json(directory/"started.json",start)
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info,threadpool_limits
    t0=time.perf_counter()
    with np.load(HERE/OLD/"causal_inputs.npz",allow_pickle=False) as archive:
        inputs={key:archive[key] for key in KEYS}
    with np.load(HERE/OLD/"proxy_values.npz",allow_pickle=False) as archive:
        u,reference_omega=archive["u"],archive["omega"]
    for key,value in inputs.items():
        shape=(1024,1024) if key=="W13" else (1024,)
        dtype="float32" if key=="W13" else "bool" if key=="alive12" else "float64"
        h.require(value.shape==shape and value.dtype==np.dtype(dtype) and bool(np.isfinite(value).all()),
                  "Wrong captured input: "+key)
    h.require(all(v.shape==(1024,) and v.dtype==np.dtype("float64") and bool(np.isfinite(v).all())
                  for v in (u,reference_omega)),"Wrong old direction/omega")
    h.require(bool((inputs["sigma12"]>0).all()) and bool((inputs["sigma13"]>0).all())
              and bool((inputs["pre_v13"]!=0).all()),"Degenerate captured response")
    theta,Q=old["proxy_scalars"]["theta"],old["proxy_scalars"]["bank_Q"]
    h.require(math.isfinite(theta) and theta>0 and math.isfinite(Q) and Q>0,"Invalid fixed theta/Q")
    loading=time.perf_counter()-t0
    with threadpool_limits(limits=1):
        pools=threadpool_info()
        h.require(all(p["num_threads"]==1 for p in pools if p.get("user_api")=="blas"),"Wrong reported BLAS threads")
        began=h.utc(); t0=time.perf_counter()
        with flops.BudgetContext(flop_budget=2**27,wall_time_limit_s=20,quiet=True) as ctx:
            vectors,scalars=source_arithmetic(fnp,inputs,u,theta,Q)
        elapsed=time.perf_counter()-t0
    arrays={key:np.asarray(value).copy() for key,value in vectors.items()}
    values={key:float(value) for key,value in scalars.items()}
    h.require(all(v.shape==(1024,) and v.dtype==np.dtype("float64") and bool(np.isfinite(v).all())
                  for v in arrays.values()) and all(math.isfinite(v) for v in values.values()),"Invalid source outputs")
    omega_error=float(np.linalg.norm(arrays["omega"]-reference_omega)/max(np.linalg.norm(reference_omega),1e-300))
    h.require(abs(values["unit_norm"]-1)<=1e-10 and omega_error<=1e-12,"Unit/omega identity gate failed")
    path=directory/"source_vectors.npz"
    with path.open("xb") as stream:np.savez(stream,**arrays)
    vector_binding=h.binding(path)
    bill=dict(began_utc=began,ended_utc=h.utc(),flops=int(ctx.flops_used),external_wall_s=elapsed,
              meter_wall_s=float(ctx.wall_time_s),backend_s=float(ctx.flopscope_backend_time_s),
              overhead_s=float(ctx.flopscope_overhead_time_s),residual_s=float(ctx.residual_wall_time_s),
              peak_rss_bytes=h.peak_rss_bytes(),reused_old_bank_and_power_acquisition=True,
              full_candidate_bill=False)
    h.write_json(directory/"bill.json",bill)
    h.require(0<bill["flops"]<=2**27 and 0<bill["meter_wall_s"]<20 and elapsed<20
              and 0<=bill["residual_s"]<float("inf") and bill["peak_rss_bytes"]<h.RSS_LIMIT,
              "Source arithmetic resource gate failed")
    h.check_bindings(source_set)
    result=dict(schema=SCHEMA,qualification_passed=True,completed=True,utc=h.utc(),started_sha256=start_sha,
                source_bindings=source_set,runtime=runtime,threadpools=pools,loading_wall_s=loading,
                vectors=vector_binding,bill_binding=h.binding(directory/"bill.json"),bill=bill,scalars=values,
                omega_relative_error=omega_error,truth_files_accessed=False,source_proxy_not_raw_score=True,
                source_layer=12,comparison_layer=13,rank=35,fits=0,model_executed=False,hashes_unchanged=True)
    sha=h.write_json(directory/"result.json",result)
    h.write_json(directory/"completion.json",dict(schema=SCHEMA,qualification_passed=True,result_sha256=sha,utc=h.utc()))
    print(json.dumps(dict(result_sha256=sha,bill=bill,scalars=values,omega_relative_error=omega_error)),flush=True)


def score(h,directory):
    runtime=environment(h); source_set,old=bindings(h)
    acq=ROOT/"acquire"; complete=h.load_json(acq/"completion.json")
    h.require(complete["qualification_passed"] and h.digest(acq/"result.json")==complete["result_sha256"],"Acquisition seal failed")
    result=h.load_json(acq/"result.json")
    h.require(result["qualification_passed"] and result["completed"] and result["hashes_unchanged"]
              and not result["truth_files_accessed"] and result["source_bindings"]==source_set
              and result["runtime"]["pid"]!=runtime["pid"]
              and result["runtime"]["versions"]==runtime["versions"],"Acquisition frame changed")
    h.require(h.digest(acq/"started.json")==result["started_sha256"]
              and h.load_json(acq/"bill.json")==result["bill"],"Start or bill changed")
    h.check_bindings(dict(vectors=result["vectors"],bill=result["bill_binding"]))
    import numpy as np
    with np.load(result["vectors"]["path"],allow_pickle=False) as archive:delta=archive["delta"]
    output=np.load(HERE/OLD/"bank_output.npy",allow_pickle=False); h.assert_output(np,output)
    before=h.write_json(directory/"verified_before_truth.json",dict(schema=SCHEMA,utc=h.utc(),
        source_bindings=source_set,acquisition=h.binding(acq/"result.json"),truth_files_accessed=False))
    truth_time=h.utc(); h.write_json(directory/"truth_access_started.json",dict(utc=truth_time,prior_seal_sha256=before))
    path=h.TRUTH_DIR/"truthpost_rank0035.npz"; truth_binding=h.binding(path)
    h.require(truth_binding["sha256"]==TRUTH_SHA,"Development truth changed")
    with np.load(path,allow_pickle=False) as archive:truth=archive["mc_post"][13].astype(np.float64)
    h.require(truth.shape==(1024,) and bool(np.isfinite(truth).all()),"Invalid development reference")
    e=output[13].astype(np.float64)-truth
    E=float(np.mean(e*e)); D=float(np.mean(delta*delta)); ED=float(np.mean(e*delta))
    h.require(E>0 and D>=0 and all(math.isfinite(v) for v in (E,D,ED)),"Degenerate direct witness")
    natural=float(np.mean((e+delta)**2)); lower=max(0,math.sqrt(E)-math.sqrt(D))**2
    h.require(abs(natural-(E+2*ED+D))<=1e-12*max(E,natural),"Direct-vector expansion failed")
    h.check_bindings(source_set); h.check_bindings(dict(truth=truth_binding))
    value=dict(schema=SCHEMA,qualification_passed=True,utc=h.utc(),source_bindings=source_set,runtime=runtime,
        acquisition_sha256=complete["result_sha256"],verified_before_truth_sha256=before,
        truth_access_started_utc=truth_time,truth_binding=truth_binding,rank=35,comparison_layer=13,
        bank_layer13_mse=E,delta_mse=D,mean_error_times_delta=ED,natural_direct_vector_mse=natural,
        natural_direct_over_bank=natural/E,delta_rms=math.sqrt(D),delta_rms_over_error_rms=math.sqrt(D/E),
        needed_correction_cosine=None if D==0 else -ED/math.sqrt(E*D),direct_vector_triangle_lower_mse=lower,
        direct_vector_triangle_max_relative_gain=1-lower/E,source_arithmetic_bill=result["bill"],
        natural_direction_improves_direct_comparison=natural<E,fitted_strength=False,
        scientific_status="local_direct_signal_only_full_route_not_admitted",full_candidate_admitted=False,
        final_score_evidence=False,adjusted_score=None,reference="Stored1e9 development mean13; old bank rank35",
        actual_finite_strength_replay=False,model_executed=False,retained_parent_validated=False,
        source_proxy_not_raw_score=True,hashes_unchanged=True,peak_rss_bytes=h.peak_rss_bytes())
    h.require(value["peak_rss_bytes"]<h.RSS_LIMIT,"Score-only whole-process RSS gate failed")
    h.check_bindings(dict(vectors=result["vectors"],bill=result["bill_binding"]))
    h.require(h.digest(acq/"result.json")==complete["result_sha256"],"Acquisition result changed during score")
    sha=h.write_json(directory/"result.json",value)
    h.write_json(directory/"completion.json",dict(schema=SCHEMA,qualification_passed=True,result_sha256=sha,utc=h.utc()))
    print(json.dumps({k:v for k,v in value.items() if k not in {"source_bindings","runtime","truth_binding"}}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument("--acquire",action="store_true");mode.add_argument("--score",action="store_true")
    args=parser.parse_args()
    if not args.acquire and not args.score:parser.print_help();return 0
    h=helper()
    if args.acquire:ROOT.mkdir(exist_ok=False)
    h.require(ROOT.is_dir() and not ROOT.is_symlink(),"Acquire first in the new fixed directory")
    directory=ROOT/("acquire" if args.acquire else "score");directory.mkdir(exist_ok=False)
    try:
        acquire(h,directory) if args.acquire else score(h,directory)
    except Exception as error:
        h.write_json(directory/"failure.json",dict(schema=SCHEMA,utc=h.utc(),qualification_passed=False,
            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
        print(f"{type(error).__name__}: {error}",file=sys.stderr,flush=True);return 1
    return 0


if __name__=="__main__":raise SystemExit(main())
