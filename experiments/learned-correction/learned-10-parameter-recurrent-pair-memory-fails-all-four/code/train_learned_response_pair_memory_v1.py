"""Inert fixed100-epoch paired fit; no validation before BOTH final seals.

The learned10-parameter predictor never receives a loss dictionary. This
is a teacher-action compression test, not a raw-score or autonomous card.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
DATA = HERE/"learned_response_pair_memory_data_v1"
OUT = HERE/"learned_response_pair_memory_training_v1"
VALID = (37,39,42,48)
TRAIN = tuple(r for r in range(35,59) if r not in VALID)
SEED, EPOCHS = 930074, 100
RSS_LIMIT, FIT_LIMIT = 6*(1<<30), 600.
MODEL_PINS = {
    "LEARNED_RESPONSE_PAIR_MEMORY_PROTOCOL_DRAFT_V1.md": "1e277cd16c6596cca947172d0dcd2bd7c21c8962256596ca8ea070c65fc8754c",
    "LEARNED_RESPONSE_PAIR_MEMORY_ROOT_ACCEPTANCE_V1.md": "45c590e373fb35c6af0c9a248addf8b98d5e6830c8b6b66861fd5ced3b537e2a",
    "learned_response_pair_memory_numpy_v1.py": "5a24a77c24f08ba521fa8d0cd36a569a9eeee6bf3045cb4ee15d0117fbf4129e",
    "learned_response_pair_memory_torch_v1.py": "54e4040947cfd818246b1cc604fd79516e426771ca23b1568eb2391b394c2789",
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, obj):
    with path.open("x") as stream:
        json.dump(obj,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")


def execute():
    require("/anaconda3/" in sys.executable,"Use AP/Torch interpreter")
    require("threadpoolctl" not in sys.modules and "torch" not in sys.modules,"FreshAP process required")
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,"Launchnice>=10")
    require(not OUT.exists(),"Existing experiment; no overwrite/resume")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="0"
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[name]="2"
    paths={name:HERE/name for name in MODEL_PINS}
    paths[Path(__file__).name]=Path(__file__).resolve()
    for name,digest in MODEL_PINS.items():
        require(sha(paths[name])==digest,f"Frozen source changed:{name}")
    for name in ("learned_response_pair_memory_synthetic_certificate_v1.json",
                 "learned_response_pair_memory_wide_certificate_v1.json"):
        p=HERE/name
        certificate=json.loads(p.read_text())
        require(certificate["passed"],f"Prerequisite did notpass:{name}")
        paths[name]=p
        for source,digest in certificate["source_hashes"].items():
            require(sha(HERE/source)==digest,f"Certificate sourcechanged:{source}")
            paths[source]=HERE/source
    declared_row_hashes={}
    for name in ("rank35_receipt.json","remaining_receipt.json"):
        p=DATA/name
        receipt=json.loads(p.read_text())
        require(receipt["passed"] and receipt["hashes_unchanged"],"Data preparation incomplete")
        expected_ranks={35} if name=="rank35_receipt.json" else set(range(36,59))
        require({r["rank"] for r in receipt["rows"]}==expected_ranks and
                all(r["passed"] for r in receipt["rows"]),"Incomplete global data seal")
        declared_row_hashes.update({r["rank"]:r["result_sha256"] for r in receipt["rows"]})
        paths[str(p.relative_to(HERE))]=p
    receipts={}
    for rank in range(35,59):
        directory=DATA/f"row_{rank:04d}"
        p=directory/"result.json"
        require(sha(p)==declared_row_hashes[rank],"Row result differs from global completion seal")
        paths[str(p.relative_to(HERE))]=p
        if rank in VALID:
            # This JSON contains target-energy scalars. Authenticate its bytes,
            # but do not parse even that metadata before BOTH model seals.
            continue
        r=json.loads(p.read_text())
        require(r["passed"] and r["rank"]==rank and r["peak_rss_bytes"]<=RSS_LIMIT,
                f"Unqualified data row:{rank}")
        paths[str(p.relative_to(HERE))]=p
        for name,item in r["artifacts"].items():
            q=directory/name
            require(sha(q)==item["sha256"],f"Data artifactchanged:{rank}/{name}")
            paths[str(q.relative_to(HERE))]=q
        for name,digest in r["source_hashes"].items():
            require(sha(HERE/name)==digest,f"Data sourcechanged:{name}")
            paths[name]=HERE/name
        receipts[rank]=r
    hashes={name:sha(p) for name,p in paths.items()}
    OUT.mkdir(exist_ok=False)
    started=time.perf_counter()
    result=dict(passed=False,completed=False,stage="load_train_only",train_ranks=TRAIN,
        valid_ranks=VALID,seed=SEED,epochs=EPOCHS,learning_rate=.001,ridge=.001,gradient_cap=5.,
        training_limit_s=FIT_LIMIT,source_and_input_hashes=hashes,
        held_out_losses_decoded=False,raw_score_claim=False,autonomous_card=False,arms={})
    write_json(OUT/"started.json",result)
    try:
        import numpy as np
        import torch
        from learned_response_pair_memory_torch_v1 import predict, DATA_KEYS
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.manual_seed(SEED)
        require(torch.backends.mps.is_available(),"Native MPS required; nofallback")
        require("threadpoolctl" not in sys.modules,"AP threadpoolctl trap")
        device=torch.device("mps")

        def resource_gate():
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=RSS_LIMIT,"6GiB processRSS limit")
            require("threadpoolctl" not in sys.modules,"AP threadpoolctl trap")

        def tensor_dict(arrays):
            return {k:torch.tensor(v,device=device,dtype=torch.bool if v.dtype==np.bool_ else torch.float32)
                    for k,v in arrays.items()}

        def load_input(rank):
            with np.load(DATA/f"row_{rank:04d}/inputs.npz",allow_pickle=False) as z:
                require(set(z.files)==DATA_KEYS,"Causal dictionary changed / loss fieldpresent")
                arrays={k:z[k] for k in z.files}
            require(all(np.isfinite(v).all() for v in arrays.values()),"Nonfinitecausal input")
            return tensor_dict(arrays)

        def load_loss(rank, tensors):
            with np.load(DATA/f"row_{rank:04d}/loss.npz",allow_pickle=False) as z:
                arrays={k:z[k] for k in z.files}
            expected={"covariance_forward","covariance_reverse","next_weight","next_response",
                      "covariance_target","next_mean_target"}
            require(set(arrays)==expected and all(np.isfinite(v).all() for v in arrays.values()),
                    "Loss dictionary contract changed")
            return tensor_dict(arrays) if tensors else arrays

        def consumers(delta,loss):
            cov=loss["covariance_forward"]*delta+loss["covariance_reverse"]*delta.T
            w=loss["next_weight"]
            nxt=torch.sum(w*(cov@w),dim=0)*loss["next_response"]
            return cov,nxt

        norms={"covariance":float(np.mean([receipts[r]["zero_covariance_energy"] for r in TRAIN])),
               "next_mean":float(np.mean([receipts[r]["zero_next_mean_energy"] for r in TRAIN]))}
        require(all(np.isfinite(v) and v>0 for v in norms.values()),"InvalidTRAIN-only energy")
        data,losses={},{}
        for rank in TRAIN:
            data[rank]=load_input(rank)
            losses[rank]=load_loss(rank,True)
            resource_gate()
        torch.mps.synchronize()
        result["train_normalizations"]=norms
        result["mps_ready_allocated_bytes"]=int(torch.mps.current_allocated_memory())
        result["mps_ready_driver_bytes"]=int(torch.mps.driver_allocated_memory())
        print(json.dumps(dict(stage="train_ready",normalizations=norms,
            allocated_bytes=result["mps_ready_allocated_bytes"],
            driver_bytes=result["mps_ready_driver_bytes"])),flush=True)
        fit_started=time.perf_counter()
        orders=[]
        rng=np.random.default_rng(SEED)
        for _ in range(EPOCHS):
            orders.append([int(r) for r in rng.permutation(TRAIN)])
        fitted={}
        for arm,memory in (("history",True),("reset",False)):
            result["stage"]="fit_"+arm
            parameters=torch.nn.Parameter(torch.zeros(10,device=device))
            optimizer=torch.optim.Adam([parameters],lr=.001)
            records=[]
            max_gate_gradient=0.
            active_gate_gradient_seen=False
            for epoch,order in enumerate(orders):
                total_cov,total_next=0.,0.
                for rank in order:
                    require(time.perf_counter()-fit_started<=FIT_LIMIT,"Paired600s cap; incomplete, noresume")
                    optimizer.zero_grad(set_to_none=True)
                    delta=predict(data[rank],parameters,memory=memory)
                    cov,nxt=consumers(delta,losses[rank])
                    ce=torch.mean((cov-losses[rank]["covariance_target"])**2)
                    ne=torch.mean((nxt-losses[rank]["next_mean_target"])**2)
                    objective=ce/norms["covariance"]+ne/norms["next_mean"]+.001*torch.mean(parameters*parameters)
                    require(bool(torch.isfinite(objective).item()),"Nonfiniteobjective")
                    objective.backward()
                    require(parameters.grad is not None and bool(torch.isfinite(parameters.grad).all().item()),
                            "Missing/nonfiniteparameter gradient")
                    gate_norm=float(torch.linalg.vector_norm(parameters.grad[:8]).detach().cpu())
                    max_gate_gradient=max(max_gate_gradient,gate_norm)
                    active_gate_gradient_seen=active_gate_gradient_seen or gate_norm>0
                    if not memory:
                        require(float(parameters.grad[1].detach().cpu())==0 and
                                float(parameters.grad[3].detach().cpu())==0,"RESET inactivegradient")
                    torch.nn.utils.clip_grad_norm_([parameters],5.)
                    optimizer.step()
                    total_cov+=float(ce.detach().cpu())
                    total_next+=float(ne.detach().cpu())
                    resource_gate()
                torch.mps.synchronize()
                require(time.perf_counter()-fit_started<=FIT_LIMIT,"Paired600s cap; incomplete, noresume")
                rec=dict(epoch=epoch+1,train_online_covariance_mse=total_cov/len(TRAIN),
                    train_online_next_mean_mse=total_next/len(TRAIN),
                    fit_wall_s=time.perf_counter()-fit_started)
                records.append(rec)
                write_json(OUT/f"{arm}_epoch_{epoch+1:03d}.json",rec)
                if epoch==0 or (epoch+1)%10==0:
                    print(json.dumps(dict(arm=arm,**rec)),flush=True)
            require(active_gate_gradient_seen,"No trained gategradient after beta initialization")
            p=parameters.detach().cpu().numpy()
            require(np.isfinite(p).all(),"Nonfinitefinalparameters")
            with (OUT/f"{arm}_model.npz").open("xb") as f:
                np.savez(f,parameters=p)
            model_sha=sha(OUT/f"{arm}_model.npz")
            fitted[arm]=p.copy()
            result["arms"][arm]=dict(completed=True,parameters=p.tolist(),allocated_parameters=10,
                structurally_active_parameters=10 if memory else 8,gate_gradients_seen=True,
                max_gate_gradient_norm=max_gate_gradient,model_sha256=model_sha,history=records)
            del optimizer,parameters,delta,cov,nxt,objective,ce,ne
        result["fit_wall_s"]=time.perf_counter()-fit_started
        write_json(OUT/"both_models_sealed.json",dict(held_out_losses_decoded=False,
            sealed_utc=datetime.now(timezone.utc).isoformat(),
            models={arm:result["arms"][arm]["model_sha256"] for arm in fitted},
            train_normalizations=norms))
        result["held_out_losses_decoded"]=True
        result["held_out_loss_arrays_decoded"]=False
        result["held_out_metadata_decoded_after_both_model_seals"]=True
        for rank in VALID:
            directory=DATA/f"row_{rank:04d}"
            row_path=directory/"result.json"
            require(sha(row_path)==declared_row_hashes[rank],"Held-out row receipt changed")
            r=json.loads(row_path.read_text())
            require(r["passed"] and r["rank"]==rank and r["peak_rss_bytes"]<=RSS_LIMIT,
                    "Unqualified held-out preparation")
            for name,item in r["artifacts"].items():
                p=directory/name
                require(sha(p)==item["sha256"],"Held-out artifact changed")
                key=str(p.relative_to(HERE))
                paths[key]=p
                hashes[key]=item["sha256"]
            for name,digest in r["source_hashes"].items():
                require(sha(HERE/name)==digest,"Held-out preparation source changed")
                paths[name]=HERE/name
                hashes[name]=digest
        # Free trainingcache before the final held-out stage. No new update follows.
        data.clear()
        losses.clear()
        torch.mps.empty_cache()
        result["stage"]="freeze_valid_predictions"
        predictions={}
        with torch.no_grad():
            for rank in VALID:
                inp=load_input(rank)
                for arm,memory in (("history",True),("reset",False)):
                    p=torch.tensor(fitted[arm],device=device)
                    pred=predict(inp,p,memory=memory).cpu().numpy()
                    require(np.isfinite(pred).all(),"Nonfiniteheld-out prediction")
                    predictions[arm,rank]=pred
                    with (OUT/f"{arm}_rank_{rank:04d}_prediction.npz").open("xb") as f:
                        np.savez(f,deltaD=pred)
                del inp
        write_json(OUT/"valid_predictions_sealed.json",dict(held_out_loss_arrays_decoded=False,
            held_out_target_metadata_decoded_after_model_seals=True,
            sealed_utc=datetime.now(timezone.utc).isoformat(),
            hashes={p.name:sha(p) for p in sorted(OUT.glob("*_prediction.npz"))}))
        result["held_out_loss_arrays_decoded"]=True
        result["stage"]="fixed_valid_gate"
        held={arm:[] for arm in fitted}
        for rank in VALID:
            loss={k:v.astype(np.float64) for k,v in load_loss(rank,False).items()}
            for arm in fitted:
                delta=predictions[arm,rank].astype(np.float64)
                cov=loss["covariance_forward"]*delta+loss["covariance_reverse"]*delta.T
                w=loss["next_weight"]
                nxt=np.sum(w*(cov@w),axis=0)*loss["next_response"]
                ce=float(np.mean((cov-loss["covariance_target"])**2))
                ne=float(np.mean((nxt-loss["next_mean_target"])**2))
                cz=float(np.mean(loss["covariance_target"]**2))
                nz=float(np.mean(loss["next_mean_target"]**2))
                require(cz>0 and nz>0,"Zero held-out denominator")
                held[arm].append(dict(rank=rank,covariance_mse=ce,next_mean_mse=ne,
                    zero_covariance_mse=cz,zero_next_mean_mse=nz,
                    covariance_ratio=ce/cz,next_mean_ratio=ne/nz))
        pools={arm:{k:float(np.mean([r[k] for r in rows])) for k in
                    ("covariance_mse","next_mean_mse","zero_covariance_mse","zero_next_mean_mse")}
               for arm,rows in held.items()}
        h,r=pools["history"],pools["reset"]
        gates=dict(history_covariance_ratio=h["covariance_mse"]/h["zero_covariance_mse"],
            history_next_mean_ratio=h["next_mean_mse"]/h["zero_next_mean_mse"],
            history_reset_next_mean_ratio=h["next_mean_mse"]/r["next_mean_mse"],
            history_reset_wins=sum(a["next_mean_mse"]<b["next_mean_mse"]
                                   for a,b in zip(held["history"],held["reset"])))
        result.update(held_out=held,pooled=pools,gates=gates,completed=True,stage="complete")
        result["passed"]=(gates["history_covariance_ratio"]<=.5 and gates["history_next_mean_ratio"]<=.5
            and gates["history_reset_next_mean_ratio"]<=.8 and gates["history_reset_wins"]>=3)
        result["mps_final_allocated_bytes"]=int(torch.mps.current_allocated_memory())
        result["mps_final_driver_bytes"]=int(torch.mps.driver_allocated_memory())
        resource_gate()
    except BaseException as exc:
        result["passed"]=False
        result.update(exception_type=type(exc).__name__,exception=str(exc))
        if "parameters" in locals():
            try:
                partial=parameters.detach().cpu().numpy()
                with (OUT/"incomplete_parameters_not_for_selection.npz").open("xb") as f:
                    np.savez(f,parameters=partial)
                result["incomplete_parameter_arm"]=arm
            except BaseException as save_error:
                result["partial_save_error"]=str(save_error)
    finally:
        result["total_wall_s"]=time.perf_counter()-started
        result["peak_rss_bytes"]=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result["finished_utc"]=datetime.now(timezone.utc).isoformat()
        result["final_hashes"]={name:sha(p) for name,p in paths.items()}
        result["hashes_unchanged"]=result["final_hashes"]==hashes
        result["passed"]=bool(result["passed"] and result["hashes_unchanged"] and
                              result["peak_rss_bytes"]<=RSS_LIMIT)
        write_json(OUT/"result.json",result)
        print(json.dumps({k:v for k,v in result.items() if k not in
                          ("arms","source_and_input_hashes","final_hashes")},allow_nan=False),flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.execute:
        execute()
    else:
        print("INERT: one paired100-epoch20/4fit; allnumerical/dataprerequisites required.")
