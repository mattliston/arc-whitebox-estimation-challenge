"""Fixed, net-held-out information gate; not a deployable correction head."""
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
target=HERE/"joint_plane_probe_verdict_v1.json"
assert not target.exists()
assert (HERE/"joint_plane_probe_rows_v1.json").is_file()
VALID={37,39,42,48}
data={r:dict(np.load(HERE/"rows_joint_plane_probe_v1"/f"row_{r:04d}"/"data.npz")) for r in range(35,59)}
train=[r for r in data if r not in VALID];valid=sorted(VALID)
for rank in data:
    result=json.loads((HERE/"rows_joint_plane_probe_v1"/f"row_{rank:04d}"/"result.json").read_text())
    assert result["parent_byte_exact"]

def raw_features(rank,layer,joint):
    d=data[rank]
    f=d["x"][layer].astype(np.float64)
    if joint:f=np.concatenate((f,d["joint"][layer].astype(np.float64)),axis=1)
    return f

def design(rank,layer,joint,center,scale):
    d=data[rank];x=d["x"][layer].astype(np.float64)
    f=np.clip((raw_features(rank,layer,joint)-center)/scale,-8,8)
    g=x[:,5:6]-.5
    ap=x[:,3:4]*x[:,6:7]
    z=np.concatenate((np.ones((len(f),1)),f,f*g,f*ap),axis=1)
    return z*d["sigma"][layer,:,None]

records=[];predictions={kind:{rank:[] for rank in valid} for kind in ("base","control","joint")}
started=time.time()
with threadpool_limits(limits=2):
    for layer in range(10,16):
        for joint in (False,True):
            kind="joint" if joint else "control"
            raw=np.concatenate([raw_features(r,layer,joint) for r in train])
            center=raw.mean(0);scale=np.maximum(raw.std(0),1e-12)
            xx=np.concatenate([design(r,layer,joint,center,scale) for r in train])
            y=np.concatenate([data[r]["target"][layer]-data[r]["delta"][layer] for r in train])
            gram=xx.T@xx
            # One preregistered ridge scale, no tuning on held-out identities.
            ridge=1e-3*float(np.trace(gram))/len(gram)
            coef=np.linalg.solve(gram+ridge*np.eye(len(gram)),xx.T@y)
            tr=float(np.mean((xx@coef-y)**2))
            per_net={}
            for rank in valid:
                residual=data[rank]["target"][layer]-data[rank]["delta"][layer]
                error=residual-design(rank,layer,joint,center,scale)@coef
                per_net[rank]=float(np.mean(error*error))
                predictions[kind][rank].append(error)
                if not joint:predictions["base"][rank].append(residual)
            records.append(dict(layer=layer,kind=kind,feature_columns=len(gram),ridge=ridge,
                train_mse=tr,valid_mse=float(np.mean(list(per_net.values()))),per_net=per_net))
summary={kind:float(np.mean([np.mean(np.asarray(errors)**2) for errors in nets.values()])) for kind,nets in predictions.items()}
per_net={rank:{kind:float(np.mean(np.asarray(predictions[kind][rank])**2)) for kind in predictions} for rank in valid}
ratio=summary["joint"]/summary["control"]
receipt=dict(train_nets=train,valid_nets=valid,layers=list(range(10,16)),
    summary=summary,per_net=per_net,joint_vs_control_ratio=ratio,information_gate_pass=ratio<=.95,
    records=records,wall_s=time.time()-started,
    scope="Frozen teacher-forced parent mean-head residual, stored 1e9 targets; net-held-out ridge diagnostic only, not closed-loop or panel accuracy.")
target.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2),flush=True)
