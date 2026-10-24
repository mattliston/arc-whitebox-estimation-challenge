"""One fixed final-only refit; scalar features permit omitting the final plane."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
receipt = HERE / "compact_final_head_training_v1.json"
payload = HERE / "payload_compact_final_head_v1.npz"
if receipt.exists() or payload.exists():
    raise FileExistsError("Existing final-only training artifacts preserved")
VALID = {37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234}
P = dict(np.load(HERE / "payload_binary_compact_g01_v1.npz"))
torch.manual_seed(1753362478)
torch.set_num_threads(2)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
nets = [(r, dict(np.load(HERE / "rows_compact_final_head_v1" / f"row_{r:04d}" / "data.npz"))) for r in range(35,235)]
scale = float(P["rms"][15])


def batch(valid):
    rows = [d for r,d in nets if (r in VALID) == valid]
    X = np.concatenate([d["h"] for d in rows])
    sigma = np.concatenate([d["sigma"] for d in rows]).astype(np.float32)
    target = np.concatenate([d["target"] for d in rows]).astype(np.float32)
    old = np.concatenate([d["old_delta"] for d in rows]).astype(np.float32)
    return tuple(torch.tensor(a, dtype=torch.float32, device=dev) for a in (X,sigma,target,old))


train = batch(False); valid = batch(True)
model = torch.nn.Sequential(torch.nn.Linear(126,128), torch.nn.SiLU(), torch.nn.Linear(128,128), torch.nn.SiLU(), torch.nn.Linear(128,1)).to(dev)
gain = 1+.25*np.tanh(P["gain"][15])
with torch.no_grad():
    model[0].weight.copy_(torch.tensor(P["enc0_w"], device=dev)); model[0].bias.copy_(torch.tensor(P["enc0_b"], device=dev))
    model[2].weight.copy_(torch.tensor(P["enc2_w"], device=dev)); model[2].bias.copy_(torch.tensor(P["enc2_b"], device=dev))
    model[4].weight.copy_(torch.tensor((gain*(P["head"][15]@P["enc4_w"]))[None,:], device=dev))
    model[4].bias.copy_(torch.tensor([gain*(P["head"][15]@P["enc4_b"]+P["head_b"][15])], device=dev))
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-5)
epochs = 40; size = 8192
steps = math.ceil(len(train[0])/size)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*steps, eta_min=5e-5)
normalizer = float(torch.mean(train[2]**2).cpu())
history = []
started = time.time()


def evaluate(data):
    x,s,t,old = data
    with torch.no_grad():
        pred = s*scale*model(x)[:,0]
        return dict(mse=float(torch.mean((pred-t)**2).cpu()), old_mse=float(torch.mean((old-t)**2).cpu()), host_mse=float(torch.mean(t*t).cpu()))


print(json.dumps(dict(device=str(dev), train_nets=160, valid_nets=40, features=126, epochs=epochs, initial_valid=evaluate(valid))), flush=True)
for epoch in range(epochs):
    order = torch.randperm(len(train[0]), device=dev)
    for j in range(0, len(order), size):
        idx = order[j:j+size]
        x,s,t,_ = (v[idx] for v in train)
        pred = s*scale*model(x)[:,0]
        loss = torch.mean((pred-t)**2)/normalizer
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step(); sched.step()
    if epoch%5 == 4:
        rec = dict(epoch=epoch+1, train=evaluate(train), valid=evaluate(valid), wall_s=time.time()-started)
        history.append(rec); print(json.dumps(rec), flush=True)
Q = {}
for i in (0,2,4):
    Q[f"w{i}"] = model[i].weight.detach().cpu().numpy()
    Q[f"b{i}"] = model[i].bias.detach().cpu().numpy()
Q["rms"] = np.asarray([scale], np.float32)
np.savez(payload, **Q)
receipt.write_text(json.dumps(dict(history=history, train_nets=160, valid_nets=40, wall_s=time.time()-started, payload=str(payload),
                                   interpretation="Final-only on-policy scalar-input cost refit, not an iterative DAgger program."), indent=2)+"\n")
print("COMPACT_FINAL_HEAD_TRAINED", flush=True)
