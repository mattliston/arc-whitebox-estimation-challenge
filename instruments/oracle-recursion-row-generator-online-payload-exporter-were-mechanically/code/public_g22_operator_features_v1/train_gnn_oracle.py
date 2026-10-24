#!/usr/bin/env python3
"""TAG g22-gnn-oracle (T-114): a LEARNED CLOSURE OVER THE WEIGHT GRAPH for the one-step error.
Rows = the oracle recursion (rows_meanleak_mech: tape raw20 (16,1024,20), mech (16,1024,31), one-step host, sigma);
targets = the organizers' 1e9 all-layer means minus the one-step host (per unit per layer). Model = a recurrent
message-passing network across layers: node state e^{l} at every unit from [tape, mech, messages from e^{l-1} through
W_l, W_l^2, W_l^3 (gate-weighted)]; per-layer output head -> delta_i = sigma_i * rms_l * head_l(e_i^l). Same 160/40
split as run_refit_ext.py (valid ranks = EXPECTED_VALID_GLOBAL) so the valid-40 final MSE compares with the refits
(20-feature teacher-forced refit: 8.93e-9, 60% removal). Trains on MPS in float32.
Usage: train_gnn_oracle.py <rows_dir> <tag> [epochs] [channels]"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ROWS = Path(sys.argv[1]); TAG = sys.argv[2]; EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
C = int(sys.argv[4]) if len(sys.argv) > 4 else 32
VALID = {37, 39, 42, 48, 59, 63, 66, 67, 71, 73, 75, 79, 93, 101, 122, 130, 132, 135, 139, 145, 146, 149, 150, 151, 155, 159, 163, 169, 173, 175, 181, 192, 198, 199, 204, 219, 224, 227, 233, 234}
import os as _os_fp
FIRST = int(_os_fp.environ.get("GNN_FIRST", "2"))                 # T-116f: first corrected layer (the oracle rows correct layers 1-15; the loop must match)
_FINAL_PUSH = _os_fp.environ.get("GNN_FINAL_PUSH") == "1"          # T-115d: fill the final layer's zero plane columns with pushes of layer 14's
FINAL_ZERO_COLS = [5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 21, 22]      # age0-3, older, energy, k21_row, k21_col, cp4_k31_row, cp4_k22_row, k22_row, k31_row
_ABLATE = _os_fp.environ.get("GNN_ABLATE_COLS", "")                # T-115g: e.g. "5,6,7,8,9,11" ; layers via GNN_ABLATE_LAYERS "13,14"
_ABLATE_COLS = [int(c) for c in _ABLATE.split(",") if c]
_ABLATE_LAYERS = [int(l) for l in _os_fp.environ.get("GNN_ABLATE_LAYERS", "14").split(",") if l]
_EXPOSURE = _os_fp.environ.get("GNN_EXPOSURE") == "1"                # T-116b: two exposure-state columns per layer (fdim 53)
_COVFEAT = _os_fp.environ.get("GNN_COVFEAT") == "1"                  # T-116m: four pre-covariance contraction columns per layer
_VLABELS = _os_fp.environ.get("GNN_VLABELS", "")                   # T-116o: dir of mps_truth_rank{r}_P1M.npz -> next-layer variance head
_VLOSS = float(_os_fp.environ.get("GNN_VLOSS", "1.0"))
_KLABELS = _os_fp.environ.get("GNN_KLABELS") == "1"                  # T-116u: next-layer skew/kurtosis heads (labels from the same vlabels files)
_KLOSS = float(_os_fp.environ.get("GNN_KLOSS", "1.0"))
_LOSS_MASK = None                                                   # T-115h: GNN_LOSS_LAYERS="15" trains the loss on those layers only
if _os_fp.environ.get("GNN_LOSS_LAYERS"):
    _LOSS_MASK = torch.zeros(16 - FIRST)
    for _l in _os_fp.environ["GNN_LOSS_LAYERS"].split(","):
        _LOSS_MASK[int(_l) - FIRST] = 1.0
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1753362478)

man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]


def load_rows():
    nets = []
    dirs = [ROWS] + [Path(x) for x in _os_fp.environ.get("GNN_EXTRA_ROWS", "").split(",") if x]   # T-116: dataset aggregation (DAgger)
    for d in sorted(p for rd in dirs for p in rd.glob("row_*")):
        if not (d / "result.json").exists():
            continue
        rank = int(d.name.split("_")[1])
        tape = np.load(d / "raw20.npy").astype(np.float32); mech = np.load(d / "mech.npy").astype(np.float32) if (d / "mech.npy").exists() else np.zeros((16, 1024, 0), np.float32)
        host = np.load(d / "host.npy").astype(np.float64); sigma = np.load(d / "sigma.npy").astype(np.float32)
        truth = np.load(HERE / "truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
        wpath = STORE / Path(man[rank]["path"]).parent / "weights.npy"
        if _FINAL_PUSH and mech.shape[2] == 31:
            # T-115d: the final layer has no planes / K21 read, so its 12 plane-derived mech columns are zero; fill them with the
            # W^2-weighted fan-in average of layer 14's columns (the previous layer's plane structure pushed through W15)
            W2 = np.asarray(np.load(wpath, mmap_mode="r")[15], dtype=np.float64) ** 2
            mech[15][:, FINAL_ZERO_COLS] = ((W2.T @ mech[14][:, FINAL_ZERO_COLS].astype(np.float64)) / (W2.sum(0)[:, None] + 1e-30)).astype(np.float32)
        if _EXPOSURE:
            # T-116b exposure state: the previous layer's APPLIED correction pushed through this layer's weights (W^T d / sigma,
            # (W^2)^T d^2 / sigma^2); oracle rows have no delta.npy -> the applied correction there is the target itself
            delta = np.load(d / "delta.npy").astype(np.float64) if (d / "delta.npy").exists() else (truth - host)
            Wm = np.load(wpath, mmap_mode="r"); ex = np.zeros((16, 1024, 2), np.float32)
            for l in range(1, 16):
                Wl = np.asarray(Wm[l], dtype=np.float64); s2 = sigma[l].astype(np.float64) ** 2 + 1e-30
                ex[l, :, 0] = (Wl.T @ delta[l - 1]) / np.sqrt(s2); ex[l, :, 1] = ((Wl * Wl).T @ (delta[l - 1] ** 2)) / s2
            mech = np.concatenate([mech, ex], axis=2)
        if _COVFEAT:
            # T-116m: four pre-covariance contractions per unit per layer (covfeat.npy from the row generator; zeros if absent)
            _cff = _os_fp.environ.get("GNN_COVFEAT_FILE", "covfeat.npy")   # T-137: any extra per-unit feature block (e.g. puffifeat.npy, 20 cols)
            cf = np.load(d / _cff).astype(np.float32) if (d / _cff).exists() else np.zeros((16, 1024, 4), np.float32)
            mech = np.concatenate([mech, cf], axis=2)
        if _ABLATE and mech.shape[2] == 31:
            # T-115g ablation: zero the chosen mech columns at the chosen interior layers (does the interior cliff without them?)
            for l in _ABLATE_LAYERS:
                mech[l][:, _ABLATE_COLS] = 0.0
        dvl = np.zeros((16, 1024), np.float32)
        if _VLABELS:
            vf = Path(_VLABELS) / f"mps_truth_rank{rank:04d}_P1M.npz"
            if not vf.exists(): continue                      # T-116o: only nets with variance labels
            vt = np.load(vf)["mc_pre_v"].astype(np.float64); vc = sigma.astype(np.float64) ** 2
            dvl = ((vt - vc) / (vc + 1e-30)).astype(np.float32)   # relative pre-variance error per unit per layer
            if _KLABELS:
                zz = np.load(vf); vc64 = vc
                k3t = zz["mc_pre_k3"].astype(np.float64); k4t = zz["mc_pre_k4"].astype(np.float64)
                k3c = mech[:, :, 1].astype(np.float64) * vc64 ** 1.5; k4c = mech[:, :, 17].astype(np.float64)
                r3 = np.sqrt(np.mean(k3c ** 2, axis=1, keepdims=True)) + 1e-30; r4 = np.sqrt(np.mean(k4c ** 2, axis=1, keepdims=True)) + 1e-30   # the CARD's per-layer rms (known at deploy)
                dkl = np.stack([(k3t - k3c) / r3, (k4t - k4c) / r4], axis=2).astype(np.float32)   # (16,1024,2) errors in units of the card's per-layer rms
                krms = np.concatenate([r3, r4], axis=1).astype(np.float32)                        # (16,2) the per-layer rms of the truth
        nets.append({"rank": rank, "x": np.concatenate([tape, mech], axis=2), "sigma": sigma, "target": (truth - host).astype(np.float32), "mpost": truth.astype(np.float32),
                     "wpath": wpath, "valid": rank in VALID, "dv": dvl, "dk": dkl if (_VLABELS and _KLABELS) else np.zeros((16, 1024, 2), np.float32),
                     "krms": krms if (_VLABELS and _KLABELS) else np.ones((16, 2), np.float32)})
    return nets


class GNN(torch.nn.Module):
    def __init__(self, fdim, C, hidden=int(_os_fp.environ.get("GNN_HIDDEN", "128")), cm=int(_os_fp.environ.get("GNN_CM", "16")), depth=16):   # T-123: widths via env
        super().__init__()
        self.C, self.cm = C, cm
        self.centers = torch.nn.Parameter(torch.zeros(depth, fdim), requires_grad=False)
        self.scales = torch.nn.Parameter(torch.ones(depth, fdim), requires_grad=False)
        self.rms = torch.nn.Parameter(torch.ones(depth), requires_grad=False)
        self.msg = torch.nn.Linear(C, 3 * cm)                     # three edge powers, cm channels each
        self.enc = torch.nn.Sequential(torch.nn.Linear(fdim + 3 * cm + 3 * cm, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, C))
        self.head = torch.nn.Parameter(torch.zeros(depth, C)); self.head_b = torch.nn.Parameter(torch.zeros(depth))
        self.gain = torch.nn.Parameter(torch.zeros(depth))
        # T-116o next-layer variance head: from e_l pushed through W_{l+1} (cm channels + energies) -> dv_{l+1}/v_{l+1}
        vcm = int(_os_fp.environ.get("GNN_VCM", "16")); vhid = int(_os_fp.environ.get("GNN_VHID", "32")); self.vcm = vcm   # T-116p: v-head width
        self.msgv = torch.nn.Linear(C, vcm); self.vh = torch.nn.Sequential(torch.nn.Linear(2 * vcm, vhid), torch.nn.SiLU(), torch.nn.Linear(vhid, 1))
        self.vbil = int(_os_fp.environ.get("GNN_VBILIN", "0"))                 # T-116q: k bilinear pushed-product channels
        self.acov = _os_fp.environ.get("GNN_ACOV") == "1"                       # T-116x: covariance-level mean-aligned rank-2 head
        if self.acov:
            self.ah = torch.nn.Sequential(torch.nn.Linear(C, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)); self.arms = torch.nn.Parameter(torch.ones(depth), requires_grad=False)
        if self.vbil:
            self.vu = torch.nn.Linear(C, self.vbil); self.vw = torch.nn.Linear(C, self.vbil); self.vbil_out = torch.nn.Linear(self.vbil, 1)
        self.vrms = torch.nn.Parameter(torch.ones(depth), requires_grad=False)
        # T-116u next-layer skew/kurtosis heads (same structure as the variance head): dk3_{l+1}, dk4_{l+1} in units of the truth's per-layer rms
        self.msgk = torch.nn.Linear(C, vcm); self.kh = torch.nn.Sequential(torch.nn.Linear(2 * vcm, vhid), torch.nn.SiLU(), torch.nn.Linear(vhid, 2))
        self.krms = torch.nn.Parameter(torch.ones(depth, 2), requires_grad=False)

    def forward(self, x, sigma, W):
        # x (16,1024,F), sigma (16,1024), W list of (1024,1024) tensors (W[l] maps layer l-1 -> l)
        depth, n, _ = x.shape
        e_prev = torch.zeros(n, self.C, device=x.device); out = []; dvout = [torch.zeros(n, device=x.device) for _ in range(depth)]
        dkout = [torch.zeros(n, 2, device=x.device) for _ in range(depth)]
        aout = [torch.zeros(n, device=x.device) for _ in range(depth)]; dvcov = [torch.zeros(n, device=x.device) for _ in range(depth)]
        for l in range(depth):
            xl = torch.clamp((x[l] - self.centers[l]) / self.scales[l], -8.0, 8.0)
            if l < FIRST:
                e_prev = torch.zeros(n, self.C, device=x.device); out.append(torch.zeros(n, device=x.device)); continue
            g = xl[:, 5:6]                                            # the gate feature (index 5 of raw20)
            src = self.msg(e_prev)                                    # (n, 3cm)
            Wl = W[l]; W2 = Wl * Wl; W3 = W2 * Wl
            m1 = Wl.T @ src[:, :self.cm]; m2 = W2.T @ src[:, self.cm:2 * self.cm]; m3 = W3.T @ src[:, 2 * self.cm:]
            en1 = W2.T @ (src[:, :self.cm] ** 2); en2 = (W2 * W2).T @ (src[:, self.cm:2 * self.cm] ** 2); en3 = (W3 * W3).T @ (src[:, 2 * self.cm:] ** 2)
            msgs = torch.cat([g * m1, g * m2, g * m3, torch.log1p(torch.clamp(en1, min=0)), torch.log1p(torch.clamp(en2, min=0)), torch.log1p(torch.clamp(en3, min=0))], dim=1)
            e = self.enc(torch.cat([xl, torch.clamp(msgs, -8.0, 8.0)], dim=1))
            delta = (sigma[l] * self.rms[l]) * (e @ self.head[l] + self.head_b[l]) * (1.0 + 0.25 * torch.tanh(self.gain[l]))
            out.append(delta); e_prev = e
            if l + 1 < depth:
                Wn = W[l + 1]; sv = self.msgv(e); mv = Wn.T @ sv; ev = torch.log1p(torch.clamp((Wn * Wn).T @ (sv * sv), min=0))
                dv = self.vh(torch.cat([torch.clamp(mv, -8.0, 8.0), ev], dim=1))[:, 0]
                if self.vbil:
                    pu = Wn.T @ self.vu(e); pw = Wn.T @ self.vw(e)                      # pushed factor vectors at layer l+1
                    dv = dv + self.vbil_out(torch.clamp(pu * pw, -64.0, 64.0) / (sigma[l + 1][:, None] ** 2 + 1e-12))[:, 0]
                dvout[l + 1] = self.vrms[l + 1] * dv
                if self.acov:
                    a = self.arms[l] * self.ah(e)[:, 0]; aout[l] = a                       # per-unit factor at layer l (post side)
                    mpost = self._mpost[l]                                               # the carried post mean at l (truth in oracle rows)
                    dvcov[l + 1] = 2.0 * (Wn.T @ mpost) * (Wn.T @ a) / (sigma[l + 1] ** 2 + 1e-12)
                sk = self.msgk(e); mk = Wn.T @ sk; ek = torch.log1p(torch.clamp((Wn * Wn).T @ (sk * sk), min=0))
                dkout[l + 1] = self.krms[l + 1] * self.kh(torch.cat([torch.clamp(mk, -8.0, 8.0), ek], dim=1))
        self._dv = torch.stack(dvout); self._dk = torch.stack(dkout); self._a = torch.stack(aout); self._dvcov = torch.stack(dvcov)
        return torch.stack(out)


def main():
    nets = load_rows(); tr = [n for n in nets if not n["valid"]]; va = [n for n in nets if n["valid"]]
    fdim = nets[0]["x"].shape[2]
    print(f"rows: {len(nets)} (train {len(tr)}, valid {len(va)}), features {fdim}, device {dev}", flush=True)
    X = np.concatenate([n["x"] for n in tr], axis=1)                 # (16, N*1024, F)
    model = GNN(fdim, C).to(dev)
    import os as _os
    if _os.environ.get("GNN_INIT"):
        # fine-tune from a trained model (scheduled sampling): keep its normalization so the features mean the same thing
        _sd = torch.load(_os.environ["GNN_INIT"], map_location=dev); _own = model.state_dict()
        _sd = {k: v for k, v in _sd.items() if k in _own and tuple(v.shape) == tuple(_own[k].shape)}   # T-116p: skip shape-mismatched (re-sized) parts
        model.load_state_dict(_sd, strict=False); print("initialized from", _os.environ["GNN_INIT"], "|", len(_sd), "tensors", flush=True)
    else:
        with torch.no_grad():
            model.centers.copy_(torch.tensor(X.mean(1))); model.scales.copy_(torch.tensor(X.std(1) + 1e-6))
            model.rms.copy_(torch.tensor(np.sqrt(np.mean(np.concatenate([n["target"] for n in tr], axis=1) ** 2, axis=1)) + 1e-12))
            if _VLABELS: model.vrms.copy_(torch.tensor(np.sqrt(np.mean(np.concatenate([n["dv"] for n in tr], axis=1) ** 2, axis=1)) + 1e-12))
            if _VLABELS and _KLABELS: model.krms.copy_(torch.tensor(np.sqrt(np.mean(np.concatenate([n["dk"] for n in tr], axis=1) ** 2, axis=1)) + 1e-12))
            if _VLABELS and model.acov: model.arms.fill_(1e-3)   # T-116x: a ~ 1e-3 per unit gives induced relative dv of the label's order
    if _os.environ.get("GNN_VFREEZE") == "1":   # T-116p: keep the variance head fixed while the mean head co-adapts to it
        for _prm in list(model.msgv.parameters()) + list(model.vh.parameters()): _prm.requires_grad_(False)
        print("variance head frozen", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(_os.environ.get("GNN_LR", "1e-3")), weight_decay=2e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(tr), eta_min=float(_os.environ.get("GNN_LR_MIN", "1e-4")))
    lw = torch.linspace(0.5, 1.5, 16 - FIRST).to(dev)

    def tensors(n):
        W = np.load(n["wpath"], mmap_mode="r")
        model._mpost = torch.tensor(n["mpost"], device=dev)
        return (torch.tensor(n["x"], device=dev), torch.tensor(n["sigma"], device=dev), [torch.tensor(np.asarray(W[l], dtype=np.float32), device=dev) for l in range(16)], torch.tensor(n["target"], device=dev), torch.tensor(n["dv"], device=dev), torch.tensor(n["dk"], device=dev))

    def evaluate(ns):
        model.eval(); fin = []; layer = np.zeros(16); host = np.zeros(16); vres = [np.zeros(16), np.zeros(16)]; kres = [np.zeros((16, 2)), np.zeros((16, 2))]; ares = [np.zeros(16), np.zeros(16)]
        with torch.no_grad():
            for n in ns:
                x, s, W, t, dv, dk = tensors(n); p = model(x, s, W)
                layer += ((p - t) ** 2).mean(1).cpu().numpy(); host += (t ** 2).mean(1).cpu().numpy()
                if _VLABELS: vres[0] += ((model._dv - dv) ** 2).mean(1).cpu().numpy(); vres[1] += (dv ** 2).mean(1).cpu().numpy()
                if _VLABELS and _KLABELS: kres[0] += ((model._dk - dk) ** 2).mean(1).cpu().numpy(); kres[1] += (dk ** 2).mean(1).cpu().numpy()
                if _VLABELS and model.acov: ares[0] += ((model._dvcov - dv) ** 2).mean(1).cpu().numpy(); ares[1] += (dv ** 2).mean(1).cpu().numpy()
        model.train(); evaluate.v = (vres[0] / len(ns), vres[1] / len(ns)); evaluate.k = (kres[0] / len(ns), kres[1] / len(ns)); evaluate.a = (ares[0] / len(ns), ares[1] / len(ns)); return layer / len(ns), host / len(ns)

    t0 = time.time(); rng = np.random.default_rng(40)
    for ep in range(EPOCHS):
        order = rng.permutation(len(tr)); tot = 0.0
        for i in order:
            x, s, W, t, dv, dk = tensors(tr[i]); p = model(x, s, W)
            per = ((p[FIRST:] - t[FIRST:]) ** 2).mean(1) / (model.rms[FIRST:] ** 2)
            loss = (per * lw).mean() + per[-1] if _LOSS_MASK is None else (per * _LOSS_MASK.to(per.device)).sum() / _LOSS_MASK.sum()
            if _VLABELS: loss = loss + _VLOSS * (((model._dv[FIRST:] - dv[FIRST:]) ** 2).mean(1) / (model.vrms[FIRST:] ** 2)).mean()
            if _VLABELS and _KLABELS: loss = loss + _KLOSS * (((model._dk[FIRST:] - dk[FIRST:]) ** 2).mean(1) / (model.krms[FIRST:] ** 2)).mean()
            if _VLABELS and model.acov: loss = loss + _VLOSS * (((model._dvcov[FIRST:] - dv[FIRST:]) ** 2).mean(1) / (model.vrms[FIRST:] ** 2)).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step(); tot += float(loss)
        if ep % 5 == 4 or ep == EPOCHS - 1:
            lv, hv = evaluate(va)
            print(f"epoch {ep+1}: train loss {tot/len(tr):.4f} | valid final {lv[15]:.3e} vs one-step host {hv[15]:.3e} (removal {100*(1-lv[15]/hv[15]):.1f}%) | L8 {100*(1-lv[8]/hv[8]):.0f}% L12 {100*(1-lv[12]/hv[12]):.0f}% L14 {100*(1-lv[14]/hv[14]):.0f}% | {time.time()-t0:.0f}s", flush=True)
            if _VLABELS: vr, vh = evaluate.v; print(f"   variance head held-out removal of the relative v error: " + " ".join(f"L{l} {100*(1-vr[l]/max(vh[l],1e-30)):.0f}%" for l in (4, 8, 12, 14, 15)), flush=True)
            if _VLABELS and model.acov: ar, ah_ = evaluate.a; print(f"   covariance-level (rank-2 mean-aligned) head, induced dv removal: " + " ".join(f"L{l} {100*(1-ar[l]/max(ah_[l],1e-30)):.0f}%" for l in (8, 12, 14, 15)), flush=True)
            if _VLABELS and _KLABELS: kr, kh = evaluate.k; print(f"   k3 head: " + " ".join(f"L{l} {100*(1-kr[l,0]/max(kh[l,0],1e-30)):.0f}%" for l in (8, 12, 15)) + " | k4 head: " + " ".join(f"L{l} {100*(1-kr[l,1]/max(kh[l,1],1e-30)):.0f}%" for l in (8, 12, 15)), flush=True)
    lv, hv = evaluate(va); lt, ht = evaluate(tr)
    res = {"tag": TAG, "rows": str(ROWS), "n_train": len(tr), "n_valid": len(va), "features": fdim, "channels": C, "epochs": EPOCHS,
           "valid_layer_mse": lv.tolist(), "valid_host_layer_mse": hv.tolist(), "valid_final": float(lv[15]), "valid_removal_final": float(1 - lv[15] / hv[15]),
           "train_final": float(lt[15]), "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (HERE / f"GNN_{TAG}_RESULT.json").write_text(json.dumps(res, indent=1)); torch.save(model.state_dict(), HERE / f"GNN_{TAG}_MODEL.pt")
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}), flush=True)


if __name__ == "__main__":
    main()
