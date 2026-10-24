#!/usr/bin/env python3
"""Export a train_gnn_oracle.py state_dict to the fnp closed-loop payload (RETRO_GNN_PAYLOAD). Usage: export_gnn_payload.py <GNN_MODEL.pt> <out.npz>"""
import sys, numpy as np, torch
sd = torch.load(sys.argv[1], map_location="cpu")
f = lambda k: np.asarray(sd[k].detach().cpu().numpy(), dtype=np.float32)
out = {"centers": f("centers"), "scales": f("scales"), "rms": f("rms"), "msg_w": f("msg.weight"), "msg_b": f("msg.bias"),
       "enc0_w": f("enc.0.weight"), "enc0_b": f("enc.0.bias"), "enc2_w": f("enc.2.weight"), "enc2_b": f("enc.2.bias"), "enc4_w": f("enc.4.weight"), "enc4_b": f("enc.4.bias"),
       "head": f("head"), "head_b": f("head_b"), "gain": f("gain"),
       "first": np.array([int(__import__("os").environ.get("EXPORT_FIRST", "2"))], dtype=np.float32)}   # T-116f: first corrected layer
if "msgv.weight" in sd and float(sd["vrms"].abs().sum()) > 0 and __import__("os").environ.get("EXPORT_VHEAD", "1") == "1":
    # T-116o: the next-layer variance head (only when trained: vrms != ones marks it)
    if not bool((sd["vrms"] == 1.0).all()):
        out.update({"msgv_w": f("msgv.weight"), "msgv_b": f("msgv.bias"), "vh0_w": f("vh.0.weight"), "vh0_b": f("vh.0.bias"), "vh2_w": f("vh.2.weight"), "vh2_b": f("vh.2.bias"), "vrms": f("vrms")})
        if "msgk.weight" in sd and not bool((sd["krms"] == 1.0).all()):   # T-116u skew/kurtosis heads (trained)
            out.update({"msgk_w": f("msgk.weight"), "msgk_b": f("msgk.bias"), "kh0_w": f("kh.0.weight"), "kh0_b": f("kh.0.bias"), "kh2_w": f("kh.2.weight"), "kh2_b": f("kh.2.bias"), "krms": f("krms")})
        if "ah.0.weight" in sd:   # T-116x covariance-level rank-2 head
            out.update({"ah0_w": f("ah.0.weight"), "ah0_b": f("ah.0.bias"), "ah2_w": f("ah.2.weight"), "ah2_b": f("ah.2.bias"), "arms": f("arms")})
        if "vu.weight" in sd:   # T-116q bilinear channels
            out.update({"vu_w": f("vu.weight"), "vu_b": f("vu.bias"), "vw_w": f("vw.weight"), "vw_b": f("vw.bias"), "vbil_w": f("vbil_out.weight"), "vbil_b": f("vbil_out.bias")})
np.savez(sys.argv[2], **out); print("exported", sys.argv[2], {k: v.shape for k, v in out.items()})
