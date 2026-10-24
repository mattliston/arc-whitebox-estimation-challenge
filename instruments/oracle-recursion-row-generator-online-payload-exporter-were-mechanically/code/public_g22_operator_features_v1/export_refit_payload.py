#!/usr/bin/env python3
"""TAG g22-export-refit-payload (T-105): a refit operator's state_dict (REFIT_*_MODEL.pt, keys = the payload keys with
trunk.N.weight/bias) -> payload npz for estimator_retro's RETRO_TRANSPORT_PAYLOAD (closed-loop instrument).
Usage: export_refit_payload.py <model.pt> <out.npz>"""
import sys
import numpy as np
import torch
m = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = m["state_dict"] if isinstance(m, dict) and "state_dict" in m else m
out = {k.replace("trunk.0.", "trunk0_").replace("trunk.2.", "trunk2_"): np.asarray(v.detach().cpu().numpy(), dtype=np.float32) for k, v in sd.items()}
need = {"centers", "scales", "layer_rms", "trunk0_weight", "trunk0_bias", "trunk2_weight", "trunk2_bias", "source_weight", "source_bias", "gate_weight", "gate_bias", "transport_delta", "mix_delta"}
assert set(out) == need, (set(out) ^ need)
np.savez(sys.argv[2], **out); print("exported", sys.argv[2], {k: v.shape for k, v in out.items()})
