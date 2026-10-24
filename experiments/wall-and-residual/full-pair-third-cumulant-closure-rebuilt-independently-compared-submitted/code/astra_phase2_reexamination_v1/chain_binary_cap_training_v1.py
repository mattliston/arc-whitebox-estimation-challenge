"""One bounded cap-specific coadaptation using the frozen 160/40 split."""
import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--cap", type=int, choices=(4096, 6144), required=True)
args = ap.parse_args()
os.environ["BINARY_PAIR_CAP"] = str(args.cap)
source = (HERE / "chain_radial_training.py").read_text()
replacements = {
    "rows_radial_covmetric_v1": f"rows_binary_paircap{args.cap}_v1",
    "ASTRA_RADIALCOV_V1_C64_E40": f"ASTRA_BINARY_PAIRCAP{args.cap}_V1_C64_E40",
    "chain_radial_training_v1.json": f"chain_binary_paircap{args.cap}_training_v1.json",
    "payload_radialcov_v1.npz": f"payload_binary_paircap{args.cap}_v1.npz",
    'str(HERE/"generate_radial_rows.py")': 'str(HERE/"generate_binary_cap_rows.py")',
    'frozen_names = ("k3_radial_numpy.py","radial_features.py","generate_radial_rows.py")':
    'frozen_names = ("k3_radial_numpy.py","radial_cubic.py","radial_binary_cubic.py","radial_binary_cap.py","radial_features.py","generate_radial_rows.py","generate_binary_cap_rows.py")',
    "row_seconds = time.time()-started":
    'profile = [sum(r["one_step_mse"][l] for r in rows)/len(rows) for l in range(16)]\n'
    'assert 1e-10 < profile[-1] < 2*profile[10], ("unhealthy honest-row profile",profile)\n'
    'print("PAIRCAP_HONEST_PROFILE",profile,flush=True)\nrow_seconds = time.time()-started',
}
for old, new in replacements.items():
    assert old in source, old
    source = source.replace(old, new)
exec(compile(source, str(HERE / "chain_radial_training.py"), "exec"),
     dict(__file__=str(HERE / "chain_radial_training.py"), __name__="__main__"))
