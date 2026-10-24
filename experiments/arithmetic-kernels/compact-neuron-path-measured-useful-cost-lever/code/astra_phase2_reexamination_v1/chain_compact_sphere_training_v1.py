"""One matched angular-trajectory refit: fixed 200 nets, 160/40, C64, 40 epochs.

Unlearned Gaussian-target MSE improved 1.6%/4.0% on r35/r100 with no extra
carried state. This is only a necessary condition; the existing Gaussian
head is not evidence for the changed trajectory. Run one honest matched
round, then judge in the full closed loop. No grid or adaptive epochs.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
path = HERE / "chain_radial_training.py"
source = path.read_text()
edits = {
    "rows_radial_covmetric_v1": "rows_compact_sphere_v1",
    "ASTRA_RADIALCOV_V1_C64_E40": "ASTRA_COMPACT_SPHERE_V1_C64_E40",
    "chain_radial_training_v1.json": "chain_compact_sphere_training_v1.json",
    "payload_radialcov_v1.npz": "payload_compact_sphere_v1.npz",
    'str(HERE/"generate_radial_rows.py")': 'str(HERE/"generate_compact_sphere_rows_v1.py")',
    'frozen_names = ("k3_radial_numpy.py","radial_features.py","generate_radial_rows.py")':
        'frozen_names = ("k3_radial_numpy.py","radial_cubic.py","radial_binary_compact.py","radial_compact_radialfeed_v1.py","radial_mehler.py","radial_compact_sphere_v1.py","radial_features.py","generate_radial_rows.py","generate_compact_sphere_rows_v1.py")',
    "row_seconds = time.time()-started":
        'profile=[sum(r["one_step_mse"][l] for r in rows)/len(rows) for l in range(16)]\nassert 1e-10<profile[-1]<2*profile[10],("unhealthy honest rows",profile)\nassert all(r["saved_host_and_sigma_frame"]=="Gaussian" for r in rows)\nprint("SPHERE_HONEST_PROFILE_GAUSSIAN_UNITS",profile,flush=True)\nrow_seconds = time.time()-started',
}
for old, new in edits.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
exec(compile(source, str(path), "exec"), dict(__file__=str(path), __name__="__main__"))
