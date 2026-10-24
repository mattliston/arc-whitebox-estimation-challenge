"""One natural-coefficient coadaptation round, fixed original training protocol."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
runner=HERE/"chain_radial_training.py"
source=runner.read_text()
edits={
    "rows_radial_covmetric_v1":"rows_compact_radialfeed_v1",
    "ASTRA_RADIALCOV_V1_C64_E40":"ASTRA_COMPACT_RADIALFEED_V1_C64_E40",
    "chain_radial_training_v1.json":"chain_compact_radialfeed_training_v1.json",
    "payload_radialcov_v1.npz":"payload_compact_radialfeed_v1.npz",
    'str(HERE/"generate_radial_rows.py")':'str(HERE/"generate_compact_radialfeed_rows_v1.py")',
    'frozen_names = ("k3_radial_numpy.py","radial_features.py","generate_radial_rows.py")':
        'frozen_names = ("k3_radial_numpy.py","radial_cubic.py","radial_binary_compact.py","radial_compact_radialfeed_v1.py","radial_features.py","generate_radial_rows.py","generate_compact_radialfeed_rows_v1.py")',
    "row_seconds = time.time()-started":
        'profile=[sum(r["one_step_mse"][l] for r in rows)/len(rows) for l in range(16)]\nassert 1e-10<profile[-1]<2*profile[10],("unhealthy honest rows",profile)\nprint("RADIALFEED_HONEST_PROFILE",profile,flush=True)\nrow_seconds = time.time()-started',
}
for old,new in edits.items():
    assert old in source,old
    source=source.replace(old,new)
exec(compile(source,str(runner),"exec"),dict(__file__=str(runner),__name__="__main__"))
