"""Honest pre-correction feature rows for either frozen compact host."""
import os
from pathlib import Path
import runpy
import k3_radial_numpy

mode = os.environ["BINARY_COMPACT_MODE"]
if mode == "compact":
    from radial_binary_compact import predict as base_predict
    def predict(W, **kwargs):
        return base_predict(W, gate_floor=.01, **kwargs)
elif mode == "sparsecarry":
    from radial_binary_sparsecarry import predict as base_predict
    def predict(W, **kwargs):
        return base_predict(W, gate_floor=.01, birth_floor=.001, **kwargs)
else:
    raise ValueError(mode)
k3_radial_numpy.predict = predict
runpy.run_path(str(Path(__file__).resolve().parent / "generate_radial_rows.py"), run_name="__main__")
