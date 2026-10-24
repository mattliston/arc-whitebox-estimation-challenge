"""Honest mean-reset rows for one fixed signed-pair cap."""
import os
from pathlib import Path
import runpy
import k3_radial_numpy
from radial_binary_cap import make_predict

k3_radial_numpy.predict = make_predict(int(os.environ["BINARY_PAIR_CAP"]))
runpy.run_path(str(Path(__file__).resolve().parent / "generate_radial_rows.py"), run_name="__main__")
