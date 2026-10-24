"""Reuse the honest row protocol with the frozen two-cube host."""
from pathlib import Path
import runpy
import k3_radial_numpy
from radial_binary_cubic import predict
k3_radial_numpy.predict=predict
runpy.run_path(str(Path(__file__).resolve().parent/"generate_radial_rows.py"),run_name="__main__")
