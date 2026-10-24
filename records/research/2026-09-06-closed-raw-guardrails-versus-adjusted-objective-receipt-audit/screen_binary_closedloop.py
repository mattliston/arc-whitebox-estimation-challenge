"""Reuse the strict mean-head closed-loop protocol with the two-cube host."""
from pathlib import Path
import runpy
import k3_radial_numpy
from radial_binary_cubic import predict
k3_radial_numpy.predict=predict
runpy.run_path(str(Path(__file__).resolve().parent/"screen_radial_closedloop.py"),run_name="__main__")
