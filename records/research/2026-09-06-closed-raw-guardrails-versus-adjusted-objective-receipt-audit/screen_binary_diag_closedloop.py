"""Strict mean-head protocol on the coadapted diagonal-feedback host."""
from pathlib import Path
import runpy
import k3_radial_numpy
from radial_binary_diagfeedback import predict
k3_radial_numpy.predict=predict
runpy.run_path(str(Path(__file__).resolve().parent/"screen_radial_closedloop.py"),run_name="__main__")
