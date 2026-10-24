"""Honest row protocol, diagonal-feedback binary host; no live target features."""
from pathlib import Path
import runpy
import k3_radial_numpy
from radial_binary_diagfeedback import predict
k3_radial_numpy.predict=predict
runpy.run_path(str(Path(__file__).resolve().parent/"generate_radial_rows.py"),run_name="__main__")
