"""Exact zero-head CP1792/H64/C96 V3 host-fidelity control meter source."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import flopscope.numpy as fnp
from whestbench import BaseEstimator
from whestbench.domain import MLP


_SOURCE = Path(__file__).with_name("estimator_generalized.py")
_SPEC = importlib.util.spec_from_file_location("v3_control_cp1792_h64_body", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load generalized V3 observer body")
_BODY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BODY)
_BODY._CP_CAP = 1792
_BODY._HIDDEN = 64
_BODY._CELL_WIDTH = 96


class EstimatorRecurrent(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        del budget
        return _BODY._predict(mlp, recurrent=True)


class EstimatorLocal(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        del budget
        return _BODY._predict(mlp, recurrent=False)
