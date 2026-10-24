#!/usr/bin/env python3
"""TAG g22-retro-f64 (T-131 PRECISION TEST): run the instrument with EVERY fnp.float32 site promoted to float64 (monkeypatch the flopscope
numpy module attribute before the estimator is loaded; billing 2x is irrelevant for this accuracy read). Same CLI/env as run_retro_screen.py."""
import sys, runpy
import flopscope.numpy as fnp
fnp.float32 = fnp.float64          # every `fnp.float32` reference in the card now resolves to float64
import flopscope as _fl
_BC = _fl.BudgetContext
def _bc(*a, **k):
    k['flop_budget'] = 2 ** 44          # accuracy read only: f64 is billed 2x, so the deployed card would not fit in 2^41
    return _BC(*a, **k)
_fl.BudgetContext = _bc
sys.argv = [str(__import__("pathlib").Path(__file__).resolve().parent.parent / "public_g20_emission_retrofit_v1/run_retro_screen.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
