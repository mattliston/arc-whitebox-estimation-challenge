#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-SP): flopscope_server under cProfile; dumps stats to $E2_PROF_OUT on SIGTERM. Measurement only."""
import os, sys, signal, cProfile, pstats
from flopscope_server.__main__ import main
prof = cProfile.Profile()
def stop(*_):
    prof.disable(); prof.dump_stats(os.environ['E2_PROF_OUT']); os._exit(0)
signal.signal(signal.SIGTERM, stop)
prof.enable()
main()
