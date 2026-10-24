"""TAG opus_engineer d01 (DIAGNOSTIC, not a candidate): RRPI with every fast-product kernel switched off (_WINOGRAD = False appended),
to measure how much of the final MSE is float32 rounding noise of the Winograd kernels. Bill roughly doubles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from buildlib import build, RRPI, RRPI_SHA
build(RRPI, Path(__file__).resolve().parent/'candidate_plain.py', append="# OPUS-ENGINEER d01 diagnostic: plain float32 matmul everywhere the dispatchers honour the switch\n_WINOGRAD = False\n", expect_parent_sha=RRPI_SHA)
