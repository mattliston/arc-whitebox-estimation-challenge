"""TAG opus_engineer a01: response-only floor dial (accuracy probe, NOT a bill lever). RRPI + one edit of the response select.
usage: build.py <floor>  -> candidate_f<floor>.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from buildlib import build, RRPI, RRPI_SHA
floor = sys.argv[1]
OLD = "            response_indices, response_mask = select(response_score, gate_floor)\n"
NEW = "            response_indices, response_mask = select(response_score, %s if l >= 1 else gate_floor)\n" % floor
build(RRPI, Path(__file__).resolve().parent/('candidate_f%s.py' % floor.replace('.', 'p')), edits=[(OLD, NEW, 1)], expect_parent_sha=RRPI_SHA)
