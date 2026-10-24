#!/usr/bin/env python3
"""TAG fable-engineer (E-3): candidate = bank source + appended fragment(s). usage: build_grouped.py <out.py> <fragment.py> [<fragment2.py> ...]"""
import sys, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
bank = (HERE / "cand_bank.py").read_text()
assert hashlib.sha256(bank.encode()).hexdigest() == "d79463f389a6b6e4172415b183601d327bc13b560fd7096ddbf781dce7a94d0e"
src = bank + "".join(Path(f).read_text() for f in sys.argv[2:])
Path(sys.argv[1]).write_text(src)
print(sys.argv[1], hashlib.sha256(src.encode()).hexdigest())
