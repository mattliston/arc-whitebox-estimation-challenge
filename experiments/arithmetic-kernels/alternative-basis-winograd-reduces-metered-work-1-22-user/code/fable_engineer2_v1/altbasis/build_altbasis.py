#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-AB build): --altbasis = alternative-basis in-place tree Winograd; appends altbasis/ab_flag_fragment.py to a
built candidate that already has the engineer's --ipt tree kernel (stack it AFTER --ipt). usage: build_altbasis.py <in.py> <out.py>"""
import sys
from pathlib import Path
if __name__ == "__main__":
    src = Path(sys.argv[1]).read_text(); out = Path(sys.argv[2])
    for need in ("\ndef _ipt_side(", "\ndef _ipt_side_p(", "\ndef _ipt_bits(", "\ndef _ipt_ok(", "\n_IPT_OLD_BB = _bb_product", "\ndef _w5_port_product_body("):
        assert need in src, f"base lacks {need.strip()[:40]!r} (needs --ipt)"
    assert "_AB_ON" not in src, "base already has --altbasis"
    assert not out.exists(), f"refusing to overwrite {out}"
    frag = (Path(__file__).resolve().parent / "ab_flag_fragment.py").read_text()
    out.write_text(src.rstrip("\n") + "\n" + frag)
    print(f"--altbasis: {sys.argv[1]} -> {sys.argv[2]}")
