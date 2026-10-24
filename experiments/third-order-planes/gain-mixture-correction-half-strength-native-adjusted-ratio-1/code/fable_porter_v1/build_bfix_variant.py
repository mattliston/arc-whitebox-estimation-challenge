#!/usr/bin/env python3
"""TAG fable-porter (PT-1): native port of the lead's GAIN-MIXTURE b FIX (reference: fable_refit_v1/model_bank_bfix.py Card.step).
In the live loop (the LAST definition of _predict_binary), right after the carried third-cumulant vector
    b = g*bpre + radial*d*vv/6
it adds
    b = b + _BFIX_GAMMA * (radial_new - radial) / 6 * m
radial = incoming (pre-activation) isotropic fourth-cumulant coefficient; radial_new = this layer's post-activation coefficient,
already including the omitted-radial source; m = post-activation mean AFTER the learned mean hook (m = m + dm). Same expression
and the same place as the torch card: before trace_post, so the fix reaches the innovation, the births and the star.
Guarded by `if _BFIX_GAMMA:` so gamma 0 runs the bank's exact op sequence. Cost per layer (15 layers): 2n FLOPs + 3 scalar ops,
5 flopscope requests. Module global _BFIX_GAMMA is appended at the end of the file.
usage: build_bfix_variant.py <base_cand.py> <gamma> <out.py>   (refuses to overwrite <out.py>)"""
import sys, ast, json, hashlib
from pathlib import Path

B_LINE = ("            b = ((g * bpre + radial * d * vv / 6) if _FULL_LOWER_MOMENTS\n"
          "                 else (g * bpre + radial * d * vv / 6) * alive)\n")
FIX = ("            if _BFIX_GAMMA:\n"
       "                b = b + _BFIX_GAMMA * (radial_new - radial) / 6 * m\n")
# statements the fix depends on, each exactly once in the live loop and in this order
ORDER = ("            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)\n",
         "                radial_new = radial_new + source_sum / denominator\n",
         "                m = m + dm\n",
         B_LINE,
         "            trace_post = fnp.diag(Bpost)[:, None] * b[None, :] + 2 * Bpost * b[:, None]\n",
         "            radial = radial_new\n")


def live_loop_span(src):
    defs = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == '_predict_binary']
    node = defs[-1]
    starts = [0]
    for line in src.split('\n'):
        starts.append(starts[-1] + len(line) + 1)
    return starts[node.lineno - 1], starts[node.end_lineno]


def patch(src, gamma):
    assert '_BFIX_GAMMA' not in src, 'base already carries the b fix'
    lo, hi = live_loop_span(src)
    body = src[lo:hi]
    pos = []
    for s in ORDER:
        assert body.count(s) == 1, (s.strip(), body.count(s))
        pos.append(body.index(s))
    assert pos == sorted(pos), pos
    assert src.count(B_LINE) == 1
    k = lo + pos[3] + len(B_LINE)
    out = (src[:k] + FIX + src[k:] +
           "\n\n# fable-porter PT-1: gain-mixture b fix, b += _BFIX_GAMMA*(radial_new-radial)/6*m in the live loop (0 = bank).\n"
           f"_BFIX_GAMMA = {float(gamma)!r}\n")
    ast.parse(out)
    return out


if __name__ == '__main__':
    base, gamma, out = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
    text = patch(base.read_text(), gamma)
    with out.open('x') as f:
        f.write(text)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    print(json.dumps(dict(out=str(out), gamma=gamma, base=str(base), base_sha256=sha(base.read_bytes()),
                          out_sha256=sha(out.read_bytes()), bytes=len(text),
                          fix_line=text[:text.index(FIX)].count('\n') + 1)))
