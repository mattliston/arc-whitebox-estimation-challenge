"""Why the torch unit-cost model misattributed the L13 vs L10 bill (lead task (3), 2026-09-25).

Product-level model of the native old tier (stack_S27s_mem.py: _shared_old384_push, _slr_*), in m*k*n units,
using per-net shapes from the torch logs (alive/resp counts, joined columns; VALID40, fixed384 arm).
Native L13 join is LAZY (_SLR_LAZY_LAYERS=(13,)): the L7+L10 columns stay in the L10 frame for L13-15, so the
prefix reconstruct (1024 x r10 x C10) and prefix read (resp x C10 x r10) scale with r10, not r13.
One constant per component (reconstruct / read_old / push_old_basis / join) is fitted to engineer2's S27 census
(net 100: 17.50 / 11.52 / 1.46 / 8.80 U; old tier = 27.7% of the bill). The two native schedule bills
(S27s 384/384/320 -0.688%, S27r 384/320/320 -3.167%, VALID40) are NOT used in the fit: they test it.
Usage: python3 r320_lazy_model.py
"""
import json
import numpy as np

CENSUS = dict(recon=17.50, read=11.52, basis=1.46, join=8.80)   # U, engineer2 bill27 census, S27 net 100
OLD_SHARE = 39.28 / 141.97                                        # old tier share of the bill, same census


def parts(r, sched, lazy13):
    r7, r10, r13 = sched
    J = {j['layer']: j['joined'] for j in r['joins']}
    C = {7: J[7], 10: J[7] + J[10], 13: J[7] + J[10] + J[13]}
    out = {k: {7: 0.0, 10: 0.0, 13: 0.0} for k in CENSUS}
    for l in range(7, 16):
        seg = 7 if l < 10 else 10 if l < 13 else 13
        rk, cols = {7: r7, 10: r10, 13: r13}[seg], C[seg]
        live = r['counts'][str(l - 1)]['alive']
        lazy = lazy13 and seg == 13
        if l in (7, 10, 13):
            w, rp = J[l], {7: 0, 10: r7, 13: r10}[l]
            j = live * w * rk + 2 * rk * live * w // 2 + rk * w * rk + live * rk * rk      # Y, Fj, joined core, QR
            if rp:
                j += live * rp * rp + rp * rp * rk + live * rp * rk + rk * live * rp + rp * rp * rk + rk * rp * rk
                if not (l == 13 and lazy13):
                    j += rk * rp * C[{10: 7, 13: 10}[l]]                                      # materialized rotation
            out['join'][seg] += j
        out['basis'][seg] += 1024 * live * rk
        if lazy:
            out['recon'][seg] += 1024 * r13 * r10 + 1024 * r10 * C[10] + 1024 * r13 * J[13]
        else:
            out['recon'][seg] += 1024 * rk * cols
        if l <= 14:
            resp, idx = r['counts'][str(l)]['resp'], r['counts'][str(l)]['alive']
            if lazy:
                out['read'][seg] += resp * C[10] * r10 + resp * J[13] * r13 + resp * r10 * r13 + resp * r13 * idx
            else:
                out['read'][seg] += resp * cols * rk + resp * rk * idx
    return out


def main():
    nets = [r for r in map(json.loads, open('r320_features_v40.jsonl')) if r['arm'] == 'fixed384']
    for lazy13 in (True, False):
        base = [parts(r, (384, 384, 384), lazy13) for r in nets]
        scale = {k: CENSUS[k] / np.mean([sum(b[k].values()) for b in base]) for k in CENSUS}   # U per m*k*n
        tot = lambda p: sum(scale[k] * v for k in p for v in p[k].values())
        seg = {s: np.mean([sum(scale[k] * b[k][s] for k in b) for b in base]) for s in (7, 10, 13)}
        print(f"\n{'LAZY L13 join (native)' if lazy13 else 'MATERIALIZED L13 join (torch model assumption)'}")
        print(f"  segment U at 384: L7-9 {seg[7]:.2f}  L10-12 {seg[10]:.2f}  L13-15 {seg[13]:.2f}   (census 9.59 / 14.82 / 14.87)")
        for sched, native in (((384, 384, 320), -0.688), ((384, 320, 320), -3.167), ((384, 320, 384), None),
                              ((320, 384, 384), None), ((384, 352, 384), None), ((352, 384, 384), None),
                              ((384, 384, 352), None), ((384, 352, 320), None), ((352, 352, 320), None),
                              ((384, 384, 288), None), ((320, 320, 320), None)):
            d = np.mean([(tot(parts(r, sched, lazy13)) - tot(b)) / tot(b) for r, b in zip(nets, base)]) * OLD_SHARE
            print(f"  {'/'.join(map(str, sched))}: model bill {100 * d:+.3f}%" + (f"   native {native:+.3f}%" if native is not None else ''))


if __name__ == '__main__':
    main()
