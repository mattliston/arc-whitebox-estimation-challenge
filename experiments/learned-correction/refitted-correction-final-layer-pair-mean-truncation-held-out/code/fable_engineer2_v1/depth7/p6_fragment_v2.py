

# BEGIN FABLE-ENGINEER2 DEPTH-5 -> PADDED DEPTH-6 (p6 v2, 2026-09-25; v2 = v1 minus the k-split's domain (partition as proposed
# to the engineer 12:2x, pending their OK): a product in the read regime with k >= 2048 (n <= 1024, m <= 1024) is left to k-split v5w/v6, which prices whole and
# chunked depth-6 reads itself, so nothing is priced twice. v1 text follows. rounding-level, NOT bit-exact; lead's depth-5 padding lever). Every
# single alt-basis product reaches _ipt_product_ab. A depth-5 product whose dims, zero-padded up to multiples of 64, fit the depth-6
# regions whole (no k chunks: the k-split sits above and keeps its products) runs at depth 6 when the fitted models say it pays at the
# rule's price: bill saved >= _P6_MU x added backend seconds. Padding as _ks_product: fnp.zeros + fnp.copyto per padded operand, the
# product into an (M, N) buffer, its leading m x n window copied into out; unpadded when the dims divide (the reads and joins that sit at
# depth 5 only because every read keeps depth 5). Models fitted on fable_engineer2_v1/depth7/bench_S27_v1.jsonl (70 classes x 2 depths,
# S27's kernel, OpenBLAS env): bill beyond the leaf = 0.7669 mk S + 0.7668 kn S + 1.5213 mn S + 2.6307 (mk + kn + mn) + 0.9768 pad with
# S = ((7/4)^L - 1) / 0.75 (max residual 0.16 Mflop); backend seconds = 4.22e-4 + 1.18e-7 7^L + 9.45e-12 leaf flops + 4.63e-10
# (mk + kn + mn) S + 1.36e-9 pad + 1.73e-10 mn (7/4)^L (delta error sd 9 ms). _P6_MU = 0.3 Gflop per backend second = about 0.05% of the
# bill per platform second on S27's page (WALL v3). Priced per net on nets 35/100: -0.22% for +0.95 s (top classes) .. -0.40% for +3.7 s
# (all paying classes), BEFORE lazy binds of the new depth-6 keys (measured in the parity run, which binds as a worker does).
_P6_ON = True
_P6_MU = 3.0e8
_P6_PLANS = {}
_P6_STATS = {'moved': 0}
_P6_OLD_IPAB = _ipt_product_ab


def _p6_bill(L, m, k, n, pad):
    S = ((7 / 4) ** L - 1) / 0.75
    return (2 * m * k * n * (7 / 8) ** L - m * n * (7 / 4) ** L + 0.7669 * m * k * S + 0.7668 * k * n * S + 1.5213 * m * n * S
            + 2.6307 * (m * k + k * n + m * n) + 0.9768 * pad)


def _p6_secs(L, m, k, n, pad):
    S = ((7 / 4) ** L - 1) / 0.75
    return (4.22e-4 + 1.18e-7 * 7 ** L + 9.45e-12 * 2 * m * k * n * (7 / 8) ** L + 4.63e-10 * (m * k + k * n + m * n) * S
            + 1.36e-9 * pad + 1.73e-10 * m * n * (7 / 4) ** L)


def _p6_plan(m, k, n):
    key = (m, k, n)
    if key in _P6_PLANS:
        return _P6_PLANS[key]
    plan = None
    M, K, N = (m + 63) // 64 * 64, (k + 63) // 64 * 64, (n + 63) // 64 * 64
    if _ipt_fits(6, M, K, N) and not (n <= 1024 and m <= 1024 and k >= 2048):
        pad = ((M * K if (M, K) != (m, k) else 0) + (K * N if (K, N) != (k, n) else 0) + (m * n if (M, N) != (m, n) else 0))
        dB = _p6_bill(6, M, K, N, pad) - _p6_bill(5, m, k, n, 0)
        dT = _p6_secs(6, M, K, N, pad) - _p6_secs(5, m, k, n, 0)
        if dB < 0 and -dB >= _P6_MU * max(dT, 0.0):
            plan = (M, K, N)
    _P6_PLANS[key] = plan
    return plan


def _ipt_product_ab(a, b, out, L):
    if L == 5 and _P6_ON and len(a.shape) == 2 and len(b.shape) == 2 and len(out.shape) == 2:
        m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
        plan = _p6_plan(m, k, n)
        if plan is not None:
            M, K, N = plan
            hold = []
            if (M, K) != (m, k):
                aa = fnp.zeros((M, K), dtype=fnp.float32)
                hold.append(fnp.copyto(aa[:m, :k], a))
            else:
                aa = a
            if (K, N) != (k, n):
                bb = fnp.zeros((K, N), dtype=fnp.float32)
                hold.append(fnp.copyto(bb[:k, :n], b))
            else:
                bb = b
            _P6_STATS['moved'] += 1
            if (M, N) != (m, n):
                o = fnp.empty((M, N), dtype=fnp.float32)
                _P6_OLD_IPAB(aa, bb, o, 6)
                hold.append(fnp.copyto(out, o[:m, :n]))
            else:
                _P6_OLD_IPAB(aa, bb, out, 6)
            del hold
            return out
    return _P6_OLD_IPAB(a, b, out, L)
# END FABLE-ENGINEER2 DEPTH-5 -> PADDED DEPTH-6
