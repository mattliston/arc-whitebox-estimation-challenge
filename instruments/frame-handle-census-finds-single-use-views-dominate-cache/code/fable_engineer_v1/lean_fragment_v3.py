

# BEGIN FABLE-ENGINEER LEAN ROUND TRIPS (E-45 v3, 2026-09-23; bit-exact): fewer requests and free trips WITHOUT extending any array's
# lifetime past its function or kernel call. v2 held a layer's temporaries to the layer boundary: round trips -1.8k per net but live-array
# peak +1.47 GB and server CPU +9-13 s (bulk frees and a bigger live set), so v3 keeps only: the Winograd kernels' level-1 quadrants and
# output corners by fnp.split (3 requests per operand instead of 4 getitems; builder edits), the read5 packet slices held for the kernel
# call (builder edit), and local holds in select(), _shared_old384_finite() and _sym_upper(), which free their temporaries together when
# the call returns (one hidden round trip instead of one per temporary). Same arrays, same ops, same order: bit-exact.
def select(score, floor):
    # = the bank's select; its temporaries are freed together when it returns
    keep = []
    ge = score >= floor
    nz = fnp.nonzero(ge)
    keep.append((ge, nz))
    count = min(len(score), (int(nz[0].shape[0]) + 31) // 32 * 32)
    neg = -score
    order = fnp.argsort(neg, kind='stable')
    keep.append((neg, order))
    indices = order[:count]
    mask = fnp.zeros(len(score), dtype=fnp.bool_)
    fnp.put(mask, indices, True)
    return (indices, mask)


def _shared_old384_finite(value, name):
    # = the shared-old384 finite guard; its temporaries are freed together when it returns
    if str(value.dtype) != 'float32':
        raise RuntimeError('Shared-old384 nonfinite or non-FP32 ' + name)
    finite = fnp.isfinite(value)
    ok = fnp.all(finite)
    if not bool(ok):
        raise RuntimeError('Shared-old384 nonfinite or non-FP32 ' + name)
    del finite, ok
    return value


if '_sym_upper' in globals():
    def _sym_upper(A, B):
        # = --symout's covariance upper blocks; the views and block products are freed together when it returns
        m = int(A.shape[0]); h = m // 2
        out = fnp.empty((m, m), dtype=fnp.float32)
        a_top = A[:h]
        top = _mm(a_top, B)
        o_top = out[:h, :]
        fnp.copyto(o_top, top)
        o_low, t_right = out[h:, :h], top[:, h:]
        t_right_T = t_right.T
        fnp.copyto(o_low, t_right_T)
        a_bot, b_right = A[h:], B[:, h:]
        part = _mm(a_bot, b_right)
        o_br = out[h:, h:]
        fnp.copyto(o_br, part)
        return out
# END FABLE-ENGINEER LEAN ROUND TRIPS
