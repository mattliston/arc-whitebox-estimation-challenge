"""Inert CPU physical-arithmetic alignment of the frozen shared384 proposal.

This is NOT a new candidate, a fit, a native-gradient certificate, or a promise
of output fidelity.  The original model/operator/numerics remain untouched.
`model(reviewed_sha256)` returns a separately compiled, source-inverse-qualified
copy of model.py with the same authenticate/runtime/frozen/card API.  Call its
authenticate(shared_operator_sha256) BEFORE any numerical import, as before.

Only compact covariance association, compact-star repeated plane/diagonal, and
the native raw-moment/pair table order are aligned.  CPU scalar state is FP64;
native pair accumulation is FP32 except the first Gaussian layer.  All matrix
products here remain Torch GEMM, not native Winograd/resident arena kernels.
The already-qualified Torch CDF callback is unchanged, not claimed native-byte
equivalent.  Shared QR/GEMM, explicit newest legs, P/NODE/cstar and selectors are
unchanged.  No MPS execution, targets, optimization, or launch is implemented.
"""
import ast
import hashlib
import math
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
MODEL = HERE / 'model.py'
C2 = NEW / 'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
PINS = {
    MODEL: '65805a9ffbdb310fb374b82279b79d2f0d0931b00f45191fbb3d377b10802311',
    HERE / 'shared_operator.py': '09136ff645869df9d8fae3e2e8c16ff65c777971c308d1ba8c124d35649d2ad2',
    C2: 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea',
    NEW / 'full_carrier_autodiff_v1.py': 'd6480f443e807688dca380d19d34e7cfe5c1f7e625104aae87201cfba7d199f7',
    NEW / 'centered_torch_numerics_v1.py': '2b62c3bf08ec59f7d0f749fc4526417c0c2952ed274b8aa5cba67762ff7929cf',
    NEW / 'centered_torch_numerics_v3.py': '18df077006603d1b434216b2a4ffebc1b1d1e631983163fa0a9213eb0a11412e',
    NEW / 'gaussian_cdf_quadrature_v1.py': 'f6f96fbc8f47bd4ffab93ba9cafc805b2ea79435f5a2106649d94a3bc0754398',
}
NATIVE_FUNCTIONS = ('gaussian_raw', 'derivatives', '_cov_compact', '_star_compact',
    '_nonlinear_pair_batch', '_nonlinear_first_gaussian', '_nonlinear_terminal_live',
    '_resident_gram512')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _replace(source, edits):
    """Whole-text inverse with recorded positions, never basename reversal."""
    original = source
    spans = []
    for before, after in edits:
        require(source.count(before) == 1, 'Unique physical source splice: ' + before[:80])
        at = source.index(before)
        source = source[:at] + after + source[at+len(before):]
        spans.append((at, before, after))
    inverse = source
    for at, before, after in reversed(spans):
        require(inverse[at:at+len(after)] == after, 'Positional physical inverse span')
        inverse = inverse[:at] + before + inverse[at+len(after):]
    require(inverse == original, 'Complete physical source inverse')
    return source, dict(original_sha256=_digest(original), derived_sha256=_digest(source),
                        whole_source_inverse=True, exact_splices=len(edits))


def source_certificate(reviewed_sha256):
    """Stdlib-only source binding; no model/array import or saved output read."""
    require(not {'numpy', 'torch', 'flopscope', 'threadpoolctl'} & set(sys.modules),
            'Physical source admission precedes numerical imports')
    bindings = {str(path): expected for path, expected in PINS.items()}
    bindings[str(Path(__file__).resolve())] = reviewed_sha256
    require(all(sha(path) == expected for path, expected in bindings.items()),
            'Exact reviewed physical-alignment sources')
    raw = C2.read_bytes()
    lines = raw.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    owners = {}
    for node in ast.parse(raw.decode()).body:
        if isinstance(node, ast.FunctionDef) and node.name in NATIVE_FUNCTIONS:
            require(node.name not in owners, 'Unique actual native physical owner')
            block = raw[offsets[node.lineno-1]+node.col_offset:
                        offsets[node.end_lineno-1]+node.end_col_offset]
            owners[node.name] = dict(line=node.lineno, end_line=node.end_lineno,
                                    sha256=hashlib.sha256(block).hexdigest())
    require(set(owners) == set(NATIVE_FUNCTIONS), 'Complete native physical owner roster')
    return dict(source_bindings=bindings, native_function_owners=owners,
                targets_accessed=False, numerical_imports=False)


def _cpu(torch, anchor):
    require(anchor.device.type == 'cpu' and anchor.dtype == torch.float64,
            'This prospective physical alignment is CPU FP64 scalar state only')


def _diagonal(torch, matrix, values):
    # Functional overwrite, not matrix-diag(old)+diag(new) reassociation.
    i = torch.arange(matrix.shape[0], device=matrix.device)
    return matrix.index_put((i, i), values)


def _resident_gram(torch, left, right):
    """Native three-block512 association/mirror, but Torch GEMM internally."""
    require(tuple(left.shape[:1]) == (1024,) and right.shape[1] == 1024,
            'Exact current resident Gram geometry')
    aa = left[:512, :] @ right[:, :512]
    ab = left[:512, :] @ right[:, 512:]
    bb = left[512:, :] @ right[:, 512:]
    return torch.cat((torch.cat((aa, ab), dim=1),
                      torch.cat((ab.T, bb), dim=1)), dim=0)


def covariance(torch, layer, w, C, previous, previous_alive):
    _cpu(torch, C)
    require(w.dtype == torch.float32 and w.device == C.device,
            'Original FP32 physical weights')
    if layer == 0:
        value = _resident_gram(torch, w.T, w).to(torch.float64)
        return value, value
    wa = w[previous, :]
    ca = C[previous[:, None], previous[None, :]].to(torch.float32)
    result = wa.T @ (ca @ wa)
    active_metric = result.to(torch.float64)
    if previous.numel() < w.shape[0]:
        cold = torch.nonzero(~previous_alive.bool(), as_tuple=True)[0]
        wc = w[cold, :]
        diagonal = torch.diagonal(C)[cold].to(torch.float32)
        result = result + _resident_gram(torch, (wc * diagonal[:, None]).T, wc)
    return result.to(torch.float64), active_metric


def star(torch, layer, w, A, previous, previous_alive, current):
    require(w.device.type == 'cpu' and w.dtype == A.dtype == torch.float32,
            'Original CPU FP32 star/weight operands')
    n = w.shape[1]
    if layer == 0:
        return torch.zeros((n, n), dtype=torch.float64, device=w.device)
    wp = w[previous, :]
    aa = A[previous[:, None], previous[None, :]].to(torch.float32)
    pushed = wp.T @ aa
    wr, pr = wp[:, current].T, pushed[current, :]
    sub = (2 * ((pr * wr) @ wr.T) + (wr * wr) @ pr.T) / 3
    diagonal = torch.sum(pushed * wp.T * wp.T, dim=1, dtype=torch.float64)
    if previous.numel() < w.shape[0]:
        cold = torch.nonzero(~previous_alive.bool(), as_tuple=True)[0]
        wc = w[cold, :]
        ad = torch.diagonal(A)[cold].to(torch.float32)
        wc_current = wc[:, current]
        sub = sub + (wc_current * wc_current).T @ (wc_current * ad[:, None])
        diagonal = diagonal + torch.sum(wc * wc * wc * ad[:, None], dim=0,
                                        dtype=torch.float64)
    result = torch.zeros((n, n), dtype=torch.float64, device=w.device)
    result = result.index_put((current[:, None], current[None, :]), sub.to(torch.float64))
    return _diagonal(torch, result, diagonal)


def _gaussian_raw(torch, mu, v, cdf):
    v = torch.maximum(v, v.new_tensor(1e-10))
    s = torch.sqrt(v)
    a = mu / s
    pdf = torch.exp(-a * a / 2) / torch.sqrt(mu.new_tensor(2 * math.pi))
    g = cdf(a)  # Deliberately the unchanged, explicitly qualified Torch CDF.
    m = s * pdf + mu * g
    r2 = (mu * mu + v) * g + mu * s * pdf
    r3 = mu * (mu * mu + 3 * v) * g + (mu * mu + 2 * v) * s * pdf
    r4 = (mu ** 4 + 6 * mu * mu * v + 3 * v * v) * g + mu * (mu * mu + 5 * v) * s * pdf
    return s, a, pdf, g, m, r2, r3, r4


def _derivatives(torch, a, pdf, g, mg_over_s):
    A = [mg_over_s, g]
    h0, h1 = torch.ones_like(a), a
    for r in range(2, 7):
        if r == 2:
            h = h0
        elif r == 3:
            h = h1
        else:
            h = a * h1 - (r - 3) * h0
            h0, h1 = h1, h
        A.append((-1) ** (r - 2) * h * pdf)
    B = [(a * a + 1) * g + a * pdf, 2 * mg_over_s, 2 * g]
    for r in range(3, 7):
        B.append(2 * A[r-1])
    return A, B


def pair(torch, layer, mu, C, K21, radial, cdf):
    """Current raw-moment/FP32 pair law; no centered-pair replacement."""
    _cpu(torch, mu)
    require(C.dtype == K21.dtype == mu.dtype and C.device == K21.device == mu.device,
            'Original same-device FP64 pair inputs')
    require(0 <= layer <= 15, 'Original sixteen-layer dispatch')
    n = mu.numel()
    v = torch.diagonal(C).clone()
    k4 = radial * v * v
    k22 = radial * (torch.outer(v, v) + 2 * C * C) / 3
    s, a, pdf, g, mg, r2g, r3g, r4g = _gaussian_raw(torch, mu, v, cdf)
    d, e = pdf / s, -a * pdf / v
    f = (a * a - 1) * pdf / (v * s)
    k3 = torch.diagonal(K21).clone()
    m = mg + k3 * e / 6 + k4 * f / 24
    r2 = r2g + k3 * d / 3 + k4 * e / 12
    r3 = r3g + k3 * g + k4 * d / 4
    r4 = r4g + 4 * k3 * mg + k4 * g
    var = r2 - m * m
    post_k3 = r3 - 3 * m * r2 + 2 * m ** 3
    post_k4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * var * var
    zero = torch.zeros_like(v)
    rho = _diagonal(torch, C / torch.outer(s, s), zero)
    T21 = _diagonal(torch, K21 / (v[:, None] * s[None, :]), zero)
    T22 = k22 / torch.outer(v, v)
    t3, t4 = k3 / s ** 3, k4 / v ** 2
    T31 = t4[:, None] * rho
    A, B = _derivatives(torch, a, pdf, g, mg / s)
    aa, bb = torch.stack(A, dim=0), torch.stack(B, dim=0)
    if layer == 15:
        F, G = torch.stack([aa], dim=0), torch.stack([aa], dim=0)
    else:
        F, G = torch.stack([aa, bb, bb], dim=0), torch.stack([aa, aa, bb], dim=0)
    accumulator_dtype = torch.float64 if layer == 0 else torch.float32
    ans = torch.zeros((1 if layer == 15 else 3, n, n),
                      dtype=accumulator_dtype, device=mu.device)
    if layer:
        F, G = F.to(torch.float32), G.to(torch.float32)
        rho, T21, T22 = rho.to(torch.float32), T21.to(torch.float32), T22.to(torch.float32)
        T31, t3, t4 = T31.to(torch.float32), t3.to(torch.float32), t4.to(torch.float32)
    rr = torch.ones_like(rho)
    for r in range(7):
        if r:
            ans = ans + rr[None, :, :] * (F[:, r, :, None] * G[:, r, None, :])
        if layer and r <= 2:
            ec = (T21[None, :, :] * (F[:, r+2, :, None] * G[:, r+1, None, :])
                  + T21.T[None, :, :] * (F[:, r+1, :, None] * G[:, r+2, None, :])) / 2
            ec = ec + T22[None, :, :] * (F[:, r+2, :, None] * G[:, r+2, None, :]) / 4
            ec = ec + (T31[None, :, :] * (F[:, r+3, :, None] * G[:, r+1, None, :])
                       + T31.T[None, :, :] * (F[:, r+1, :, None] * G[:, r+3, None, :])) / 6
            if r:
                ec = ec + ((t3[None, :] * F[:, r+3, :])[:, :, None] * G[:, r, None, :]
                           + F[:, r, :, None] * (t3[None, :] * G[:, r+3, :])[:, None, :]) / 6
                ec = ec + ((t4[None, :] * F[:, r+4, :])[:, :, None] * G[:, r, None, :]
                           + F[:, r, :, None] * (t4[None, :] * G[:, r+4, :])[:, None, :]) / 24
            ans = ans + rr[None, :, :] * ec
        rr = rr * rho / (r + 1)
    require(ans.dtype == rr.dtype == accumulator_dtype,
            'Exact first-Gaussian/later-FP32 accumulator and power table precision')
    ans = ans.to(torch.float64)
    cov = ans[0] * torch.outer(s, s)
    aux = dict(sigma=s, alpha=a, pdf=pdf, gate=g, gaussian_mean=mg, pre_v=v,
        pre_cov=C, pre_k3=k3, pre_k4=k4.clone(), post_v=var,
        post_k3=post_k3, post_k4=post_k4,
        # The inherited BD/radial source asks for this centered-name alias.
        # Its value is the native raw expression, NOT the old centered formula.
        birth_beta=2 * (mg - m * g))
    if layer == 15:
        return m, _diagonal(torch, cov, var), None, mu.new_zeros(()), aux
    pk21 = ans[1] * torch.outer(v, s)
    pk22 = ans[2] * torch.outer(v, v)
    post21 = pk21 - 2 * m[:, None] * cov
    post22 = pk22 - 2 * m[:, None] * pk21.T - 2 * m[None, :] * pk21
    post22 = post22 + 4 * torch.outer(m, m) * cov - 2 * cov * cov
    post22 = _diagonal(torch, post22, post_k4)
    cov = _diagonal(torch, cov, var)
    post21 = _diagonal(torch, post21, post_k3)
    new_radial = 3 * torch.sum(post22) / (torch.trace(cov) ** 2 + 2 * torch.sum(cov * cov))
    return m, cov, post21, new_radial, aux


STEP_EDITS = (
    ('    Cpre = ((w.T@C.float())@w).to(dt)\n'
     '    metric = diagonal(C, torch.diagonal(C)*previous_alive)\n'
     '    Bpre = ((w.T@metric.float())@w).to(dt)\n',
     '    Cpre, Bpre = physical_covariance(torch, l, w, C, previous_indices, previous_alive)\n'),
    ('    wt = w.T\n    wa = wt@star\n'
     '    sp = (2*(wa*wt)@wt.T+(wt*wt)@wa.T).to(dt)/3\n'
     '    sp = diagonal(sp*live, torch.diagonal(sp))\n',
     '    sp = physical_star(torch, l, w, star, previous_indices, previous_alive, indices)\n'),
    ('    host, C, target, new_radial, aux = nonlinear(mu, Cpre, k21, radial)\n',
     '    host, C, target, new_radial, aux = physical_pair(torch, l, mu, Cpre, k21, radial, normal_cdf)\n'),
)


def align_step(source):
    result, proof = _replace(source, STEP_EDITS)
    proof.update(current_CP_and_shared_algebra_unchanged=True,
        P_NODE_cstar_and_selectors_unchanged=True, CPU_scalar_FP64=True,
        native_pair_accumulator_dtype=True, compact_covariance_association=True,
        compact_star_FP64_diagonal=True, native_Winograd_reproduced=False,
        original_Torch_CDF_unchanged=True, native_output_fidelity_qualified=False)
    return result, proof


MODEL_EDITS = (
    ("    code, count = base.rewrite_clamps(source, str(__file__) + ':shared_step')\n",
     "    source, physical_proof = _physical_align_source(source)\n"
     "    code, count = base.rewrite_clamps(source, str(__file__) + ':shared_step')\n"),
    ('               node_coefficients=_JOINT._NODE_FN)\n',
     '               node_coefficients=_JOINT._NODE_FN,\n'
     '               physical_covariance=_physical_covariance, physical_star=_physical_star,\n'
     '               physical_pair=_physical_pair)\n'),
    ('        native_gradient_certified=False, dense_young_real_algebra_only=True)\n',
     '        native_gradient_certified=False, dense_young_real_algebra_only=True,\n'
     '        physical_alignment=physical_proof)\n'),
)


def derive_model():
    """Only source text; caller's independent preflight may compile the result."""
    require(sha(MODEL) == PINS[MODEL], 'Frozen shared model source')
    return _replace(MODEL.read_text(), MODEL_EDITS)


def model(reviewed_sha256):
    """Create the additive model API, before authenticate or numerical runtime.

    Typical owner sequence (not executed by this source):
        aligned = alignment.model(ACTUAL_REVIEWED_ALIGNMENT_SHA)
        bindings = aligned.authenticate(ACTUAL_SHARED_OPERATOR_SHA)
        P, A = aligned.frozen('cpu', trainable_P=False)
        card = aligned.card(35, P=P, A=A, device='cpu', weights=owned_weights)

    The caller still owns all prerequisite seals, fixtures, actual CPU runtime,
    retirement, resource monitoring and the unchanged <=1e-5 all16 gate.
    """
    certificate = source_certificate(reviewed_sha256)
    text, inverse = derive_model()
    result = types.ModuleType('current_shared384_physical_alignment_model')
    # Original relative owner resolution remains exactly the frozen CAUSAL dir.
    result.__file__ = str(MODEL)
    result.__dict__.update(_physical_align_source=align_step,
        _physical_covariance=covariance, _physical_star=star, _physical_pair=pair)
    exec(compile(text, str(HERE / 'physical_alignment.py') + ':derived_model', 'exec'),
         result.__dict__)
    original_authenticate = result.authenticate

    def authenticate(shared_sha256):
        fresh = source_certificate(reviewed_sha256)
        bindings = original_authenticate(shared_sha256)
        bindings.update(fresh['source_bindings'])
        result._BINDINGS = dict(bindings)
        return dict(bindings)

    result.authenticate = authenticate
    result.physical_alignment_source_proof = dict(inverse=inverse,
        certificate=certificate, source_sha256=reviewed_sha256,
        original_model_and_operator_unmodified=True, no_numerical_execution=True)
    return result
