"""Inert fixed five-fold terminal information math; no imports or file access.

An admitted offline worker supplies its qualified native FNP API and TRAIN-only
arrays. This module cannot load targets, fit at import, create a candidate or
publish a result. The dense FP32 held-output circuit, not an expanded-dot
proxy, supplies the prospective information score.
"""

RIDGE = 1e-4
FLOOR = 1e-12
HIDDEN = 64
AFFINE = 65
TOTAL = 2145
BUDGET = 2**41
AFFINE_INCREMENT = 1_000_000
QUADRATIC_INCREMENT = 10_000_000


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def finite(np, value, name):
    require(bool(np.all(np.isfinite(value))), 'Finite '+name)
    return value


def score_weights(np, own_cold_flops):
    """The absolute cost multiplier, never a ratio to another network's F."""
    require(own_cold_flops.shape == (20,) and own_cold_flops.dtype == np.int64
            and bool(np.all((own_cold_flops > 0) & (own_cold_flops <= BUDGET))),
            'Exactly twenty independently measured complete cold model bills')
    return np.maximum(0.1,own_cold_flops.astype(np.float64)/BUDGET)


def frame(np, hidden, indices, network_weight):
    """TRAIN-fold-only normalization, rounded once to inference FP32."""
    require(hidden.shape == (20,1024,HIDDEN) and hidden.dtype == np.float32,
            'Actual unchanged H15 twenty-network feature frame')
    finite(np,hidden,'captured H15')
    require(len(indices) == 16 and len(set(indices)) == 16, 'Sixteen fitting networks')
    h64 = hidden[indices].reshape(-1,HIDDEN).astype(np.float64)
    weights = np.repeat(network_weight[indices],1024).astype(np.float64)
    weights = weights/np.sum(weights,dtype=np.float64,keepdims=True)
    center64 = weights @ h64
    centered = h64-center64[None,:]
    scale64 = np.maximum(np.sqrt(weights @ (centered*centered)),FLOOR)
    center,scale = center64.astype(np.float32),scale64.astype(np.float32)
    require(bool(np.all(scale > 0)), 'Positive once-rounded hidden scales')
    normalized = (hidden-center[None,None,:])/scale[None,None,:]
    require(normalized.dtype == np.float32, 'Inference FP32 hidden normalization')
    finite(np,normalized,'standardized H15')
    return normalized,weights,dict(center=center,scale=scale)


def dictionary(np, normalized, sigma, indices):
    """Real monomials of the exact FP32 x, multiplied by exact sigma32."""
    require(normalized.dtype == np.float32 and normalized.shape == (20,1024,HIDDEN)
            and sigma.shape == (20,1024) and sigma.dtype == np.float64,
            'Qualified hidden/sigma capture types')
    x = normalized[indices].reshape(-1,HIDDEN).astype(np.float64)
    s32 = sigma[indices].astype(np.float32)
    require(bool(np.all(s32 > 0)), 'Positive actual FP32 sigma; no target division')
    s = s32.reshape(-1).astype(np.float64)
    parts = [s[:,None],x*s[:,None]]
    col = AFFINE
    for i in range(HIDDEN):
        width = HIDDEN-i
        parts.append((x[:,i:i+1]*x[:,i:])*s[:,None])
        col += width
    require(col == TOTAL, 'All2080 fixed upper-triangle monomials')
    design = np.concatenate(parts,axis=1)
    return finite(np,design,'complete sigma-scaled2145-column design')


def gram(np, design, weights, residual, *, expected_rows=16384):
    """One shared Gram; its affine principal block uses identical weights/RMS."""
    require(expected_rows in (16,16384), 'Only a fixed full fold or target-free unit fixture')
    require(design.dtype == weights.dtype == residual.dtype == np.float64
            and design.shape == (expected_rows,TOTAL)
            and weights.shape == residual.shape == (expected_rows,), 'One fixed fitting fold')
    finite(np,residual,'TRAIN-only raw residual')
    rms = np.maximum(np.sqrt(weights @ (design*design)),FLOOR)
    normalized = design/rms[None,:]
    weighted = normalized*weights[:,None]
    matrix = normalized.T @ weighted
    rhs = normalized.T @ (weights*residual)
    finite(np,matrix,'shared Gram'); finite(np,rhs,'shared RHS')
    require(bool(np.all(np.diag(matrix) <= 1.0+1e-10)), 'Normalized diagonal <=1')
    # A single explicit symmetrization removes asymmetric accumulation noise;
    # it is fixed, accounted offline work, not a rescue after factor failure.
    matrix = (matrix+matrix.T)*0.5
    return matrix,rhs,rms


def triangular_solve(np, lower, rhs):
    """Two fixed triangular sweeps; no second factorization or solver fallback."""
    n = len(rhs)
    forward = np.empty(0,dtype=np.float64)
    for i in range(n):
        value = (rhs[i:i+1]-lower[i:i+1,:i] @ forward)/lower[i:i+1,i]
        forward = np.concatenate([forward,value])
    answer = np.empty(0,dtype=np.float64)
    for i in range(n-1,-1,-1):
        value = (forward[i:i+1]-lower[i+1:,i][None,:] @ answer)/lower[i:i+1,i]
        answer = np.concatenate([value,answer])
    return answer


def solve(np, matrix, rhs, rms, quadratic):
    """Exactly one1e-4 ridge Cholesky for this model and fold."""
    n = TOTAL if quadratic else AFFINE
    system = matrix[:n,:n]+np.eye(n,dtype=np.float64)*RIDGE
    target = rhs[:n].copy()
    factor = np.linalg.cholesky(system)
    require(bool(np.all(np.diag(factor) > 0)), 'Positive Cholesky diagonal; no rescue')
    normalized = triangular_solve(np,factor,target)
    defect = system @ normalized-target
    error = np.sqrt(np.sum(defect*defect,keepdims=True))
    denominator = np.maximum(np.sqrt(np.sum(target*target,keepdims=True)),1e-30)
    relative = error/denominator
    require(bool(np.all(relative <= 1e-9)), 'Original relative normal-equation residual gate')
    relative_residual = float(relative[0])
    theta = finite(np,normalized/rms[:n],'finite solved monomial coefficients')
    payload = dict(b=theta[:1].astype(np.float32),
                   u=theta[1:AFFINE].astype(np.float32))
    if quadratic:
        # Indices/halves depend on declared dimensions only, never array data.
        columns = {}
        col = AFFINE
        for i in range(HIDDEN):
            for j in range(i,HIDDEN):
                columns[i,j] = columns[j,i] = col
                col += 1
        require(col == TOTAL, 'Exact fixed monomial-to-symmetric convention')
        indices = np.array([[columns[i,j] for j in range(HIDDEN)] for i in range(HIDDEN)],dtype=np.int64)
        halves = np.array([[1.0 if i == j else 0.5 for j in range(HIDDEN)] for i in range(HIDDEN)],dtype=np.float64)
        symmetric = np.take(theta,indices)*halves
        payload['S'] = symmetric.astype(np.float32)
    for name,value in payload.items():
        finite(np,value,'FP32 '+name)
    return payload,dict(columns=n,ridge=RIDGE,relative_normal_equation_residual=relative_residual,
                        cholesky_count=1,solver_rescue=False)


def served(np, normalized, sigma, raw, payload):
    """Fixed proposed dense FP32 circuit followed by actual raw-add/clamp."""
    x = normalized.copy()
    require(x.dtype == np.float32 and x.shape == (1024,HIDDEN)
            and sigma.dtype == np.float64 and sigma.shape == (1024,)
            and raw.dtype == np.float64 and raw.shape == (1024,),
            'One captured network, original sigma and raw-return frame')
    linear = x @ payload['u']
    if 'S' in payload:
        q = np.sum((x @ payload['S'])*x,axis=1,dtype=np.float32)
        correction = ((q+linear)+payload['b'])*sigma.astype(np.float32)
    else:
        correction = (linear+payload['b'])*sigma.astype(np.float32)
    require(correction.dtype == np.float32, 'Exact planned FP32 dense inference grouping')
    final_raw = raw+correction.astype(np.float64)
    answer = np.maximum(final_raw,0.0).astype(np.float32)
    return finite(np,answer,'actual proposed served prediction')


def fit_fold(np, fold, hidden, sigma, raw, truth, own_cold_flops):
    """Returns OOF outputs only; no score computation or inspection here."""
    require(type(fold) is int and 0 <= fold < 5, 'One of five fixed ordinal folds')
    require(sigma.shape == raw.shape == truth.shape == (20,1024)
            and sigma.dtype == raw.dtype == truth.dtype == np.float64,
            'Captured FP64 sigma/raw and original FP32 truth widened by owner')
    fit = [i for i in range(20) if i % 5 != fold]
    held = [i for i in range(20) if i % 5 == fold]
    w = score_weights(np,own_cold_flops)
    x,weights,normalizer = frame(np,hidden,fit,w)
    design = dictionary(np,x,sigma,fit)
    residual = (truth[fit]-raw[fit]).reshape(-1)
    matrix,rhs,rms = gram(np,design,weights,residual)
    outputs,checks = {},{}
    for name,quadratic in (('affine',False),('quadratic',True)):
        payload,checks[name] = solve(np,matrix,rhs,rms,quadratic)
        outputs[name] = np.stack([served(np,x[i],sigma[i],raw[i],payload) for i in held])
        del payload
    return dict(fold=fold,fitting_indices=fit,held_indices=held,outputs=outputs,
                solver_checks=checks,normalizer=normalizer,
                fixed_inference_increment=dict(affine=AFFINE_INCREMENT,quadratic=QUADRATIC_INCREMENT),
                actual_paid_inference_bill=False,full_fit_payload_produced=False)
