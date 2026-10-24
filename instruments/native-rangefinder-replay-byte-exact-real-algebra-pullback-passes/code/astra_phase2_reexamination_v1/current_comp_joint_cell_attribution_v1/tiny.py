"""DRAFT: fixed synthetic pair-map certificate; caller owns paid context.

No imports, file access, model call, truth, RNG, or surviving array result.
The reference uses physical derivatives and raw-moment centralization, not
the estimator's standardized/batched implementation. No launch at import.
"""

FLOP_CAP = 10_000_000
ATOL, RTOL = 2e-6, 2e-5
FACTORIAL = (1, 1, 2, 6, 24, 120, 720)
COEFFICIENTS = (0.5, 0.25, 0.16666666666666666,
                0.16666666666666666, 0.041666666666666664)


def _require(ok, message):
    if not ok:
        raise RuntimeError('JCELL tiny: '+message)


def _error(fnp, actual, expected):
    a = fnp.asarray(actual, dtype=fnp.float64)
    b = fnp.asarray(expected, dtype=fnp.float64)
    _require(tuple(a.shape) == tuple(b.shape), 'exact comparison shape')
    _require(bool(fnp.all(fnp.isfinite(a))) and bool(fnp.all(fnp.isfinite(b))),
             'finite complete comparison, including diagonals')
    difference = fnp.abs(a-b)
    tolerance = ATOL+RTOL*fnp.abs(b)
    return dict(maxabs=float(fnp.max(difference)),
                scaled=float(fnp.max(difference/tolerance)),
                within_tolerance=bool(fnp.all(fnp.less_equal(difference,tolerance))))


def _check(fnp, actual, expected):
    _require(len(actual) == len(expected) == 5, 'five pair-map return fields')
    rows = {name:_error(fnp, actual[i], expected[i]) for i,name in
            enumerate(('mean','covariance','post21','radial'))}
    for name in ('sigma','alpha','pdf','gate','gaussian_mean','pre_v','pre_cov',
                 'pre_k3','pre_k4','post_v','post_k3','post_k4'):
        rows['aux.'+name] = _error(fnp, actual[4][name], expected[4][name])
    _require(all(row['within_tolerance'] for row in rows.values()),
             'fixed direct-substitution tolerance')
    _require(bool(fnp.all(actual[0] >= 0)) and
             bool(fnp.all(fnp.diag(actual[1]) > 0)), 'positive tiny output domain')
    return rows


def _fixture(fnp):
    s = fnp.array((0.75,1.0,1.25,1.5), dtype=fnp.float64)
    a = fnp.array((-0.5,0.25,0.5,-0.25), dtype=fnp.float64)
    # AR(1) correlation with coefficient 3/4: strictly positive definite.
    rho = fnp.array(((1,.75,.5625,.421875),(.75,1,.75,.5625),
                     (.5625,.75,1,.75),(.421875,.5625,.75,1)), dtype=fnp.float64)
    t21 = fnp.array(((.125,.0625,-.03125,.09375),
                    (-.046875,-.09375,.03125,-.0625),
                    (.015625,-.078125,.0625,.046875),
                    (-.0625,.03125,-.015625,-.125)), dtype=fnp.float64)
    t22 = fnp.array(((.1875,.0625,-.03125,.046875),
                    (.0625,.125,-.015625,.03125),
                    (-.03125,-.015625,-.0625,.078125),
                    (.046875,.03125,.078125,.25)), dtype=fnp.float64)
    t31 = fnp.array(((.1875,.125,-.0625,.09375),
                    (-.03125,.125,.0625,-.125),
                    (.09375,-.046875,-.0625,.03125),
                    (.0625,.015625,-.09375,.25)), dtype=fnp.float64)
    v = s*s
    C = rho*fnp.outer(s,s)
    T = t21*(v[:,None]*s[None,:])
    Q22 = t22*fnp.outer(v,v)
    Q31 = t31*(s[:,None]**3*s[None,:])
    k4 = fnp.diag(Q22).copy()
    _require(bool(fnp.all(fnp.diag(C) > 1e-10)), 'no variance-floor branch')
    _require(bool(fnp.any(T != T.T)) and bool(fnp.any(Q31 != Q31.T)),
             'ordered nonsymmetric physical planes')
    _require(bool(fnp.array_equal(fnp.diag(Q31),k4)), 'fourth diagonals agree')
    return s*a,C,T,k4,Q22,Q31,s,a


def _direct(fnp, mu, C, T, k4, Q22, Q31, s, a, metric,
            gauss=6, edge=2, coefficients=COEFFICIENTS):
    """Independent FP64 physical-unit substitution, no module helpers."""
    n = 4
    # Normal CDF at the four fixed alphas, nearest high-precision constants.
    g = fnp.array((.3085375387259869,.5987063256829237,
                   .6914624612740131,.4012936743170763), dtype=fnp.float64)
    pdf = fnp.exp(-a*a/2)/fnp.sqrt(6.283185307179586)
    v = s*s
    mg = s*(pdf+a*g)
    r2g = v*((a*a+1)*g+a*pdf)
    r3g = s**3*((a**3+3*a)*g+(a*a+2)*pdf)
    r4g = v*v*((a**4+6*a*a+3)*g+(a**3+5*a)*pdf)
    k3 = fnp.diag(T).copy()
    m = mg-k3*a*pdf/(6*v)+k4*(a*a-1)*pdf/(24*v*s)
    r2 = r2g+k3*pdf/(3*s)-k4*a*pdf/(12*v)
    r3 = r3g+k3*g+k4*pdf/(4*s)
    r4 = r4g+4*k3*mg+k4*g
    var = r2-m*m
    central3 = r3-3*m*r2+3*m*m*m-m**3
    central4 = r4-4*m*r3+6*m*m*r2-4*m**3*m+m**4-3*var*var
    # Explicit Hermite polynomials, not the parent's recurrence or batch stack.
    A = (mg/s, g, pdf, -a*pdf, (a*a-1)*pdf,
         -(a**3-3*a)*pdf, (a**4-6*a*a+3)*pdf)
    B = ((a*a+1)*g+a*pdf, 2*(pdf+a*g), 2*g, 2*pdf,
         -2*a*pdf, 2*(a*a-1)*pdf, -2*(a**3-3*a)*pdf)
    # Physical derivatives of ReLU and ReLU squared, respectively.
    D1 = tuple(s**(1-r)*A[r] for r in range(7))
    D2 = tuple(s**(2-r)*B[r] for r in range(7))
    c = fnp.copy(C); t = fnp.copy(T); q31 = fnp.copy(Q31)
    fnp.fill_diagonal(c,0); fnp.fill_diagonal(t,0); fnp.fill_diagonal(q31,0)
    ct,c22,c31,cm3,cm4 = coefficients

    def pair(F,G):
        answer = fnp.zeros((n,n),dtype=fnp.float64)
        for r in range(1,gauss+1):
            answer = answer+(c**r/FACTORIAL[r])*fnp.outer(F[r],G[r])
        for r in range(edge+1):
            term = ct*(t*fnp.outer(F[r+2],G[r+1])+
                       t.T*fnp.outer(F[r+1],G[r+2]))
            term = term+c22*Q22*fnp.outer(F[r+2],G[r+2])
            term = term+c31*(q31*fnp.outer(F[r+3],G[r+1])+
                            q31.T*fnp.outer(F[r+1],G[r+3]))
            if r:
                term = term+cm3*(k3[:,None]*fnp.outer(F[r+3],G[r])+
                                 k3[None,:]*fnp.outer(F[r],G[r+3]))
                term = term+cm4*(k4[:,None]*fnp.outer(F[r+4],G[r])+
                                 k4[None,:]*fnp.outer(F[r],G[r+4]))
            answer = answer+(c**r/FACTORIAL[r])*term
        return answer

    cov = pair(D1,D1)
    connected21 = pair(D2,D1)
    connected22 = pair(D2,D2)
    # Reconstruct raw moments, THEN centralize all substitutions together.
    raw11 = cov+fnp.outer(m,m)
    raw21 = connected21+fnp.outer(r2,m)
    raw12 = connected21.T+fnp.outer(m,r2)
    raw22 = connected22+fnp.outer(r2,r2)
    post21 = raw21-fnp.outer(r2,m)-2*m[:,None]*raw11+2*fnp.outer(m*m,m)
    post22 = (raw22-2*m[:,None]*raw12-2*m[None,:]*raw21+
              fnp.outer(m*m,r2)+fnp.outer(r2,m*m)+4*fnp.outer(m,m)*raw11-
              3*fnp.outer(m*m,m*m)-fnp.outer(var,var)-2*cov*cov)
    fnp.fill_diagonal(post22,central4)
    fnp.fill_diagonal(cov,var); fnp.fill_diagonal(post21,central3)
    denominator = fnp.asarray(n*(n+2),dtype=fnp.float64) if metric == 'identity' else (
        fnp.add(fnp.square(fnp.trace(cov)),fnp.multiply(2,fnp.sum(cov*cov))))
    _require(bool(fnp.all(fnp.asarray(denominator) > 0)), 'positive radial denominator')
    radial = fnp.divide(fnp.multiply(3,fnp.sum(post22)),denominator)
    aux = dict(sigma=s,alpha=a,pdf=pdf,gate=g,gaussian_mean=mg,pre_v=v,
               pre_cov=C,pre_k3=k3,pre_k4=k4,post_v=var,
               post_k3=central3,post_k4=central4)
    return m,cov,post21,radial,aux


def _separation(fnp, left, right):
    values = tuple(_error(fnp,left[i],right[i])['scaled'] for i in range(4))
    return fnp.max(fnp.array(values,dtype=fnp.float64))


def qualify(fnp, module):
    """Run once inside a fresh caller-metered <=FLOP_CAP context.

    module supplies the two authenticated pair functions; its capture hook
    is inert. Return contains only detached Python scalars/dicts, no arrays.
    The caller records complete F/opcount and owns cleanup/resource gates.
    """
    _require(callable(module._nonlinear_pair_batch) and callable(module._jca_explicit_pair),
             'both actual source-qualified pair entry points')
    mu,C,T,k4,Q22,Q31,s,a = _fixture(fnp)
    z = fnp.zeros_like(T); z4 = fnp.zeros_like(k4)
    modelQ31 = (k4/fnp.diag(C))[:,None]*C
    fnp.fill_diagonal(modelQ31,k4)
    cases = (('G',z,z4,z,z), ('T',T,z4,z,z), ('Q',z,k4,Q22,Q31),
             ('TQ',T,k4,Q22,Q31), ('physical_model',T,k4,Q22,modelQ31))
    reports, references, null_reports = {}, {}, {}
    for metric in ('identity','covariance'):
        for name,t,kk,qq,q31 in cases:
            reference = _direct(fnp,mu,C,t,kk,qq,q31,s,a,metric)
            actual = module._jca_explicit_pair(mu,C,t,kk,qq,metric,q31)
            reports[metric+'.'+name] = _check(fnp,actual,reference)
            references[metric+'.'+name] = reference
            if name == 'physical_model':
                original = module._nonlinear_pair_batch(mu,C,t,kk,qq,metric)
                rows = _check(fnp,original,reference)
                comparison = {label:_error(fnp,original[i],actual[i]) for i,label in
                              enumerate(('mean','covariance','post21','radial'))}
                _require(all(x['within_tolerance'] for x in comparison.values()),
                         'physical-model versus original algebraic equivalence')
                null_reports[metric] = dict(original_reference=rows,
                    original_versus_physical=comparison,
                    elementwise_equal=all(bool(fnp.array_equal(original[i],actual[i]))
                                          for i in range(4)),
                    byte_identity_required=False,byte_identity_claimed=False)
                del original
            del actual, reference
    correct = references['covariance.TQ']
    sentinels = {}
    mutations = [('Q31_transpose',dict(Q31=Q31.T)),
                 ('Q31_wrong_sigma_power',dict(Q31=Q31*(s[:,None]/s[None,:]))),
                 ('Gaussian_order5',dict(gauss=5)), ('Edgeworth_order1',dict(edge=1))]
    for i,name in enumerate(('T21_half','Q22_quarter','Q31_sixth',
                             'local_k3_sixth','local_k4_twentyfourth')):
        coefficients = tuple(0.0 if j == i else COEFFICIENTS[j] for j in range(5))
        mutations.append(('omit_'+name,dict(coefficients=coefficients)))
    for name,changes in mutations:
        supplied_q31 = changes.get('Q31',Q31)
        wrong = _direct(fnp,mu,C,T,k4,Q22,supplied_q31,s,a,'covariance',
                        gauss=changes.get('gauss',6),edge=changes.get('edge',2),
                        coefficients=changes.get('coefficients',COEFFICIENTS))
        separation = _separation(fnp,correct,wrong)
        _require(bool(fnp.greater(separation,2)), 'fixed fixture distinguishes '+name)
        sentinels[name] = float(separation)
        del wrong, separation
    # A simultaneous substitution is not the sum of separately recentered arms.
    g = references['covariance.G']; t = references['covariance.T']
    q = references['covariance.Q']
    superposed = tuple(fnp.subtract(fnp.add(t[i],q[i]),g[i]) for i in range(4))
    joint_separation = _separation(fnp,correct,superposed)
    _require(bool(fnp.greater(joint_separation,1)), 'nonzero simultaneous-centralization contrast')
    joint_separation_value = float(joint_separation)
    del joint_separation
    del references, correct, superposed, g, t, q, mutations, cases
    return dict(schema='current_comp_joint_cell_tiny_v1',passed=True,completed=True,
        fixture_units=4,fixture_random=False,truth_accessed=False,model_calls=0,
        explicit_pair_calls=10,original_pair_calls=2,direct_reference_calls=19,
        reference_dtype='float64',actual_pair_accumulator_dtype='float32',
        physical_Q31_sigma_cubed_normalization=True,ordered_transpose_checked=True,
        gaussian6_edgeworth2_factors_checked=True,simultaneous_centralization_checked=True,
        null_original_T31_operation_order_preserved=True,
        null_physical_FP32_byte_identity_claimed=False,
        atol=ATOL,rtol=RTOL,flop_cap=FLOP_CAP,complete_bill_owned_by_caller=True,
        comparisons=reports,null_physical=null_reports,incorrect_reference_separation=sentinels,
        simultaneous_centralization_separation=joint_separation_value,
        returns_only_detached_scalars=True,law_realizability_claimed=False,
        scientific_gain_measured=False)
