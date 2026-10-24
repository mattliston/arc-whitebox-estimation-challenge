# Algebra for a possible single-direction implementation

This is source/design work prepared while fixed TRAIN acquisition runs.
It selects no native kernel or candidate before the cross-network gate.
The feature acquisition and fit sources/seal are unchanged. No runtime,
full bill, fidelity, accuracy improvement or deployable result is claimed.

For the fixed projection/ridge fit, write normalized affine features X=B/sx,
normalized tangent features Z=J/sz, projection matrix P, retained columns K,
residual scales sr and ridge coefficients beta/gamma. The held correction is

    X beta + ((Z-X P)[:,K]/sr[K]) gamma.

Define g[K]=gamma/sr[K], zero elsewhere. Then the SAME real-arithmetic
predictor is B beta_effective + J gamma_effective, where

    beta_effective = (beta - P g)/sx,
    gamma_effective = g/sz.

This removes all normalizer/projection matrices from deployment. It is not
a new fit, feature or data-dependent pruning; the65+65 effective coefficients
are determined by the selected fit. Floating arithmetic equivalence still
needs verification against the sealed predictor. Add the affine correction
explicitly after the unchanged parent rather than silently changing its
FP32 head-literal rounding.

For q=gamma_effective, define s=.005*(H14 q[:64]+q[64]). The required
directional covariance response is

    delta_Ca = Ca*s_active[:,None] + Ca*s_active[None,:]
    delta_active_v = sum(wa * (delta_Ca @ wa), axis=0)
    delta_v = delta_active_v
              + sum(wcold**2 * (2*Cdiag_cold*s_cold)[:,None], axis=0).

This is algebraically the old complete65-column Jacobian multiplied by q,
including nonsymmetric C and signed cold diagonals. A single FP64 matrix
product suffices; there is no assumption that C is symmetric. At maximum
1024 active rows, its conventional leading tariff is at most about4.30GF
before elementwise/copy/other work, consistent with investigating the fixed
10GF allowance but not proving a complete cost bound. No full65-direction
matrix is needed in a production predictor.

Propagate only this one delta_v/delta_active_v direction through the same
scalar formulas and native-anchored final feature/GNN derivative. H14,
original messages, parameters and all native preactivation anchors are
fixed. Preserve feature clipping masks and the parent final positive branch.
There is NO finite DCD write, exp/tanh intervention, further-layer state
change or claim that the predictor is a physical distribution update.

If and only if the fixed CV gate passes, a separately sealed implementation
must establish actual FNP accounting, coefficient-collapse and directional
parity, unchanged parent earlier outputs, context/cache retirement, full
native CHECK/NC14/confirmation and client runtime. The earlier qualified
probe directions do not certify the eventual fitted direction automatically.
