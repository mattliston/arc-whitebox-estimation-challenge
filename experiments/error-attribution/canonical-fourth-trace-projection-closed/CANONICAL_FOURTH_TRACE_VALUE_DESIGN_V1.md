# Canonical harmonic trace projection: one existing-data value audit

2026-09-06, Phase 2. No new sampling, target set, source fit, or full carrier.

The preceding oracle solved trace(Sym(A,C_bank))=R_residual. This is not an
orthogonal projection: the adjoint of the Euclidean partial-trace operator
is A -> Sym(A,I), not A -> Sym(A,C_bank). The original cost-rejected full
augmented proposal explicitly used a radial part plus Sym(A,I). Its
canonical information value must not be inferred from a test of the
different covariance-metric trace-matching reconstruction.

For a symmetric fourth tensor T and symmetric A,

    <T,Sym(A,I)>_F = <trace_pair(T),A>_F.

Consequently P=Sym(A,I), with

    A = 6*R/(n+4) - 3*trace(R)*I/((n+2)*(n+4)),

is the UNIQUE orthogonal projection onto the trace-generated tensor
subspace when R=trace_pair(T). Its residual is traceless and orthogonal
to every Sym(B,I). This is a mathematical reason for this SINGLE
canonical alternative, not a search over metrics, scales, or eigenspaces.
Frobenius optimality does not guarantee optimal downstream mean error.

First certify the trace identity, adjoint identity, and residual
orthogonality against explicit small symmetric fourth tensors, <=1e-10.
Then reuse exactly the frozen rank35 post12/pre13 bank inputs and the two
new8M A/B statistic files. Subtract the SAME bank radial fourth trace,
use the canonical A above, and push metric I by actual W13. All baseline
fourth inputs, K3, pair response, W14 sensitivity, statistical definitions,
and gates remain identical to FOURTH_TRACE_ORACLE_VALUE_DESIGN_V1.
The target response arrays must be byte-identical to the preceding audit.

Require positive cross-signal, residual cross-energy ratio<=0.50, both
ordinary cross-replica error ratios<=0.65, for covariance and next-mean
response. No coefficient, metric, rank, age, layer, sample-count, or seed
search. A failure closes this canonical projection value test. A pass
still authorizes no expensive full carrier: affordable source and complete
third-carry feasibility must be separately established. No forum posting,
bank, source-package edit, or commit.
