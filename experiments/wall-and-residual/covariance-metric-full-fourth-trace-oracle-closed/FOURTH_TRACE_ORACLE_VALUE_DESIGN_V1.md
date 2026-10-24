# Full fourth-trace representational value, independent-replica oracle

2026-09-06, Phase 2 only. This is not a deployment experiment or a restart
of the cost-rejected full carrier. It asks a missing information question:
does the COMPLETE covariance-metric trace core carry a useful part of the
fourth-plane error seen by the next pair/mean consumer? Existing records
have repeated fourth planes and four directional source slices, not the
full matrix R_ij=sum_a kappa4(Y_i,Y_j,Y_a,Y_a) at post12.

## Fixed scope and input freeze

Rank35 only, post12 -> pre13 -> next mean response14. Freeze the actual
on-policy bank post12 covariance after mean feedback and masking, radial12,
and actual pre13 mu/C/K21/k4/K22/unlearned host/sigma/alpha/pdf/gate. Require
prefix features and sigmas to match the current tape byte-for-byte and the
independent-fourth map's zero-change baseline to match actual m/C to1e-12.
Use the bank's stored pre14 scalar response (float32 fields, explicitly
qualified) and actual W14. Freeze this before accessing new statistics.

## New statistics, same two independent sample streams

Use the exact prior rank35 A/B streams: 8,000,000 standard-normal float32
inputs each, PCG64DXSM with seed XOR0x4F4E454849474831/32, batch65536.
The streams are independent of each other but reused research streams,
not new validation nets. Compute new post12 full fourth trace R, and
pre13 full K22/K31/marginal k4. Do not change seeds, counts, layers, or
sample sizes based on results. One nice-10 MPS job at a time.

Center each batch before powers. For empirical centered covariance S and
fourth moment M, unbiased multivariate batch k-statistics are

    n^2/((n-1)(n-2)(n-3))*[(n+1)*M-(n-1)*Wick(S)].

For R, M=mean[h_i*h_j*||h||^2], Wick=trace(S)*S+2*S@S.
Weight each batch by its sample count. This differs from one full-sample
k-statistic, but remains unbiased for IID inputs. MPS raw moments are
float32; Wick products and accumulation are CPU float64. Repeated-plane
diagonals must agree with separately reduced marginal k4 within1% relative.
The R diagonal must agree with an independent norm-weighted vector
reduction within1%. Reproduce prior post12 and pre13 shifted mean sums
within1e-7 absolute (report whether exactly equal). Shared floating-point
bias is not removed by independent replicas.

Before acquisition, certify the trace and plane formulas against explicit
dense fourth tensors at small non-Gaussian synthetic batches, <=1e-10.

## Fixed oracle reconstruction and value gate

For each stream, subtract the bank radial trace from true R12, then use
the certified full-rank inverse of Q=Sym(A,C12_bank) to obtain A. Push both
A and its fixed metric by actual W13. Its K4/K22/K31 increments are added
to the ACTUAL pre13 baseline fourth inputs. Nothing else changes. Apply
the already certified exact finite pair response, including the quadratic
diagonal mean-centering remainder, with current K3 held fixed.

The target is the same response from FULL true pre13 fourth planes minus
the same bank baseline. This isolates representation of the fourth-input
contribution; it is NOT the entire covariance error, true final MSE, or
the effect of a co-adapted complete carrier. Compare covariance response
and diag(W14.T@response@W14)*fixed Gaussian variance-to-mean derivative.

For each consumer, require positive replica cross-signal energy, residual
cross-energy / baseline cross-energy <=0.50, and BOTH cross-replica ordinary
error ratios (predictionA vs truthB, predictionB vs truthA) <=0.65. No
coefficient, source, eigenspace, rank, layer, seed, or count selection.
Report noise and absolute energies. Failure closes this full-trace oracle
route at the fixed cell; insufficient signal stops it as unresolved, not
as proof of no value. Neither outcome authorizes acquisition expansion.

A pass only establishes an information opportunity. It still requires a
separate affordable, accurate source/third-carry design, matched heads,
14-net adjusted panel, and persistent official runtime qualification.
No forum posts, uploads, frozen-package changes, or commits.
