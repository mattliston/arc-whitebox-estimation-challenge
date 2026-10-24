# Source-only arithmetic census: closed learned pair-memory law

Phase 2 only; September 6, 2026. No model, array, meter, port, or numerical
experiment was executed for this census. Integer arithmetic below was evaluated
from source dimensions only. This is reusable accounting, NOT a candidate or
reason to reopen the law: root reports the completed100-epoch paired fit failed
all four held-out gates (covariance .9036964070, next-mean .9911990675,
history/RESET1.0002367769,1/4 wins).

## Verdict

The121 classical FP32 square products are259.846B, not the whole card. A
conservative census of the specified circuit plus existing bank maps/heads is
approximately278.775B, leaving11.225B below the prospective290B ceiling. The
large omitted charges are the FP64 full-pair maps (about10.085B including
radial fourth-plane setup), P/Q contractions (4.350B), and new source/gate/unit
work (about2.391B). Thus this particular design did not face an obvious
arithmetic-budget obstruction. Its measured information test nevertheless
failed.

278.775B is a SOURCE ENVELOPE with stated O(n) allowances and implementation
assumptions, not an exact or achieved metered bill. It does not close native
allocation/copy spelling, precision fidelity, final diagonal accuracy, or wall
time. Those gaps are listed below rather than assigned zero cost.

## Billing convention and source ownership

Let n=1024 and N=n^2=1,048,576. Use2mnk as a conservative classical GEMM
count, no Strassen/Winograd, symmetry, sparse, final-diagonal, or zero-value
discounts. The installed official weight table prices float32 at1 and float64
at2; exp/log/log1p/tanh/power at16; sqrt/divide/basic arithmetic at1; clip at
one comparison per bound; three-argument where at4. Transpose, basic slicing,
zeros and empty are free in this installed accounting implementation, not in
wall time. Ones, real copies, casts, reshape and assembly are chargeable.
Advanced gathers cost4 per output element before precision multiplication;
put/fill_diagonal charge writes. A cast bills at the heavier source/destination
rate. These are locally inspected rules, not an assertion about future server
versions or external overrides.

Pinned sources:

- learned_response_pair_memory_numpy_v1.py:
  5a24a77c24f08ba521fa8d0cd36a569a9eeee6bf3045cb4ee15d0117fbf4129e
- candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py:
  3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757
- Installed flopscope/data/default_weights.json:
  9ff1647a0048d2bd23a7a3d76ee0c60bfd3670d03b15ad8bf2b911c2ae19539f

The bank functions inspected were gaussian_raw, derivatives,
_nonlinear_pair_batch, _nonlinear_first_gaussian, nonlinear_scalar, features,
gnn_step, final_scalar_features, final_input, final_delta, _cov_compact,
select and the active _predict_binary. Their smaller NumPy source counterparts
and the original GNN/final-head architecture definitions were also inspected.
No payload or truth arrays were decoded.

## Whole-circuit ledger

All entries are billed classical FLOPs. Small-vector entries are deliberately
rounded upper allowances, not purported instruction-exact totals.

| Component | Formula / scope | Billions |
|---|---|---:|
| Covariance, star, H congruence, KU |121*2*n^3;15 transitions plus initial Gram|259.845521408|
| Existing full-pair moment maps |2*N*(14*330+99)|9.896460288|
| Radial K22 plane construction |15*12*N|0.188743680|
| P0 and actual Q contractions |14 prefix calls and1 final call|4.350017536|
| P0/final feature matrix operations |(15*58+10)*N|0.922746880|
| P/Q repeated W casts/powers |15*5*N|0.078643200|
| P/Q scalar-channel operations |15*16384*n allowance|0.251658240|
| Gaussian/moment/feature/select vectors |16*8192*n allowance|0.134217728|
| New law, source and units |15*152*N envelope; detailed below|2.390753280|
| Compact covariance gathers/casts |15*20*N envelope|0.314572800|
| Mean feedback and physical cold masks |15*17*N|0.267386880|
| Weight ingress/materializations |64*N, conservative two wide copies|0.067108864|
|16 FP64 affine means |16*4*N|0.067108864|
| Mean/host copies and final stack/clamp/cast |158*n|0.000161792|
| Literal head-array materialization |200,000 allowance|0.000200000|
| **Source-envelope total** | |**278.775301440**|

This envelope corresponds to utilization .1267723298 under2^41. That is a
hypothetical arithmetic conversion, NOT a score. The remaining ceiling margin
is11,224,698,560 FLOPs.

### Why the full-pair maps cost about10B

For the native three-consumer batch (A/A, B/A, B/B), the general map has these
N-sized unit-cost operations, before the float64 rate:

- Gaussian r=1..6 terms:54N.
- r=0..2 repeated-third/fourth edge terms and ans accumulation:171N.
- r=1,2 local third/fourth terms:60N.
- Shared rho-power recurrence including the final unused update:14N.
- rho/T21/T22/T31 setup:7N; initial ones:1N.
- Output dimensional scaling:6N; post21 centering:2N.
- post22 centering:11N; both radial sums and covariance-square statistic:4N.

Total330N. The Gaussian-first source retains the setup but skips all edge
terms, giving99N. The final layer uses nonlinear_scalar, not a sixteenth
full-pair map. The radial K22 setup outside each hidden map is another6N
at float64. Gaussian CDF, exp, marginal powers, Hermite vectors, diagonal
copies/writes, feature stacks and sorting are in the separate O(n) allowance.
The cubic and local fourth maps are already in330N; they are not an additional
unpriced tensor operation.

### P/Q contractions and features

Use the existing30 input features,64-channel state,16 channels for each of
three edge powers, and128-wide encoders. A prefix call costs

    2*n*64*48 + 6*2*N*16
      + 2*n*(126*128 + 128*128 + 128*64 + 64)
    =291,110,912.

The actual final Q uses the same source/message construction followed by
126->128->128->1, costing274,464,768. Prefix layers1..14 run; layer0 skips
the GNN but still computes its feature call in the inherited wrapper.

Each hidden feature call has at most24N float64 operations, including the
rho-alpha matrix-vector product, and10N float32 weight/power/reduction
operations:58N billed. The final scalar-feature weight terms add10N. A
separate5N per active head pays its repeated W cast, W^2, W^3, W^4 and W^6
preparations. The16384*n per-call allowance covers the48-channel powers
(power charged16), exp/SiLU, energy log1p, clipping, concatenations, biases,
feature normalization and scalar output operations. It is over the direct
source tally; no neural message multiply is hidden in that allowance.

### New-law152N per-transition envelope

Keep physical moment arithmetic/units in float64 and the recurrent H/read
arithmetic in float32. The conservative components are:

- 34N: construct post standardized D in float64, retain its float32 copy,
  form the weighted/symmetric/offdiagonal B with its real copy, stable psi(B)
  in float64, and cast the source to float32. Vector skew/unit operations are
  in the O(n) allowance.
- 81N: post-scale the reused Upre, symmetrize and apply psi (13N); then the
  learned gate/update (68N).
- 17N: query sym/psi (11N), masks, fixed1/sqrt(n), two beta multiplications
  and sum (6N). KU's GEMM is already in121.
- 6N: float64 formation of V followed by its float32 materialization.
- 5N: star edge copy, V^2, V*Q, factor2 and sum; its three products are
  already in121.
- 6N: promote the predicted standardized plane and restore physical units.
- 3N: current live mask, base-plane masking, and correction addition.

The stable psi costs9 ordinary pointwise operations. The NumPy stable sigmoid
costs27 including exp weight16, both branches, comparison and where weight4.
The gate source features cost22N, logit15N, sigmoid27N and convex mix4N,
giving68N. No RMS normalizer, PSD Gram, eigendecomposition or Cholesky is
assumed. Layer0's zero-U shortcut and final-only diagonal optimizations are
not credited.

### Cold covariance, copying and masks

With a active and c=n-a cold rows, the existing compact covariance products
cost2*a^2*n+2*a*n^2+2*c*n^2 =2*a^2*n+2*n^3 <=4*n^3.
Thus the three actual hot/hot/cold calls fit the two-full-product envelope;
the cold diagonal does NOT force a ninth full square product. Its gathers,
FP64->FP32 covariance cast, active/full FP64 result casts, cold weighting and
addition cost at most20N plus vector work. Full-width star transport already
contains the cold D diagonal and needs no separate cold cubic product.

The three mean-feedback outer products/subtractions are12N at float64;
physical C/target cold masks and live-mask formation add5N. Diagonals remain
retained. Copies from pointwise outputs are not charged again merely because
the operation allocates; explicit copies/assemblies must be billed separately.

## Unclosed deployment qualifications

1. There is no native CP-free implementation. Row-blocked gate output
   assembly, any extra cast/reshape/gather/put, constant caching and input
   wrapper materialization must be counted from that eventual source. They
   are not certified by the11.225B unused budget margin. Neither backend
   allocation overhead nor native wall/residual-wall time follows from FLOPs.
2. The pilot supplies teacher post D and precomputed K. Deployment must form
   its OWN post D from the moment map, using the exact current mean-feedback,
   variance-unit and cold-mask convention, and pay all three star products.
   The121 tally does pay the latter; the pilot does not validate endogenous
   state fidelity or FP32 star-action precision.
3. The fixed read correction excludes current-query diagonals. Its teacher
   covariance/next-mean proxy does not establish final diagonal skew or final
   raw accuracy. No affordability conclusion fills that information gap.
4. The retained radial fourth scalar/planes are fully priced. The old explicit
   third trace and CP cohort birth/restoration arena are NOT secretly retained
   in addition to total D. A separate trace path would require explicit
   subtraction/recombination ownership to avoid double counting, and a new
   arithmetic census. No such variant is proposed or authorized here.
5. No PSD repair or full covariance corrective Gram is priced. Adding one
   classical full square product per transition would add32.212B, already
   exceeding the remaining margin. Broader covariance repair policies are
   not an uncharged implementation detail.
6. Vector allowances are auditable conservative budgets, not a literal native
   op trace. Native billing/version parity and persistent wall-time/resource
   qualification remain mandatory before any future release claim.

Conclusion: no new experiment or port is justified by this cost accounting.
The accepted law has a qualified negative information result and remains
closed. The useful surviving artifact is the explicit map/head/copy census.
