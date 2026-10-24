# Fixed layer7 error transport: gate failed

The privileged Gaussian-gain observer has a partial signal, but fails both
predeclared advancement gates. It supplies no deployable estimator or new
board candidate. The submitted Lazy archive remains unchanged.

| Measurement | Full / parent error | Full / mean-only error | Required |
| --- | ---: | ---: | ---: |
| Rank35 cross-replica point estimate | 0.8179387637 | 0.8305631563 | Both <=0.75 |
| Rank100 ordinary 1B mean reference | 1.1694576703 | 0.9877257047 | Both <=0.75 |

Rank35 cross-replica error falls by about18.2%, so this is not a zero-signal
result. However, its reduction is smaller than required even with measured
mean/covariance errors supplied at post7 and zero inference cost assigned to
the forecast. Against the ordinary1B references, the actual noisy forecasts
have raw error ratios1.1300656207 (rank35, averaged across both streams) and
1.1694576703 (rank100). These are privileged diagnostic outputs, not complete
candidate scores. Sampling error in the8M initializer contributes to their
observed error; do not reinterpret those ratios as exact-population bounds.

There is a useful distinction between one-step and autonomous transport.
On rank35, resetting the observer to measured true errors at every preceding
layer gives covariance residual cross-ratios0.32463 at layer8 and0.19641 at
layer14; the corresponding mean ratios are0.19794 and0.11361. In contrast,
the free forecast initialized only at7 has covariance residual0.84305 and
mean residual0.80762 at14. Its final mean residual is0.81794. Thus the fixed
response can describe much of a single transition while the forecast loses
accuracy without later source information. This does not prove innovations
are independent, unpredictable, or easy to learn. No supplier for them has
been established, and no teacher-forced result is called an autonomous score.

The rank35 final parent-error cross estimate is1.8212163540e-8; the observed
8M error is2.7157768645e-8 and the two-stream noise term8.9456051050e-9. The
cross full-forecast residual is1.4896434530e-8, versus1.7935342323e-8 for the
mean-only initializer. These are point estimates, not confidence bounds.
The same MPS numerical implementation is shared by the independent sample
streams; common numerical bias and empirical-centering bias remain unbounded.

All qualifications passed. Canonical weights agree byte-for-byte with the
older covariance source. Three source post7->pre8 checks have maximum mean
relative error7.677e-9 and covariance relative error2.698e-5, below the fixed
1e-4 limit. The small tangent central-difference maximum is4.340e-10; its
linearity maximum8.370e-17. This certifies the stated Gaussian-gain map only,
not the full cumulant, selector, masking or learned-head derivative.

Passive native35/100/35 capture reproduces ALL16 returned rows and the three
complete original bills398080917697/388883126034/377556934779F exactly. Every
captured rank35 array repeats byte-for-byte; W5 numeric caches retire. The
three archives total379315752 bytes. Captures were sealed before targets;
both networks' forecasts were saved before either ordinary1B reference was
decoded. Zero fits, new samples, primary state writes, archives or uploads.

One initial unit startup lacked SciPy and exited before qualification or
target access. The preserved dependency-only v2 uses math.erf on identical
fixtures and thresholds; no package was installed. The qualified unit,
capture and analysis guards all completed. All four groups, including that
startup failure, retired with zero added swap; see resources.json for totals.

The fixed layer7/response/control screen is closed. No cutoff, strength,
rank, feature, precision or fitting rescue is selected. The older linear
matrix observer and oracle-input nonlinear covariance studies retain their
original negative results; this finding is not a universal matrix-memory
or covariance-correction impossibility claim.

Official Lazy #331599 remains adjusted3.8164492657795035e-9 and
raw2.169264529783277e-8. The4e-9 bank is achieved;3/2/~1e-9 remain unmet.
The goal stays active; qualified Phase2 submissions remain preauthorized.
