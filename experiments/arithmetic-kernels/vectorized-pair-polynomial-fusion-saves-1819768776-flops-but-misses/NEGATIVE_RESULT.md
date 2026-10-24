# Polynomial fusion on Lazy: panel consistency gate failed

The fixed vectorized candidate passes algebra, drift, repeat, cost and
ordinary1B development checks. It does not qualify for a board upload:
its fixed14-network panel wins11/14, below the predeclared12/14 requirement.
All other panel conditions pass. No confirmation, real-client run, package,
archive or upload follows. Prepared downstream scripts were not executed.
The older September13 RPC component gate remains failed and unchanged.

Candidate SHA256: 29e9211b4136bde7a74c9f554e697bd0e7e0c77e69326e5c365bafedfa9d8b07.
Only the series helper and unused T31 construction/cast in two callers differ
from submitted Lazy. The whole-source inverse is exact; coefficients, learned
parameters, compression, Lazy schedule and kernels are unchanged. Polynomial
identity holds for symmetric covariance in real arithmetic. Actual covariance
asymmetry and FP32 association are retained/reordered respectively, with no
whole-network byte-equivalence claim.

The full-width helper saves125789180F; removing T31 construction/cast saves
4194304F per pair. Fourteen pair calls yield exactly1819768776F saved per
full prediction, reconciled on every development and panel call. Full native
35/100/35 plus independent cold100 repeat exactly; first layer matches Lazy.
Development raw ratios are1.0021265463/1.0033758399; own-cold adjusted ratios
.9975454711/.9988212287, pooled.9981807591. The development gate passes.

The fixed existing4M panel was scored after all outputs and bills were sealed.
Raw/noise-corrected references were recomputed for both arms:

| Group | Candidate adjusted NC | Lazy adjusted NC | Ratio | Wins |
| --- | ---: | ---: | ---: | ---: |
| Public8 | 4.2006435396e-9 | 4.2169820297e-9 | .9961255491 | 6/8 |
| Fresh6 | 3.2722334405e-9 | 3.2819169879e-9 | .9970494234 | 5/6 |
| All14 | 3.8027534972e-9 | 3.8162398689e-9 | .9964660576 | 11/14 |

Three ratios exceed1: ranks35(1.0016043017),100(1.0017584351),
3016503(1.0012674893). Ordinary1B development and4M panel reference estimates
are distinct; their different rank35/100 outcomes are preserved. These reused
panel networks are not an independent generalization or official score.
The3.8029621540e-9 official-parent projection is only arithmetic, not a result.

Sixteen full predictions total6317049140725F. All17 guarded stages completed
and retired,500.0443seconds in summed guard elapsed time, peak6518603776B
including worker self-report, zero added swap. Terminal sessions45656,49823,
52010,56230 are reaped. No numerical or upload worker remains. About3.8MB of
new versioned artifacts; no new Monte Carlo dataset or fit.

Official best stays Lazy #331599: adjusted3.8164492657795035e-9,
raw2.169264529783277e-8. The4e-9 bank is achieved;3e-9,2e-9 and approximately
1e-9 remain unmet. Goal active. Future qualified Phase2 submissions remain
preauthorized; no permission question is pending. This candidate is closed,
with no changed association, coefficient, precision, roster or gate rescue.
