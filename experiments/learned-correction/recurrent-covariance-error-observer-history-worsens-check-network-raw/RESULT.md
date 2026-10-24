# Recurrent covariance-error observer: paired fit failed

No new banked gain. The fixed HISTORY / RESET comparison completed all
30 epochs and 180 whole-network updates per arm. Its prospective advancement
gates fail; the failed payloads do not proceed to a native candidate.

| Arm | Excluded MINI | Final raw MSE ratio | Mean covariance residual ratio |
|---|---:|---:|---:|
| HISTORY | 14 | 1.024427029574 | 0.972889379473 |
| RESET | 14 | 1.018408533099 | 0.977010152183 |
| HISTORY | 71 | 1.005045955899 | 0.958302587715 |
| RESET | 71 | 0.996822139856 | 0.963140539046 |

HISTORY CHECK pooled raw ratio is 1.015261425333;
HISTORY / RESET pooled final MSE is 1.007003992870.
The fixed gates are: pooled raw <= .90, each raw <= .95, history/reset <= .975,
history beats reset on both networks, and mean covariance residual <= .90.
Their recorded decisions are {"pooled_raw": false, "each_raw": false, "incremental": false, "both_vs_reset": false, "covariance": false}.
The .90 raw gate reserved 6% additional deployment compute; this reserve
was an assumption, not a metered bill or an adjusted-score observation.

Both models have 6,209 allocated parameters; RESET's 32 memory-edge weights
are inactive. The shared pair source uses current-S31 raw30 + H64 fields,
normalized post covariance and incoming K21. HISTORY also receives its own
transported covariance-error matrix and carries it through Gaussian gates.
RESET retains the same mean-error observer and discards only covariance
memory. Full gradients pass through the generated states. Neither arm gets
true covariance or true means as runtime inputs. The parent S31 trajectory
stays fixed; this is an approximate error observer appended at the output.

TRAIN uses six MINI networks and CHECK uses two. These public networks have
prior research use and are not pristine holdouts. Each of the seven post
covariance targets uses existing 100M moments. Final means use the documented
copied public 1B gt_mean field. Both terminal models and all eight forecasts
per arm were frozen before this study's CHECK labels were opened. No epoch,
coefficient, amplitude, layer or label selection follows. Reference noise
and shared numerical bias were not removed.

Eight new native captures reproduce all 16 S31 output arrays and the complete
process-order-matched bills exactly. The 56 covariance target cells pass
identity, sample-count, raw-diagonal and affine mean/covariance checks.
Tiny independent recurrence replay and full directional-gradient checks pass;
HISTORY has a nonzero memory-input gradient and RESET has exactly zero.
Independent full-width NumPy recurrence replay and scalar score aggregation
pass: maximum score difference 2.22e-16, maximum
state relative difference 1.96e-07,
maximum forecast difference 9.89e-12.
Strict upper-triangle FP64 storage reconstructs the symmetric zero-diagonal
targets and predictions without numerical compression.

This closes the specified state, source, observer and paired fit. It does not
establish a general bound on recurrent covariance models. No fitted native
port, full deployment bill, VALID40, client qualification or upload follows.
All four guards complete and retire, peak 5.312299 GB, zero added swap.
No new samples or downloads. S31 #332517 remains 2.1736465056908115e-9
adjusted in its saved official receipt; the 2e-9/~1e-9 goal remains active.
