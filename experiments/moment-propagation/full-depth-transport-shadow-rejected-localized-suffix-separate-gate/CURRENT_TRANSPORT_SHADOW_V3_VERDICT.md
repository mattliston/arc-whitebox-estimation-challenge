# Full-depth transport shadow rejected; localized suffix is a separate gate

The v2 baseline repair passes exactly on both ranks: clipped reported
means, all features, and all head corrections are byte-identical to the
frozen full-pair reference. V2's MPS check stopped at unsupported erfc;
v3 uses the equivalent erf CDF and otherwise identical equations/gates.

Target-free v3 final-response relative L2 errors versus the real carrier:

| Perturbed layer | Rank 35 | Rank 100 |
| --- | --- | --- |
| 4 | 0.335184 | 0.293119 |
| 9 | 0.081706 | 0.078488 |
| 13 | 0.009691 | 0.010204 |

Full finite-difference half-step disagreement is 7.43% / 6.15% at L4,
2.09% / 2.99% at L9, and .21% / .40% at L13. Shadow adjoint relative
errors are 0.0056% / 0.311%. No targets or fits were used. Receipt:
current_transport_shadow_fidelity_v3.json. Full-depth fidelity fails the
predeclared gate, and no full-depth fit will be launched on this shadow.

The failure is localized in elapsed propagation depth. A distinct follow-up
may freeze the entire prefix through layer 8 and train only layers 9..15.
Before any such fit, test one fixed simultaneous random correction across
layers 9..14 on both ranks, epsilon 2e-4 and half epsilon. Require <=10%
response error and <=5% full/shadow half-step discrepancy on each rank.
This checks the suffix as a coupled object, not just isolated endpoints.
It does not relax the failed full-depth gate or establish an accuracy gain.
