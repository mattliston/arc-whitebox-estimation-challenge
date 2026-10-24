# Source-audit clarification; original preserved

September8,2026 PDT. Independent source review of
JOINT_CORE_AND_BINI_ERROR_SOURCE_AUDIT_2026_09_08_V1.md, SHA
5f843197cbac083c2784567dff6963d2b906e5ad1c1087940498c0dcb10fa6ee,
found one weight-placement ambiguity and one attribution ambiguity.
No numerical work, source-family reopening or candidate change follows.

Section3 writes the exact paired plane with weights on the LEFT. The
section4 paired APA defect is specifically for the mathematically equal
RIGHT-weighted realization

    U*(lambda*D).T + V*(lambda*H).T,
    U=H^2+D^2/4, V=2*H*D.

For that realization the printed inner-partition defect
U2*(lambda1*D1).T+V2*(lambda1*H1).T is correct. For left-weighted operands
the defect instead is (lambda2*U2)*D1.T+(lambda2*V2)*H1.T. Exact products
are equal, but their approximate cross-block errors need not be. Both
defects are generically nonzero. No weight-placement invariance of the
approximate circuit is claimed. All defect matrices, explicit correction
costs, and no-prototype conclusion remain unchanged.

The linked arbenson/fast-matmul coefficient repository is Benson/Ballard's;
it is not Boyer-Dumas's. The ten-product circuit actually expanded in this
audit comes from the cited Boyer-Dumas poster. Literal equivalence to the
separately listed repository coefficients remains unverified.

Accuracy-agent review passed sections1--2 and status at the original SHA.
Efficiency-agent review passed the remaining algebra/costs subject to the
two clarifications above. Preserve the original note and this addendum.
