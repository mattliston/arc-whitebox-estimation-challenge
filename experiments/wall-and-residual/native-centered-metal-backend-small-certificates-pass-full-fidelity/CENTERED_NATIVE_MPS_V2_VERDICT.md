# Native centered MPS v2: small certificates pass, full fidelity fails

September6,2026,06:57-06:59PDT. Gaussian CDF quadrature is a fixed numerical
identity with32 Gauss-Legendre nodes, no fitted parameters.4017-point
CPUf64 value/gradient errors<=7.217e-16. MPSf32 CDF max6.330e-8,
derivative6.368e-8, Phi(0) exactly.5. All66 scalar/four pair-map value and
reverse derivative tests pass unchanged gates on CPUf64/f32 and MPSf32.
No CPU fallback; native operations only.

Full MPSf32 current bank35/100 still FAILS unchanged2e-6 gate:
max mean errors2.28970e-5 /2.85331e-5, finalRMS4.54110e-6 /6.35443e-6.
Every30 active/birth membership array matches. Time1.988s for both,
process RSS885,194,752B (NOT total unified GPU memory).
Output SHA933f988bbc3a2f8d1424688a2f4003ba3775b117dddceb464e02cb02e6b3bbff.
Rank35 first large error emerges L5/L6; early L1-L4 much smaller than the
CPUf32 BLAS version. This resembles the old MPS failure, so scalar/raw
central-moment cancellation is not established as its main cause.

No GPU training or release follows. A bounded operation-level numerical
attribution is legitimate: CPUf64 reference versus both dynamic MPSf32
and MPSf32 with each incoming state reset to the rounded CPU reference,
on both fixed target-free development networks. Preserve all old receipts.
