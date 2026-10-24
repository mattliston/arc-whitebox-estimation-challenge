# Centered Torch v1: small maps pass, CPUf32 full path fails

September6,2026,06:50-06:53PDT receipts. Same frozen P0/scalarQ bank,
no targets or training, no candidate or score improvement.

66 scalar cells and four pair maps passed forward and reverse derivative
gates on CPUf64/f32. Largest f32 scalar value error9.537e-7 (mean at large
positive alpha), variance5.164e-8, third4.204e-8, fourth4.395e-7.
Largest scalar normalized derivative error1.758e-6; at alpha0,2.744e-7.
Largest f32 pair normalized value1.991e-7, gradient relative2.709e-7.
F64 pair relative gradients<=4.216e-15; scalar gradient discrepancy against
the cancellation-prone original f64 formula reaches7.422e-11 at order4.

MPS direct erfc check with fallback DISABLED failed NotImplementedError
(aten::erfc.out, installed Torch2.7.0). No GPU full path ran. Preserve this
receipt; do not call CPU fallback an MPS result.

Full dynamic CPUf32 bank replay35/100: maximum mean differences7.53403e-6 /
7.22551e-6, final RMS1.20810e-6 /1.35049e-6. Fixed2e-6 maximum gate FAILS
both. All30 active/birth membership arrays match on both.5.066s,
3,539,206,144B peak. First nontrivial L1 already differs2.486e-6/2.524e-6.
Output SHA a6726c79c042a9934910fcbebdf7080859a5d83e152406c84a36d2a39551af8d.

Prereigistered CPUf64 control passes: max1.95748e-7/2.06323e-7, finalRMS
3.67615e-8/3.85762e-8, every membership equal.8.726s,5,252,759,552B peak.
Output SHA813b62c9662dcae4dad7b58ef4f9e19fc87c88b3ca4d1e7315a4c280de147fe8.
This supports a dtype/reduction issue, not a wholesale implementation error.

Next legitimate numerical work: diagnose mean matvec reduction on frozen
input-only trajectories, and independently supply an MPS-native normal
CDF from a fixed certified numerical identity. Old MPS early layers were
more accurate than the CPUf32 BLAS run, so CPU reduction behavior must not
be assumed identical to MPS. No tolerance relaxation or training follows.
