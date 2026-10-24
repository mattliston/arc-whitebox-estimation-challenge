# Additive float32 restoration clarification

September6,2026 PDT, before the approved Stage A capture. Root explicitly
approved this interpretation. Preserve the original proposal and algebra.

The rank-2N error-plane identity is exact for real arithmetic with the
ideal merged vector Z=kappa*D+eta*R. The deployed vector is stored float32,
and the plane's factor assembly and GEMM are also rounded. Therefore the
implemented rank-2N restoration is NOT byte-exact current-plane repair.

Keep the priced rank-2N restoration. Stage B, if separately authorized,
must use the actual deployable float32 factor assembly and its GEMM.
Compute actual error from the stored original p/q and stored float32 Z;
report the resulting nonzero CURRENT-plane discrepancy as well as the
NEXT-plane/scalar/consumer errors. Include both Z quantization and
restoration assembly/product roundoff. Never use a free rank-3N exact
repair in diagnostics while pricing a rank-2N implementation.

Stage A approval is unchanged: ideal tensor-error criterion<=.1, finite
actual float32 Z, first-transport-only eligibility, actual padded rank-2N
cost gate>=25% BOTH35/100, construction charged for all tested pairs.
Stage A includes no additional plane/consumer GEMMs beyond the unmodified
baseline capture and stops for root review before Stage B.
