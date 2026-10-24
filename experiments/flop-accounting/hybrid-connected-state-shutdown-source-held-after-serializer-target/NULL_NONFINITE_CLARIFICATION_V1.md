# Null-branch scope clarification

Status: `SEALED_BEFORE_SOURCE__NO_OPERATIONAL_CHANGE`.

This clarification binds null-division correction SHA-256
`4f95cdc324e31459e7e700c54c68f47d8164eca261286a93521f0df9880de253`.
Its `denominator/1.0` arm applies only when native truth conversion of the
represented denominator is false, which for the finite sum-of-squares domain
is its exact positive-zero face.

A NaN or other truthy nonfinite denominator follows the ordinary frozen parent
division `numerator/denominator`.  Candidate source must preserve that ordinary
result and downstream behavior exactly.  No claim is made here that every such
value necessarily reaches a particular fatal guard, and CS3B must not add a
nonfinite rescue, replacement, threshold, or exception.  This clarifies only
the earlier phrase "remains fatal downstream"; the exact-zero spelling,
FLOP/call laws, and all source gates are unchanged.

