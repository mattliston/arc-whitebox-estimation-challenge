# Additive preregistration correction: exact-null division spelling

Status:
`SEALED_BEFORE_SOURCE__SUPERSEDES_ONLY_NULL_CONSTRUCTOR_AND_CALL_DELTA_WORDING`.

This correction binds immutable CS3B preregistration SHA-256
`dde30939ef271523a7bda593903d9377da8372ffd56d0fa1684c479d9a9c9e5f`
and independent dense-CS3 HOLD SHA-256
`060cfbab98c85f7f9d46d49edd29726a985ae50fbc2c85b47660a35704a178db`.
It was sealed before any CS3B candidate source or execution.

The preregistration correctly freezes alpha `+0.0` as the unique minimum-norm
coefficient when the represented off-diagonal K22 fitting-plane denominator is
exact zero.  Its suggested `fnp.asarray(1.0,dtype=fnp.float64)` constructor and
the associated `+n_null` call term are not source-authoritative: materializing
a scalar through `fnp.asarray` is a billed operation in the pinned runtime.

The only authorized exact-null spelling is instead

```text
if bool(denominator):
    pairing_alpha = numerator / denominator
else:
    pairing_alpha = denominator / 1.0
```

The denominator is the fp64 sum of represented squares.  On its exact finite
null face it is represented positive zero, so division by the Python scalar
`1.0` returns represented positive zero.  The operation is the same one fp64
tracked scalar division as the ordinary branch; it introduces no constructor,
epsilon, comparison operation, or additional tracked call.  A nonfinite
denominator takes the ordinary branch and remains fatal downstream.

Accordingly, the corrected call law is exactly

```text
Delta_calls = 18*a_l + 10*a_d - 14,
```

with no `n_null` term.  The FLOP laws, selector, algebra, null-face rationale,
and all other sealed preregistration gates are unchanged.  Any future source
must meter the ordinary and null branches and require equal division operation
name, dtype, shape, call count, and FLOP charge.

