# Residual q4 bridge fails the actual value gate

September 5, 2026, 20:43 PDT. Phase 2 only. No candidate or upload.

The fixed full-plane bridge at rank35/post12 -> pair13 -> mean14 gives
the following point error ratios against the original coherent8M post13
covariance innovation:

| Natural source | Covariance | Next-mean response |
| --- | --- | --- |
| Independent q4 residual oracle A | .471327 | .765031 |
| Independent q4 residual oracle B | .472326 | .777423 |
| Exact cumulative heavy source | .566766 | .886166 |
| Positive-path cumulative source | .526647 | .871124 |
| Full incoming K3 reference | .184595 | .299712 |

The preregistered <=.75 next-mean ratio for both oracle sources AND the
exact physical source fails. No residual-matrix carrier implementation,
rank/basis/source sweep or fit follows from this bridge. The former K21
weighted-norm source success did not establish sufficient value transfer.

V1 stopped before results because a full physical tensor was compared to
an active-only bank plane. V2 applied the actual active support and stopped
on a1.0409154e-6 F64-physical-versus-F32-query difference. V3 did not loosen
that check: it replayed both saved objects independently, certifying the
physical tensor slices/plane and actual consumer state byte-exactly.
The physical/query precision difference is reported, not an accuracy gain.
The full projected residual is masked to current active support, retaining
all diagonal reads. No source coefficient or gain gate was changed.

The post13 covariance label is original8M, not a new independent pair;
the residual A/B slices are independent8M. These are qualified diagnostic
point ratios, not a final-score ceiling. Receipt:
one_high_actual_value_r35_l12_v3.json. Its reported wall excludes the
separate byte-exact replay; it is not a whole-diagnostic runtime claim.
