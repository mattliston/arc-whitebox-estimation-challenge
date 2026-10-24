# Covariance sandwich, middle setting (lead 2026-10-10 02:25Z, engineer 02:23Z): the second product w'(C w) of every layer, whose
# output is symmetric, as the file's own first form: the top 512 rows over all columns and the lower-right 512 x 512 block, the
# lower-left block mirrored (two tree products and four copies) in place of the later four-strip / diagonal-refined / 4-way block
# forms (about 398 requests a product). Fewer requests (the engineer's count: about -2,660 a network) for a larger share of a full
# product (about 0.76 against 0.70: about +0.27% of bill). The function is the one the later forms saved as their parent
# (_PS_PARENT_SYM_UPPER); _flms_cov_star reads _sym_upper by name at call time. Rounding-level: another block geometry.
_sym_upper = _PS_PARENT_SYM_UPPER
