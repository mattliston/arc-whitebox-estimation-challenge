GN r2 restoration closed after FIT4, 2026-10-01.

Candidate 802fb36745d189cb0adfa101c1ae6c192f183098bcbbc2375059fac49747982f restored only the covariance correction's historical correlation-order2 term. The main pair map remained E1. This was a deliberate partial restoration experiment, not an accidental-omission bug fix; the historical E2→E1 change removed both terms together.

Tiny independent algebra passed for each channel and their sum; zero-r2 output matched the parent byte for byte. Independent source review and root preflight passed.

Actual matched native FIT4: raw error +0.1826703995%, adjusted error +0.2140035000%, bill +0.0311068413%, 2/4 adjusted wins. Both accuracy discovery gates failed. All16 finite/L0 parity and source/control integrity passed.

Assessment SHA256 d8916e93288bc56bdc1171f9cc92d0d01d82d32660dbeafa030a24caa339e3c4. Native guard retired exit0 after107.0145356s, peakRSS5971623936B, zero added swap. Assessment guard retired exit0. No CHECK8, retained component, wider qualification, or bank. Official S35 and research RRPI remain unchanged.
