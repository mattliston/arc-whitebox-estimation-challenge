# V2 test-reference buffer repair

V1 unit stopped before comparing its first value: NumPy2.4 diagonal returns
an immutable view, and the independent dense reference used in-place scaling.
Replace only that reference assignment with an out-of-place multiplication.
No native feature, formula, coefficient, data roster, gate, fit or tolerance
changes. V1 source seal, log and terminal failed guard are preserved.
No capture, target-array access or fit occurred under V1.
All original protocol requirements apply unchanged to V2.
