

# BEGIN FABLE-ENGINEER BIG ARENA (E-21, 2026-09-22): the arena grows by 99,025,920 floats (396 MB of address space; only the part
# the six-level pushes touch becomes resident, ~482 MB at most) so the six-level push's leaf products fit in the free tail and
# the grouped kernel's last-level copy elimination also applies to the D6 pushes (their layout reuses the level-5 regions for the
# leaf). Arithmetic unchanged (bit-exact); bill -(3/7 of the D6 pushes' last-level slot writes).
_PACKED_ELEMENTS = 391602176
# END FABLE-ENGINEER BIG ARENA
