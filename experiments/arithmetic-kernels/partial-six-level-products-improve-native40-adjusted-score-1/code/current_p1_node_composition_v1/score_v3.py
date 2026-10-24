"""Additive source-owner repair; every score computation stays frozen."""
from pathlib import Path
import endpoint_v2 as e

SCORE_V2_SHA = 'caf8ba0dca7a1b03a8db47c6a8d4ddde98b3e1705604ac08934b63fff1105229'
SCORE_ENDPOINT_SHA = '7195e73f3f8f795602b0845ffab049467619946e36bf3b4817d36b8d06e0b58d'
SCORE_SOURCE_V3_SHA = '1f2f70b11a40b3e1b869c738bee43db39565e2ff1ee80d2f03fa9ea3dddd3459'
e.require(e.sha(e.HERE/'score_v2.py') == SCORE_V2_SHA, 'Preserved source-only scorer draft')
_v3_owner = (e.HERE/'score_v2.py').read_text()
_v3_edits = [
    ('_extra_sources = (\n', '_extra_sources = (\n'
     '    "        str(e.HELPER):e.HELPER_SHA,\\n"\n'
     '    "        str(HERE/\'score_v2.py\'):SCORE_V2_SHA,\\n"\n'
     '    "        str(HERE/\'SCORE_ENDPOINT_V2.md\'):SCORE_ENDPOINT_SHA,\\n"\n'
     '    "        str(HERE/\'SCORE_SOURCE_V3.md\'):SCORE_SOURCE_V3_SHA,\\n"\n'),
    ("out = HERE/'score_v2'", "out = HERE/'score_v3'"),
    ("schema='current_p1_node_composition_score_v2'", "schema='current_p1_node_composition_score_v3'"),
    ("if __name__ == '__main__':\n    raise SystemExit(main())\n",
     '# Source-v2 entrypoint suppressed; this version owns the explicit entrypoint.\n'),
]
_v3_derived = e.splice(_v3_owner, _v3_edits)
exec(compile(_v3_derived, str(Path(__file__).resolve())+':source_owner_repair', 'exec'), globals())


if __name__ == '__main__':
    raise SystemExit(main())
