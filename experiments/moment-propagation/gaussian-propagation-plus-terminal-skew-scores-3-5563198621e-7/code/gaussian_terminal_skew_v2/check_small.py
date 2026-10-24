"""Inert scalar-gated replay of the identical frozen v1 tiny tensor fixture."""
import sys
import common as h


if __name__ == '__main__':
    if '--execute' in sys.argv:
        from check_scalar import verify_scalar
        verify_scalar(h.sources())
    raise SystemExit(h.template('check_small.py').main())
