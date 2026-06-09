"""Enable ``python -m akkadian_mt`` as an alias for the ``akkadian-mt`` CLI."""

from akkadian_mt.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
