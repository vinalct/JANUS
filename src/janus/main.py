"""Process entry point. The verb table lives in `janus.cli.dispatch`; this module exists
so `python -m janus.main` and the `janus` console script keep their import path."""

from janus.cli.dispatch import main

if __name__ == "__main__":
    raise SystemExit(main())
