"""Snapshot-based data commands. Run without arguments to display usage."""

import sys

from regime_research.data.__main__ import main


if __name__ == "__main__":
    main(sys.argv[1:] or ["--help"])
