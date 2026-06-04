"""Shared pytest configuration.

Force the non-interactive ``Agg`` matplotlib backend for the whole test
session so figure-generating code never tries to open a GUI window.  This keeps
the suite reproducible on headless machines / CI and avoids Tcl/Tk errors when
the active environment lacks a working interactive backend.
"""

import matplotlib

matplotlib.use("Agg")
