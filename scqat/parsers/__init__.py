"""parsers package — file readers that convert raw files to ``xarray.Dataset``.

The public functions are re-exported here so callers can use the documented API
``from scqat.parsers import ...`` (see ``.github/copilot-instructions.md``). This
layer must stay free of physics/analysis logic and import nothing from the rest
of ``scqat``.
"""

from .xarray_h5_parser import load_xarray_h5
from .qualibrate_parser import repetition_data, parse_timestamp

__all__ = ["load_xarray_h5", "repetition_data", "parse_timestamp"]
