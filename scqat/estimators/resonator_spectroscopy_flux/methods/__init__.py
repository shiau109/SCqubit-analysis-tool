"""Method strategy registry for the resonator_spectroscopy_flux trace fit.

Adding a new extraction approach = one module implementing
:class:`~scqat.estimators.resonator_spectroscopy_flux.methods.base.FluxModel`
+ one entry here. Nothing else moves.
"""

from .base import FluxModel
from .dispersive import DispersiveMethod
from .sine import SineMethod

METHODS = {m.name: m for m in (DispersiveMethod(), SineMethod())}

__all__ = ["FluxModel", "DispersiveMethod", "SineMethod", "METHODS"]
