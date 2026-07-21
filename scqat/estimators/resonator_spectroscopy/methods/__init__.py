"""Method strategy registry for the resonator_spectroscopy estimator.

Adding a new extraction approach = one module implementing
:class:`~scqat.estimators.resonator_spectroscopy.methods.base.ResonatorMethod`
+ one entry here. Nothing else moves.
"""

from .base import ResonatorMethod
from .lorentzian import LorentzianMethod
from .circle import CircleMethod

METHODS = {m.name: m for m in (LorentzianMethod(), CircleMethod())}

__all__ = ["ResonatorMethod", "LorentzianMethod", "CircleMethod", "METHODS"]
