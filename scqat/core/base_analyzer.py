import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _json_safe(obj: Any) -> Any:
    """
    Recursively convert ``obj`` into something ``json.dump`` can serialize.

    numpy scalars/arrays become Python scalars/lists, complex numbers become
    ``{"real": ..., "imag": ...}``, and objects that cannot be represented as
    plain metadata (e.g. ``xarray`` containers, lmfit results) are dropped with
    a short ``"<skipped: type>"`` marker so the metadata file never fails to
    write.  Bulky arrays belong in the plot-data Dataset, not here.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (complex, np.complexfloating)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (xr.Dataset, xr.DataArray)):
        return f"<skipped: {type(obj).__name__} — belongs in plot_data>"
    return f"<skipped: {type(obj).__name__}>"


class BaseAnalyzer(ABC):
    """
    Abstract base class for scqat experimental/simulation protocols.

    Enforces a strict separation of Data Checking, Math, Plot-data extraction,
    Visualization, and I/O.  Each analyzer produces two distinct artifacts:

    * **metadata** — the key physical parameters (returned by
      :meth:`extract_parameters`), saved as ``<protocol_name>_metadata.json``.
    * **plot data** — the minimal arrays needed to redraw every figure with no
      recalculation (returned by :meth:`build_plot_data`), saved as
      ``<protocol_name>_plotdata.nc``.

    Subclasses must define ``protocol_name`` (str) to control default output
    filenames when ``output_dir`` is used.
    """

    protocol_name: str = "protocol"

    def _check_data(self, dataset: xr.Dataset) -> None:
        """
        Optional data validation step.
        Override this in your subclass to check for required coordinates/variables.
        """
        pass

    @abstractmethod
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Step 1: The heavy calculation. Must return the key-parameter (metadata) dict."""
        pass

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """
        Step 2: Assemble the minimal arrays needed to redraw every figure
        without any recalculation, as a single ``xarray.Dataset``.

        Default returns ``None`` (no plot-data artifact). Override to provide a
        self-sufficient Dataset; :meth:`generate_figures` should then draw using
        only this Dataset.
        """
        return None

    @abstractmethod
    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """
        Step 3: The visualization. Must return a dict of figures.

        Migrated protocols MUST draw using **only** ``plot_data`` so the figures
        stay reconstructable by an external consumer; ``dataset`` and ``results``
        are still passed for protocols that have not yet been migrated to the
        plot-data contract and must not be relied on by new code.
        """
        pass

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    def save_metadata(self, results: Dict[str, Any], output_dir: str) -> None:
        """Save the key parameters as ``<output_dir>/<protocol_name>_metadata.json``."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{self.protocol_name}_metadata.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(_json_safe(results), f, indent=2)

    def load_metadata(self, output_dir: str) -> Dict[str, Any]:
        """Load the key parameters from ``<output_dir>/<protocol_name>_metadata.json``."""
        filepath = os.path.join(output_dir, f"{self.protocol_name}_metadata.json")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_plot_data(self, plot_data: Optional[xr.Dataset], output_dir: str) -> None:
        """Save the plot-reconstruction Dataset as ``<output_dir>/<protocol_name>_plotdata.nc``."""
        if plot_data is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{self.protocol_name}_plotdata.nc")
        plot_data.to_netcdf(filepath)

    def load_plot_data(self, output_dir: str) -> xr.Dataset:
        """Load the plot-reconstruction Dataset from ``<output_dir>/<protocol_name>_plotdata.nc``."""
        filepath = os.path.join(output_dir, f"{self.protocol_name}_plotdata.nc")
        return xr.load_dataset(filepath)

    def save_figures(self, figs: Dict[str, plt.Figure], output_dir: str) -> None:
        """Saves figures as ``<output_dir>/<protocol_name>_<fig_name>.png``."""
        os.makedirs(output_dir, exist_ok=True)
        for name, fig in figs.items():
            filepath = os.path.join(output_dir, f"{self.protocol_name}_{name}.png")
            fig.savefig(filepath, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def analyze(
        self,
        dataset: xr.Dataset,
        output_dir: str = None,
        skip_figures: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, plt.Figure]]:
        """
        The Orchestrator.

        Calls Data Checking -> Math (metadata) -> Plot-data build ->
        Metadata/Plot-data I/O -> Plotting -> Figure I/O.

        Args:
            dataset: The input xarray Dataset.
            output_dir: Directory path for saving metadata, plot data, and
                figures. If None, nothing is saved.
            skip_figures: If True, skip figure generation and return empty dict.

        Returns:
            ``(metadata, figures)``. The plot-data Dataset is saved (when
            ``output_dir`` is given) and passed to ``generate_figures``; retrieve
            it via :meth:`build_plot_data` or :meth:`load_plot_data` if needed.
        """
        # 1. Input checking
        self._check_data(dataset)

        # 2. Heavy physics calculation -> key parameters (metadata)
        results = self.extract_parameters(dataset, **kwargs)

        # 3. Minimal arrays needed to redraw the figures (plot data)
        plot_data = self.build_plot_data(dataset, results, **kwargs)

        # 4. Save metadata + plot data if requested
        if output_dir:
            self.save_metadata(results, output_dir)
            self.save_plot_data(plot_data, output_dir)

        if skip_figures:
            return results, {}

        # 5. Generate figures. Migrated protocols use only plot_data; dataset and
        #    results remain available for not-yet-migrated protocols.
        figs = self.generate_figures(dataset, results, plot_data=plot_data, **kwargs)

        # 6. Save figures if requested
        if output_dir:
            self.save_figures(figs, output_dir)

        return results, figs
